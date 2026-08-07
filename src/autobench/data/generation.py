from __future__ import annotations as _annotations

import json
from collections.abc import Awaitable
from datetime import UTC, datetime
from enum import StrEnum
from hashlib import sha256
from inspect import isawaitable
from pathlib import Path
from typing import Any, Protocol, cast

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from autobench.data.datasets import (
    Case,
    DatasetSpec,
    case_to_yaml_view,
    dataset_content_hash,
    dataset_to_yaml_view,
)
from autobench.data.ingestion import ReviewStatus
from autobench.errors import GenerationError
from autobench.io import dump_yaml, load_yaml
from autobench.protocol.values import SerializedValue
from autobench.records.files import RecordDurability, atomic_write_text
from autobench.runtime.awaitables import run_sync
from autobench.runtime.tasks import resolve_python_callable

GENERATION_RECORD_VERSION = 1


class GenerationDeterminism(StrEnum):
    GUARANTEED = "guaranteed"
    NOT_GUARANTEED = "not_guaranteed"
    UNKNOWN = "unknown"


class GenerationUsage(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    input_tokens: int = Field(default=0, ge=0)
    output_tokens: int = Field(default=0, ge=0)
    cached_input_tokens: int = Field(default=0, ge=0)
    requests: int = Field(default=0, ge=0)
    metadata: dict[str, SerializedValue] = Field(default_factory=dict)


class GenerationCost(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False)

    amount: float = Field(ge=0)
    currency: str = Field(default="usd", min_length=1)


class GeneratedCaseReview(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    case_id: str = Field(min_length=1)
    status: ReviewStatus = ReviewStatus.CANDIDATE
    rejection_reason: str | None = Field(default=None, min_length=1)

    @model_validator(mode="after")
    def validate_rejection(self) -> GeneratedCaseReview:
        if self.status is ReviewStatus.REJECTED and self.rejection_reason is None:
            raise ValueError("rejected generated cases require a rejection reason")
        if self.status is not ReviewStatus.REJECTED and self.rejection_reason is not None:
            raise ValueError("only rejected generated cases may have a rejection reason")
        return self


class GeneratedCaseBatch(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    generator_asset_version: str | None = None
    model_provider: str | None = None
    model_name: str | None = None
    determinism: GenerationDeterminism = GenerationDeterminism.UNKNOWN
    usage: GenerationUsage = Field(default_factory=GenerationUsage)
    cost: GenerationCost | None = None
    reviews: tuple[GeneratedCaseReview, ...] = ()
    complete: bool = True
    incomplete_reason: str | None = Field(default=None, min_length=1)
    cases: tuple[Case, ...] = ()

    @model_validator(mode="after")
    def validate_batch(self) -> GeneratedCaseBatch:
        case_ids = tuple(case.id for case in self.cases)
        if len(case_ids) != len(set(case_ids)):
            raise ValueError("generated case ids must be unique")
        review_ids = tuple(review.case_id for review in self.reviews)
        if len(review_ids) != len(set(review_ids)):
            raise ValueError("generated case reviews must be unique")
        unknown = sorted(set(review_ids).difference(case_ids))
        if unknown:
            raise ValueError(
                f"generated case reviews reference unknown cases: {', '.join(unknown)}"
            )
        if self.complete and self.incomplete_reason is not None:
            raise ValueError("complete generation batches cannot have an incomplete reason")
        if not self.complete and self.incomplete_reason is None:
            raise ValueError("incomplete generation batches require a reason")
        return self


class CaseGeneratorInput(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    seed_cases: tuple[Case, ...] = ()
    prompt: str | None = None
    prompt_asset_version: str | None = None
    seed: int | str | None = None
    settings: dict[str, SerializedValue] = Field(default_factory=dict)
    metadata: dict[str, SerializedValue] = Field(default_factory=dict)


class CaseGenerator(Protocol):
    def __call__(
        self,
        request: CaseGeneratorInput,
        /,
    ) -> GeneratedCaseBatch | Awaitable[GeneratedCaseBatch]: ...


class GeneratedCaseRecord(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    case: Case
    review_status: ReviewStatus
    rejection_reason: str | None = None
    content_hash: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def validate_review(self) -> GeneratedCaseRecord:
        if self.review_status is ReviewStatus.REJECTED and self.rejection_reason is None:
            raise ValueError("rejected generated case records require a rejection reason")
        if self.review_status is not ReviewStatus.REJECTED and self.rejection_reason is not None:
            raise ValueError("only rejected generated case records may have a rejection reason")
        if generated_case_content_hash(self.case) != self.content_hash:
            raise ValueError("generated case record content hash does not match its case")
        return self


class GenerationResult(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    generator_id: str = Field(min_length=1)
    started_at: datetime
    completed_at: datetime
    request: CaseGeneratorInput
    request_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    batch: GeneratedCaseBatch
    generated_cases: tuple[GeneratedCaseRecord, ...]
    dataset: DatasetSpec | None = None
    dataset_hash: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def validate_completion(self) -> GenerationResult:
        if self.completed_at < self.started_at:
            raise ValueError("generation completion cannot precede its start")
        if generation_request_hash(self.request) != self.request_hash:
            raise ValueError("generation request hash does not match its request")

        batch_case_ids = tuple(case.id for case in self.batch.cases)
        record_case_ids = tuple(record.case.id for record in self.generated_cases)
        if record_case_ids != batch_case_ids:
            raise ValueError("generated case records must match batch cases in order")
        if any(
            record.case != batch_case
            for record, batch_case in zip(self.generated_cases, self.batch.cases, strict=True)
        ):
            raise ValueError("generated case records must contain the batch case payloads")

        if self.batch.complete and (self.dataset is None or self.dataset_hash is None):
            raise ValueError("complete generation requires a frozen dataset")
        if not self.batch.complete and (self.dataset is not None or self.dataset_hash is not None):
            raise ValueError("incomplete generation cannot publish a benchmark dataset")
        if self.dataset is not None:
            included_cases = [
                record.case
                for record in self.generated_cases
                if record.review_status is not ReviewStatus.REJECTED
            ]
            if self.dataset.cases != included_cases:
                raise ValueError(
                    "generated dataset must contain every non-rejected case in generation order"
                )
            if dataset_content_hash(self.dataset) != self.dataset_hash:
                raise ValueError("generated dataset hash does not match its dataset")
        return self


class GenerationWriteResult(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    complete: bool
    dataset_path: Path | None = None
    manifest_path: Path


def mark_generated_case(
    case: Case,
    *,
    generator_asset_version: str | None = None,
    model_provider: str | None = None,
    model_name: str | None = None,
    review_status: ReviewStatus = ReviewStatus.CANDIDATE,
    rejection_reason: str | None = None,
) -> Case:
    metadata = dict(case.metadata)
    metadata["source"] = "synthetic"
    metadata["review_status"] = review_status.value
    if rejection_reason is not None:
        metadata["rejection_reason"] = rejection_reason
    else:
        metadata.pop("rejection_reason", None)
    if generator_asset_version is not None:
        metadata["generator_asset_version"] = generator_asset_version
    if model_provider is not None:
        metadata["model_provider"] = model_provider
    if model_name is not None:
        metadata["model_name"] = model_name
    return case.model_copy(update={"metadata": metadata})


def generated_batch_from_cases(
    cases: list[Case],
    *,
    generator_asset_version: str | None = None,
    model_provider: str | None = None,
    model_name: str | None = None,
    determinism: GenerationDeterminism = GenerationDeterminism.UNKNOWN,
    usage: GenerationUsage | None = None,
    cost: GenerationCost | None = None,
) -> GeneratedCaseBatch:
    marked_cases = tuple(
        mark_generated_case(
            case,
            generator_asset_version=generator_asset_version,
            model_provider=model_provider,
            model_name=model_name,
        )
        for case in cases
    )
    return GeneratedCaseBatch(
        generator_asset_version=generator_asset_version,
        model_provider=model_provider,
        model_name=model_name,
        determinism=determinism,
        usage=usage or GenerationUsage(),
        cost=cost,
        cases=marked_cases,
    )


async def generate_dataset(
    generator: CaseGenerator,
    request: CaseGeneratorInput,
    *,
    generator_id: str,
    dataset_id: str,
    version: str | None = None,
    metadata: dict[str, SerializedValue] | None = None,
) -> GenerationResult:
    if not generator_id.strip():
        raise GenerationError("Generator id must not be blank.")
    if not dataset_id.strip():
        raise GenerationError("Generated dataset id must not be blank.")
    started_at = datetime.now(UTC)
    try:
        generated = generator(request)
        batch = await generated if isawaitable(generated) else generated
    except Exception as exc:
        raise GenerationError(f"Case generator {generator_id!r} failed: {exc}") from exc
    if not isinstance(batch, GeneratedCaseBatch):
        raise GenerationError("Case generators must return GeneratedCaseBatch.")

    reviews = {review.case_id: review for review in batch.reviews}
    normalized_cases: list[Case] = []
    generated_records: list[GeneratedCaseRecord] = []
    for case in batch.cases:
        review = reviews.get(case.id)
        if review is None:
            review = _review_from_case(case)
        normalized = mark_generated_case(
            case,
            generator_asset_version=batch.generator_asset_version,
            model_provider=batch.model_provider,
            model_name=batch.model_name,
            review_status=review.status,
            rejection_reason=review.rejection_reason,
        )
        content_hash = generated_case_content_hash(normalized)
        case_metadata = dict(normalized.metadata)
        case_metadata["content_hash"] = content_hash
        normalized = normalized.model_copy(update={"metadata": case_metadata})
        generated_records.append(
            GeneratedCaseRecord(
                case=normalized,
                review_status=review.status,
                rejection_reason=review.rejection_reason,
                content_hash=content_hash,
            )
        )
        if review.status is not ReviewStatus.REJECTED:
            normalized_cases.append(normalized)

    normalized_batch = batch.model_copy(
        update={"cases": tuple(record.case for record in generated_records)}
    )
    request_hash = generation_request_hash(request)
    dataset: DatasetSpec | None = None
    frozen_hash: str | None = None
    if normalized_batch.complete:
        dataset_metadata = dict(metadata or {})
        dataset_metadata["generation"] = {
            "generator": generator_id,
            "request_hash": request_hash,
            "determinism": normalized_batch.determinism.value,
        }
        dataset = DatasetSpec(
            id=dataset_id,
            version=version,
            metadata=dataset_metadata,
            cases=normalized_cases,
        )
        frozen_hash = dataset_content_hash(dataset)

    return GenerationResult(
        generator_id=generator_id,
        started_at=started_at,
        completed_at=datetime.now(UTC),
        request=request,
        request_hash=request_hash,
        batch=normalized_batch,
        generated_cases=tuple(generated_records),
        dataset=dataset,
        dataset_hash=frozen_hash,
    )


def generate_dataset_sync(
    generator: CaseGenerator,
    request: CaseGeneratorInput,
    *,
    generator_id: str,
    dataset_id: str,
    version: str | None = None,
    metadata: dict[str, SerializedValue] | None = None,
) -> GenerationResult:
    return run_sync(
        generate_dataset(
            generator,
            request,
            generator_id=generator_id,
            dataset_id=dataset_id,
            version=version,
            metadata=metadata,
        )
    )


def resolve_case_generator(
    target: str,
    *,
    search_paths: tuple[str, ...] = (),
) -> CaseGenerator:
    return cast(CaseGenerator, resolve_python_callable(target, search_paths=search_paths))


def generation_request_hash(request: CaseGeneratorInput) -> str:
    return _content_hash(request.model_dump(mode="json", exclude_none=True))


def generated_case_content_hash(case: Case) -> str:
    payload = case.model_dump(mode="json", exclude_none=True)
    metadata = dict(payload.get("metadata", {}))
    metadata.pop("content_hash", None)
    payload["metadata"] = metadata
    return _content_hash(payload)


def generation_request_to_yaml_view(request: CaseGeneratorInput) -> dict[str, Any]:
    prompt = None
    if request.prompt is not None or request.prompt_asset_version is not None:
        prompt = {
            "content": request.prompt,
            "asset_version": request.prompt_asset_version,
        }
    request_view = {
        "seed": request.seed,
        "prompt": prompt,
        "settings": request.settings or None,
        "metadata": request.metadata or None,
        "seed_cases": [case_to_yaml_view(case) for case in request.seed_cases] or None,
    }
    return {
        "generation": {
            "request": {key: value for key, value in request_view.items() if value is not None}
        }
    }


def generation_request_from_yaml_view(raw: Any) -> CaseGeneratorInput:
    if not isinstance(raw, dict):
        raise GenerationError("Generation request YAML must contain a mapping.")
    generation = raw.get("generation", raw)
    if not isinstance(generation, dict):
        raise GenerationError("generation must be a mapping.")
    request = generation.get("request", generation)
    if not isinstance(request, dict):
        raise GenerationError("generation.request must be a mapping.")
    payload = dict(request)
    prompt = payload.pop("prompt", None)
    if prompt is not None:
        if not isinstance(prompt, dict):
            raise GenerationError("generation.request.prompt must be a mapping.")
        payload["prompt"] = prompt.get("content")
        payload["prompt_asset_version"] = prompt.get("asset_version")
    try:
        return CaseGeneratorInput.model_validate(payload)
    except ValidationError as exc:
        raise GenerationError(f"Invalid generation request: {exc}") from exc


def load_generation_request(path: Path) -> CaseGeneratorInput:
    return generation_request_from_yaml_view(load_yaml(path))


def generation_result_to_yaml_view(
    result: GenerationResult,
    *,
    dataset_path: str | None = None,
) -> dict[str, Any]:
    prompt_view = None
    if result.request.prompt is not None or result.request.prompt_asset_version is not None:
        prompt_view = {
            "sha256": (
                None
                if result.request.prompt is None
                else sha256(result.request.prompt.encode("utf-8")).hexdigest()
            ),
            "asset_version": result.request.prompt_asset_version,
        }
    output_view = {
        "dataset": (
            None
            if result.dataset is None
            else {
                "id": result.dataset.id,
                "version": result.dataset.version,
                "sha256": result.dataset_hash,
                "path": dataset_path,
            }
        ),
        "generated": len(result.generated_cases),
        "included": 0 if result.dataset is None else len(result.dataset.cases),
        "rejected": sum(
            1
            for generated_case in result.generated_cases
            if generated_case.review_status is ReviewStatus.REJECTED
        ),
    }
    return {
        "record": {"type": "generation", "version": GENERATION_RECORD_VERSION},
        "generation": {
            "status": "complete" if result.batch.complete else "incomplete",
            "reason": result.batch.incomplete_reason,
            "started_at": result.started_at.isoformat(),
            "completed_at": result.completed_at.isoformat(),
            "determinism": result.batch.determinism.value,
            "generator": {
                "id": result.generator_id,
                "asset_version": result.batch.generator_asset_version,
                "provider": result.batch.model_provider,
                "model": result.batch.model_name,
            },
            "request": {
                "sha256": result.request_hash,
                "seed": result.request.seed,
                "prompt": prompt_view,
                "settings": result.request.settings or None,
                "metadata": result.request.metadata or None,
                "seed_cases": [case_to_yaml_view(case) for case in result.request.seed_cases],
            },
            "usage": result.batch.usage.model_dump(mode="json"),
            "cost": (
                None if result.batch.cost is None else result.batch.cost.model_dump(mode="json")
            ),
            "output": output_view,
            "cases": [
                {
                    "id": generated_case.case.id,
                    "status": generated_case.review_status.value,
                    "rejection_reason": generated_case.rejection_reason,
                    "sha256": generated_case.content_hash,
                    "case": case_to_yaml_view(generated_case.case),
                }
                for generated_case in result.generated_cases
            ],
        },
    }


def write_generation_result(
    result: GenerationResult,
    output_path: Path,
    *,
    force: bool = False,
    durability: RecordDurability = "atomic",
) -> GenerationWriteResult:
    manifest_path = output_path.with_name(
        f"{output_path.stem}.generation.yaml"
        if result.batch.complete
        else f"{output_path.stem}.incomplete.yaml"
    )
    targets = [manifest_path]
    if result.batch.complete:
        targets.append(output_path)
    elif output_path.exists():
        raise GenerationError(
            f"Incomplete generation did not replace existing dataset: {output_path}"
        )
    existing = [path for path in targets if path.exists()]
    if existing and not force:
        raise GenerationError(f"Generation output already exists: {existing[0]}")

    manifest = generation_result_to_yaml_view(
        result,
        dataset_path=output_path.name if result.batch.complete else None,
    )
    atomic_write_text(
        manifest_path,
        dump_yaml(_compact(manifest), schema_name="generation"),
        durability=durability,
    )
    if result.batch.complete:
        if result.dataset is None:
            raise GenerationError("Complete generation result is missing its dataset.")
        atomic_write_text(
            output_path,
            dump_yaml(dataset_to_yaml_view(result.dataset), schema_name="dataset"),
            durability=durability,
        )
    return GenerationWriteResult(
        complete=result.batch.complete,
        dataset_path=output_path if result.batch.complete else None,
        manifest_path=manifest_path,
    )


def _review_from_case(case: Case) -> GeneratedCaseReview:
    raw_status = case.metadata.get("review_status", ReviewStatus.CANDIDATE.value)
    try:
        status = ReviewStatus(raw_status)
    except (TypeError, ValueError) as exc:
        raise GenerationError(
            f"Generated case {case.id!r} has invalid review_status {raw_status!r}."
        ) from exc
    reason = case.metadata.get("rejection_reason")
    if reason is not None and not isinstance(reason, str):
        raise GenerationError(f"Generated case {case.id!r} rejection_reason must be a string.")
    return GeneratedCaseReview(case_id=case.id, status=status, rejection_reason=reason)


def _content_hash(value: Any) -> str:
    rendered = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return sha256(rendered.encode("utf-8")).hexdigest()


def _compact(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _compact(item) for key, item in value.items() if item is not None}
    if isinstance(value, list):
        return [_compact(item) for item in value]
    return value


__all__ = (
    "CaseGenerator",
    "CaseGeneratorInput",
    "GeneratedCaseBatch",
    "GeneratedCaseRecord",
    "GeneratedCaseReview",
    "GenerationCost",
    "GenerationDeterminism",
    "GenerationResult",
    "GenerationUsage",
    "GenerationWriteResult",
    "generate_dataset",
    "generate_dataset_sync",
    "generated_batch_from_cases",
    "generated_case_content_hash",
    "generation_request_from_yaml_view",
    "generation_request_hash",
    "generation_request_to_yaml_view",
    "generation_result_to_yaml_view",
    "load_generation_request",
    "mark_generated_case",
    "resolve_case_generator",
    "write_generation_result",
)
