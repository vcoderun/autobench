from __future__ import annotations as _annotations

import glob
import importlib.util
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field, ValidationError, model_validator

from autobench.data.datasets import (
    DatasetSpec,
    dataset_content_hash,
    merge_case_defaults,
)
from autobench.data.variants import Variant, normalize_variant_factors
from autobench.errors import SpecLoadError, SpecValidationError
from autobench.evaluation.comparison import PostDeriverSpec
from autobench.evaluation.derivation import DeriverSpec
from autobench.evaluation.policies import PolicySpec
from autobench.evaluation.scoring import (
    PythonScorer,
    ScoringSpec,
)
from autobench.instrumentation.config import InstrumentationConfig
from autobench.io import load_yaml, resolve_file_ref
from autobench.metrics.semantics import DEFAULT_SEMANTIC_REGISTRY, SemanticRegistry
from autobench.protocol.capture import CapturePolicy
from autobench.reports.reporting import ReportSpec
from autobench.runtime.pipeline import BenchmarkPlan

from .spec import (
    _normalize_benchmark_dsl,
    benchmark_spec_to_yaml_view,
)


class BenchmarkInfo(BaseModel):
    id: str = Field(min_length=1)
    description: str | None = None


class TaskSpec(BaseModel):
    kind: str = Field(min_length=1)
    target: str = Field(min_length=1)
    module_search_paths: tuple[str, ...] = Field(default_factory=tuple, exclude=True)


class BenchmarkSpec(BaseModel):
    benchmark: BenchmarkInfo
    capture: CapturePolicy | None = None
    dataset: DatasetSpec = Field(default_factory=DatasetSpec)
    task: TaskSpec | None = None
    variants: list[Variant] = Field(default_factory=list)
    scoring: list[ScoringSpec] = Field(default_factory=list)
    derive: list[DeriverSpec] = Field(default_factory=list)
    post_derive: list[PostDeriverSpec] = Field(default_factory=list)
    policies: list[PolicySpec] = Field(default_factory=list)
    reports: ReportSpec = Field(default_factory=ReportSpec)
    instrumentation: list[InstrumentationConfig] = Field(default_factory=list)
    semantic_registry: SemanticRegistry = Field(
        default_factory=lambda: DEFAULT_SEMANTIC_REGISTRY.model_copy(deep=True)
    )

    @model_validator(mode="after")
    def _validate_unique_ids(self) -> BenchmarkSpec:
        _validate_unique_ids([case.id for case in self.dataset.cases], kind="case")
        _validate_unique_ids([variant.id for variant in self.variants], kind="variant")
        _validate_unique_ids(
            [config.kind for config in self.instrumentation],
            kind="instrumentation",
        )
        if self.task is None and self.dataset.cases and self.variants:
            raise ValueError(
                "task is required when cases and variants are defined for a runnable benchmark"
            )
        return self


def benchmark_spec_payload_from_yaml_view(raw: Any) -> dict[str, Any]:
    if not isinstance(raw, dict):
        raise TypeError("benchmark spec snapshot must be a mapping")

    normalized = _normalize_benchmark_dsl(dict(raw))
    if "semantic_registry" in normalized:
        normalized["semantic_registry"] = _resolve_semantic_registry_section(
            normalized["semantic_registry"]
        )
    spec = BenchmarkSpec.model_validate(normalized)
    return spec.model_dump(mode="json")


def load_benchmark_spec(path: Path) -> BenchmarkSpec:
    raw = load_yaml(path)
    if raw is None:
        raw = {}
    if not isinstance(raw, dict):
        raise SpecValidationError(f"Expected mapping at top level in {path}")

    resolved_raw = _normalize_benchmark_dsl(raw)
    if "dataset" in resolved_raw:
        resolved_raw["dataset"] = _resolve_dataset_section(
            resolved_raw["dataset"],
            base_path=path,
        )
    if "variants" in resolved_raw:
        resolved_raw["variants"] = _resolve_variants_section(resolved_raw["variants"])
    if "derive" in resolved_raw:
        resolved_raw["derive"] = _resolve_derive_section(
            resolved_raw["derive"],
            base_path=path,
        )
    if "post_derive" in resolved_raw:
        resolved_raw["post_derive"] = _resolve_post_derive_section(resolved_raw["post_derive"])
    if "policies" in resolved_raw:
        resolved_raw["policies"] = _resolve_policies_section(resolved_raw["policies"])
    if "semantic_registry" in resolved_raw:
        resolved_raw["semantic_registry"] = _resolve_semantic_registry_section(
            resolved_raw["semantic_registry"]
        )

    try:
        spec = BenchmarkSpec.model_validate(resolved_raw)
    except ValidationError as exc:
        raise SpecValidationError(str(exc)) from exc

    merged_cases = [
        merge_case_defaults(case, spec.dataset.case_defaults) for case in spec.dataset.cases
    ]
    resolved_task = spec.task
    if resolved_task is not None and resolved_task.kind == "python":
        resolved_task = resolved_task.model_copy(
            update={
                "module_search_paths": _infer_module_search_paths(
                    resolved_task.target,
                    base_path=path,
                )
            }
        )
    resolved_scoring = [
        scorer.model_copy(
            update={
                "module_search_paths": _infer_module_search_paths(
                    scorer.target,
                    base_path=path,
                )
            }
        )
        if isinstance(scorer, PythonScorer)
        else scorer
        for scorer in spec.scoring
    ]
    return spec.model_copy(
        update={
            "dataset": spec.dataset.model_copy(update={"cases": merged_cases}),
            "task": resolved_task,
            "scoring": resolved_scoring,
        }
    )


def collect_benchmark_source_files(path: Path) -> tuple[Path, ...]:
    raw = load_yaml(path)
    if raw is None:
        raw = {}
    if not isinstance(raw, dict):
        raise SpecValidationError(f"Expected mapping at top level in {path}")

    source_files = [path.resolve()]
    source_files.extend(
        _collect_referenced_source_files(_normalize_benchmark_dsl(raw), base_path=path)
    )
    return tuple(_dedupe_paths(source_files))


def build_benchmark_plan(spec: BenchmarkSpec) -> BenchmarkPlan:
    case_count = len(spec.dataset.cases)
    variant_count = len(spec.variants)
    warnings: list[str] = []

    if case_count == 0:
        warnings.append("No cases defined.")
    if variant_count == 0:
        warnings.append("No variants defined.")
    if spec.task is None:
        warnings.append("No task defined.")

    return BenchmarkPlan(
        benchmark_id=spec.benchmark.id,
        dataset_id=spec.dataset.id,
        dataset_version=spec.dataset.version,
        dataset_hash=dataset_content_hash(spec.dataset),
        case_ids=tuple(case.id for case in spec.dataset.cases),
        case_count=case_count,
        variant_count=variant_count,
        planned_run_count=case_count * variant_count,
        warnings=warnings,
    )


def render_validation_summary(path: Path, spec: BenchmarkSpec) -> dict[str, Any]:
    plan = build_benchmark_plan(spec)
    return {
        "path": str(path),
        "benchmark_id": spec.benchmark.id,
        "description": spec.benchmark.description,
        "case_count": plan.case_count,
        "variant_count": plan.variant_count,
        "planned_run_count": plan.planned_run_count,
        "warnings": plan.warnings,
    }


def _resolve_dataset_section(raw_dataset: Any, *, base_path: Path) -> dict[str, Any]:
    if not isinstance(raw_dataset, dict):
        raise SpecValidationError("dataset must be a mapping")

    resolved = dict(raw_dataset)
    source = resolved.get("source")
    if source is not None:
        if not isinstance(source, str):
            raise SpecValidationError("dataset.source must be a string")
        loaded = _load_dataset_source(source, base_path=base_path)
        resolved = (
            loaded
            | {key: value for key, value in resolved.items() if key != "source"}
            | {"source": source}
        )
    return resolved


def _load_dataset_source(source: str, *, base_path: Path) -> dict[str, Any]:
    if glob.has_magic(source):
        return {"cases": _load_case_glob(source, base_path=base_path)}

    dataset_path = resolve_file_ref(source, base_path=base_path)
    loaded = load_yaml(dataset_path)
    if loaded is None:
        return {}
    if not isinstance(loaded, dict):
        raise SpecValidationError(f"Expected dataset mapping in {dataset_path}")
    return _normalize_dataset_dsl(dict(loaded))


def _load_case_glob(source: str, *, base_path: Path) -> list[Any]:
    pattern_path = resolve_file_ref(source, base_path=base_path)
    matched_paths = sorted(Path(path) for path in glob.glob(str(pattern_path)))
    if not matched_paths:
        raise SpecValidationError(f"Dataset source glob matched no files: {source}")

    cases: list[Any] = []
    for case_path in matched_paths:
        loaded = load_yaml(case_path)
        if loaded is None:
            continue
        if isinstance(loaded, list):
            cases.extend(loaded)
            continue
        if isinstance(loaded, dict):
            cases.extend(_cases_from_mapping(loaded, path=case_path))
            continue
        raise SpecValidationError(f"Expected case mapping or list in {case_path}")
    return cases


def _cases_from_mapping(loaded: dict[str, Any], *, path: Path) -> list[Any]:
    loaded = _normalize_dataset_dsl(loaded)
    raw_cases = loaded.get("cases")
    if raw_cases is not None:
        if not isinstance(raw_cases, list):
            raise SpecValidationError(f"Expected cases list in {path}")
        return list(raw_cases)
    if "id" in loaded:
        return [loaded]
    raise SpecValidationError(f"Expected case mapping or dataset mapping in {path}")


def _normalize_dataset_dsl(raw: dict[str, Any]) -> dict[str, Any]:
    raw_dataset = raw.get("dataset")
    if not isinstance(raw_dataset, dict):
        return raw
    if len(raw_dataset) != 1:
        return raw
    dataset_id, body = next(iter(raw_dataset.items()))
    if body is None:
        body = {}
    if not isinstance(body, dict):
        raise SpecValidationError("dataset.<id> must be a mapping")
    normalized: dict[str, Any] = {str(key): value for key, value in body.items() if key != "id"}
    normalized["id"] = str(dataset_id)
    if "defaults" in normalized and "case_defaults" not in normalized:
        normalized["case_defaults"] = normalized.pop("defaults")
    return normalized


def _resolve_source_paths(source: str, *, base_path: Path) -> list[Path]:
    source_path = resolve_file_ref(source, base_path=base_path)
    if not glob.has_magic(source):
        return [source_path]
    matched_paths = sorted(Path(path).resolve() for path in glob.glob(str(source_path)))
    if not matched_paths:
        raise SpecValidationError(f"Dataset source glob matched no files: {source}")
    return matched_paths


def _resolve_python_target_source_paths(target: str, *, base_path: Path) -> list[Path]:
    module_name, separator, _ = target.partition(":")
    if not separator or not module_name:
        return []
    try:
        spec = importlib.util.find_spec(module_name)
    except (ImportError, ModuleNotFoundError, ValueError):
        return _resolve_module_source_paths(module_name, base_path=base_path)
    if spec is None or spec.origin is None:
        return _resolve_module_source_paths(module_name, base_path=base_path)
    origin = Path(spec.origin)
    if not origin.exists() or not origin.is_file():
        return _resolve_module_source_paths(module_name, base_path=base_path)
    return [origin.resolve()]


def _infer_module_search_paths(target: str, *, base_path: Path) -> tuple[str, ...]:
    module_name, separator, _ = target.partition(":")
    if not separator or not module_name:
        return ()

    return tuple(str(path) for path in _candidate_module_roots(module_name, base_path=base_path))


def _resolve_module_source_paths(module_name: str, *, base_path: Path) -> list[Path]:
    module_parts = module_name.split(".")
    resolved_paths = dict.fromkeys(
        candidate_path.resolve()
        for candidate_root in _candidate_module_roots(module_name, base_path=base_path)
        for candidate_path in (
            candidate_root.joinpath(*module_parts).with_suffix(".py"),
            candidate_root.joinpath(*module_parts, "__init__.py"),
        )
        if candidate_path.exists() and candidate_path.is_file()
    )
    return list(resolved_paths)


def _candidate_module_roots(module_name: str, *, base_path: Path) -> tuple[Path, ...]:
    module_parts = module_name.split(".")
    spec_directory = base_path.resolve().parent
    resolved_roots = dict.fromkeys(
        candidate_root.resolve() for candidate_root in [spec_directory, *spec_directory.parents]
    )
    roots: list[Path] = []
    for resolved_root in resolved_roots:
        candidate_root = resolved_root
        module_file = candidate_root.joinpath(*module_parts).with_suffix(".py")
        package_init = candidate_root.joinpath(*module_parts, "__init__.py")
        if not module_file.exists() and not package_init.exists():
            continue
        roots.append(resolved_root)
    return tuple(roots)


FILE_REF_KEYS = frozenset({"source", "pricing", "file", "prompt_file", "rubric_file"})
PYTHON_TARGET_KEYS = frozenset({"target", "python_target"})


def _collect_referenced_source_files(raw: Any, *, base_path: Path) -> list[Path]:
    collected: list[Path] = []
    _visit_referenced_sources(raw, base_path=base_path, collected=collected)
    return collected


def _visit_referenced_sources(raw: Any, *, base_path: Path, collected: list[Path]) -> None:
    if isinstance(raw, dict):
        for key, value in raw.items():
            if isinstance(value, str):
                if key in FILE_REF_KEYS or key.endswith("_file"):
                    collected.extend(_resolve_file_reference_paths(value, base_path=base_path))
                if key in PYTHON_TARGET_KEYS and _looks_like_python_target(value):
                    collected.extend(
                        _resolve_python_target_source_paths(value, base_path=base_path)
                    )
            _visit_referenced_sources(value, base_path=base_path, collected=collected)
        return
    if isinstance(raw, list):
        for value in raw:
            _visit_referenced_sources(value, base_path=base_path, collected=collected)


def _resolve_file_reference_paths(value: str, *, base_path: Path) -> list[Path]:
    if glob.has_magic(value):
        return _resolve_source_paths(value, base_path=base_path)
    try:
        return [resolve_file_ref(value, base_path=base_path)]
    except (SpecLoadError, ValueError):
        return []


def _looks_like_python_target(value: str) -> bool:
    module_name, separator, attribute_name = value.partition(":")
    return bool(separator and module_name and attribute_name)


def _dedupe_paths(paths: list[Path]) -> list[Path]:
    seen: set[Path] = set()
    deduped: list[Path] = []
    for path in paths:
        normalized = path.resolve()
        if normalized in seen:
            continue
        seen.add(normalized)
        deduped.append(normalized)
    return deduped


def _resolve_variants_section(raw_variants: Any) -> list[dict[str, Any]]:
    if not isinstance(raw_variants, list):
        raise SpecValidationError("variants must be a list")

    resolved: list[dict[str, Any]] = []
    for raw_variant in raw_variants:
        if not isinstance(raw_variant, dict):
            raise SpecValidationError("variant entries must be mappings")
        payload = dict(raw_variant)
        payload["factors"] = [
            factor.model_dump() for factor in normalize_variant_factors(payload.get("factors"))
        ]
        resolved.append(payload)
    return resolved


def _resolve_semantic_registry_section(raw_registry: Any) -> dict[str, Any]:
    if not isinstance(raw_registry, dict):
        raise SpecValidationError("semantic_registry must be a mapping")

    base = DEFAULT_SEMANTIC_REGISTRY.model_dump(mode="python")
    resolved = dict(base)

    if "version" in raw_registry:
        resolved["version"] = raw_registry["version"]

    raw_aliases = raw_registry.get("aliases")
    if raw_aliases is not None:
        if not isinstance(raw_aliases, dict):
            raise SpecValidationError("semantic_registry.aliases must be a mapping")
        resolved["aliases"] = dict(base["aliases"]) | dict(raw_aliases)

    raw_types = raw_registry.get("types")
    if raw_types is not None:
        if not isinstance(raw_types, dict):
            raise SpecValidationError("semantic_registry.types must be a mapping")
        merged_types = dict(base["types"])
        for semantic_type, payload in raw_types.items():
            if not isinstance(payload, dict):
                raise SpecValidationError(
                    f"semantic_registry.types.{semantic_type} must be a mapping"
                )
            normalized_payload = dict(payload)
            if "shape" in normalized_payload and "value_shape" not in normalized_payload:
                normalized_payload["value_shape"] = normalized_payload.pop("shape")
            existing = merged_types.get(semantic_type, {"id": semantic_type})
            merged_types[semantic_type] = dict(existing) | normalized_payload
            merged_types[semantic_type].setdefault("id", semantic_type)
        resolved["types"] = merged_types

    return resolved


def _resolve_derive_section(raw_derive: Any, *, base_path: Path) -> list[dict[str, Any]]:
    if not isinstance(raw_derive, list):
        raise SpecValidationError("derive must be a list")

    resolved: list[dict[str, Any]] = []
    for raw_item in raw_derive:
        if not isinstance(raw_item, dict):
            raise SpecValidationError("derive entries must be mappings")
        payload = dict(raw_item)
        pricing = payload.get("pricing")
        if pricing is not None:
            if not isinstance(pricing, str):
                raise SpecValidationError("derive pricing must be a string")
            payload["pricing"] = str(resolve_file_ref(pricing, base_path=base_path))
        resolved.append(payload)
    return resolved


def _resolve_post_derive_section(raw_post_derive: Any) -> list[dict[str, Any]]:
    if not isinstance(raw_post_derive, list):
        raise SpecValidationError("post_derive must be a list")

    resolved: list[dict[str, Any]] = []
    for raw_item in raw_post_derive:
        if not isinstance(raw_item, dict):
            raise SpecValidationError("post_derive entries must be mappings")
        resolved.append(dict(raw_item))
    return resolved


def _resolve_policies_section(raw_policies: Any) -> list[dict[str, Any]]:
    if not isinstance(raw_policies, list):
        raise SpecValidationError("policies must be a list")

    resolved: list[dict[str, Any]] = []
    for raw_item in raw_policies:
        if not isinstance(raw_item, dict):
            raise SpecValidationError("policy entries must be mappings")
        resolved.append(dict(raw_item))
    return resolved


def _validate_unique_ids(ids: list[str], *, kind: str) -> None:
    seen: set[str] = set()
    duplicates: list[str] = []
    for item_id in ids:
        if item_id in seen:
            duplicates.append(item_id)
        seen.add(item_id)
    if duplicates:
        duplicate_str = ", ".join(sorted(set(duplicates)))
        raise ValueError(f"Duplicate {kind} ids: {duplicate_str}")


__all__ = (
    "BenchmarkInfo",
    "BenchmarkSpec",
    "TaskSpec",
    "benchmark_spec_payload_from_yaml_view",
    "benchmark_spec_to_yaml_view",
    "build_benchmark_plan",
    "collect_benchmark_source_files",
    "load_benchmark_spec",
    "render_validation_summary",
)
