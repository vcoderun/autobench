from __future__ import annotations as _annotations

from collections.abc import Collection, Mapping
from pathlib import Path
from typing import Any, Protocol

from pydantic import BaseModel, Field

from autobench.data.datasets import Case, CaseDefaults, DatasetSpec
from autobench.data.variants import Variant, normalize_variant_factors
from autobench.evaluation.derivation import DeriverSpec
from autobench.evaluation.scoring import ScoringSpec
from autobench.instrumentation.config import (
    AssetDiscoverySettings,
    AutoInstrumentation,
    HTTPXInstrumentation,
    InstrumentationConfig,
    InstrumentorName,
    OpenAIAgentsInstrumentation,
    OpenAIInstrumentation,
    PydanticAIInstrumentation,
)
from autobench.instrumentation.models import Instrumentor
from autobench.protocol.capture import CapturePolicy
from autobench.runtime.awaitables import run_sync
from autobench.runtime.pipeline import ExperimentResult
from autobench.spec import BenchmarkInfo, BenchmarkSpec, TaskSpec


class Component(Protocol):  # pragma: no cover
    id: str
    kind: str


class Stage(Protocol):  # pragma: no cover
    id: str
    consumes: set[str]
    produces: set[str]

    async def run(self, ctx: BenchContext) -> None: ...


class BenchContext(BaseModel):
    values: dict[str, Any] = Field(default_factory=dict)

    def set(self, key: str, value: Any) -> None:
        self.values[key] = value

    def get(self, key: str) -> Any:
        return self.values[key]


class Benchmark:
    def __init__(self, benchmark_id: str) -> None:
        self._benchmark = BenchmarkInfo(id=benchmark_id)
        self._capture: CapturePolicy | None = None
        self._dataset = DatasetSpec()
        self._task: TaskSpec | None = None
        self._variants: list[Variant] = []
        self._scoring: list[ScoringSpec] = []
        self._derive: list[DeriverSpec] = []
        self._instrumentation: list[InstrumentationConfig] = []
        self._instrumentors: list[Instrumentor] = []

    def description(self, value: str) -> Benchmark:
        self._benchmark = self._benchmark.model_copy(update={"description": value})
        return self

    def capture(self, policy: CapturePolicy | Mapping[str, Any]) -> Benchmark:
        self._capture = (
            policy if isinstance(policy, CapturePolicy) else CapturePolicy.model_validate(policy)
        )
        return self

    def dataset(
        self,
        cases: list[Case | dict[str, Any]] | None = None,
        *,
        source: str | Path | None = None,
        dataset_id: str | None = None,
        version: str | None = None,
        metadata: dict[str, Any] | None = None,
        case_defaults: CaseDefaults | dict[str, Any] | None = None,
    ) -> Benchmark:
        defaults = (
            case_defaults
            if isinstance(case_defaults, CaseDefaults)
            else CaseDefaults.model_validate(case_defaults or {})
        )
        self._dataset = DatasetSpec(
            id=dataset_id,
            source=str(source) if source is not None else None,
            version=version,
            metadata=metadata or {},
            cases=[
                case if isinstance(case, Case) else Case.model_validate(case)
                for case in cases or []
            ],
            case_defaults=defaults,
        )
        return self

    def variants(self, variants: list[Variant | dict[str, Any]]) -> Benchmark:
        self._variants = [_normalize_variant(variant) for variant in variants]
        return self

    def task(self, target: str | TaskSpec, *, kind: str = "python") -> Benchmark:
        self._task = target if isinstance(target, TaskSpec) else TaskSpec(kind=kind, target=target)
        return self

    def scoring(self, scoring: list[ScoringSpec]) -> Benchmark:
        self._scoring = list(scoring)
        return self

    def derive(self, derive: list[DeriverSpec]) -> Benchmark:
        self._derive = list(derive)
        return self

    def instrument(
        self,
        *instrumentation: InstrumentationConfig | Instrumentor,
    ) -> Benchmark:
        """Add serializable settings or a custom runtime instrumentor."""

        for item in instrumentation:
            if isinstance(
                item,
                (
                    AutoInstrumentation,
                    PydanticAIInstrumentation,
                    OpenAIInstrumentation,
                    OpenAIAgentsInstrumentation,
                    HTTPXInstrumentation,
                ),
            ):
                self._instrumentation.append(item)
            else:
                self._instrumentors.append(item)
        return self

    def instrument_all(
        self,
        *,
        exclude: Collection[InstrumentorName] = (),
        strict: bool = False,
        assets: AssetDiscoverySettings | Mapping[str, Any] | None = None,
    ) -> Benchmark:
        """Enable every compatible built-in instrumentor available at runtime."""

        automatic = AutoInstrumentation(
            exclude=tuple(exclude),
            strict=strict,
            assets=(None if assets is None else AssetDiscoverySettings.model_validate(assets)),
        )
        self._instrumentation = [
            automatic,
            *(
                config
                for config in self._instrumentation
                if not isinstance(config, AutoInstrumentation)
            ),
        ]
        return self

    def to_spec(self) -> BenchmarkSpec:
        return BenchmarkSpec(
            benchmark=self._benchmark,
            capture=self._capture,
            dataset=self._dataset,
            task=self._task,
            variants=self._variants,
            scoring=self._scoring,
            derive=self._derive,
            instrumentation=self._instrumentation,
        )

    async def run_async(
        self,
        *,
        experiment_id: str | None = None,
        concurrency_limit: int | None = 1,
    ) -> ExperimentResult:
        from autobench.runtime.pipeline import run_benchmark_spec

        return await run_benchmark_spec(
            self.to_spec(),
            experiment_id=experiment_id,
            concurrency_limit=concurrency_limit,
            instrumentors=self._instrumentors,
        )

    def run(
        self,
        *,
        experiment_id: str | None = None,
        concurrency_limit: int | None = 1,
    ) -> ExperimentResult:
        return run_sync(
            self.run_async(
                experiment_id=experiment_id,
                concurrency_limit=concurrency_limit,
            )
        )


def _normalize_variant(variant: Variant | dict[str, Any]) -> Variant:
    if isinstance(variant, Variant):
        return variant
    payload = dict(variant)
    payload["factors"] = normalize_variant_factors(payload.get("factors"))
    return Variant.model_validate(payload)


__all__ = (
    "BenchContext",
    "Benchmark",
    "Component",
    "Stage",
)
