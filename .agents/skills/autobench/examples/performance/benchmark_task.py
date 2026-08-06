from __future__ import annotations

from collections.abc import Callable
from typing import Literal

from pydantic import BaseModel, TypeAdapter

from autobench import Case, RunContext, measure_callable

SearchStrategy = Literal["linear", "indexed"]
STRATEGY_ADAPTER = TypeAdapter(SearchStrategy)


class SearchInput(BaseModel):
    size: int
    needle: int


class SearchOutput(BaseModel):
    found: bool
    median_ms: float
    p95_ms: float
    repetitions: int


def run(ctx: RunContext, case: Case) -> SearchOutput:
    sample = SearchInput.model_validate(case.input)
    strategy = STRATEGY_ADAPTER.validate_python(ctx.factor("strategy"))
    values = list(range(sample.size))
    search = _search_callable(values, sample.needle, strategy=strategy)

    with ctx.span(
        "measure_search",
        kind="workflow",
        input=sample.model_dump(),
        attributes={"strategy": strategy},
    ) as span:
        measurement = measure_callable(search, warmup=3, repetitions=25, max_seconds=2.0)
        recorded = span.record_measurement("search", measurement)
        output = SearchOutput(
            found=search(),
            median_ms=measurement.median_ms,
            p95_ms=measurement.p95_ms,
            repetitions=measurement.repetition_count,
        )
        span.set_output(output.model_dump())
        if recorded.samples_artifact is None:
            raise RuntimeError("measurement samples were not recorded")
        return output


def _search_callable(
    values: list[int],
    needle: int,
    *,
    strategy: SearchStrategy,
) -> Callable[[], bool]:
    if strategy == "indexed":
        index = set(values)
        return lambda: needle in index
    return lambda: any(value == needle for value in values)


__all__ = ("run",)
