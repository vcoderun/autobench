from __future__ import annotations as _annotations

from collections.abc import Callable
from math import ceil, floor
from statistics import mean, median, pstdev
from time import perf_counter
from typing import TypeAlias, TypeVar

from pydantic import BaseModel, Field, model_validator

MeasuredValue = TypeVar("MeasuredValue")
MeasurementTimer: TypeAlias = Callable[[Callable[[], MeasuredValue]], float]


class Measurement(BaseModel):
    samples_seconds: tuple[float, ...] = Field(min_length=1)
    warmup: int = Field(ge=0)
    requested_repetitions: int = Field(ge=1)
    elapsed_seconds: float = Field(ge=0.0)
    timed_out: bool = False

    @model_validator(mode="after")
    def _validate_samples(self) -> Measurement:
        if any(sample < 0.0 for sample in self.samples_seconds):
            raise ValueError("measurement samples cannot be negative")
        return self

    @property
    def repetition_count(self) -> int:
        return len(self.samples_seconds)

    @property
    def samples_ms(self) -> tuple[float, ...]:
        return tuple(sample * 1000.0 for sample in self.samples_seconds)

    @property
    def median_seconds(self) -> float:
        return median(self.samples_seconds)

    @property
    def median_ms(self) -> float:
        return self.median_seconds * 1000.0

    @property
    def mean_seconds(self) -> float:
        return mean(self.samples_seconds)

    @property
    def mean_ms(self) -> float:
        return self.mean_seconds * 1000.0

    @property
    def min_seconds(self) -> float:
        return min(self.samples_seconds)

    @property
    def min_ms(self) -> float:
        return self.min_seconds * 1000.0

    @property
    def max_seconds(self) -> float:
        return max(self.samples_seconds)

    @property
    def max_ms(self) -> float:
        return self.max_seconds * 1000.0

    @property
    def p95_ms(self) -> float:
        return self.percentile_ms(95.0)

    @property
    def standard_deviation_ms(self) -> float:
        return pstdev(self.samples_ms)

    @property
    def range_noise_pct(self) -> float | None:
        if self.median_ms == 0.0:
            return None
        return ((self.max_ms - self.min_ms) / self.median_ms) * 100.0

    def percentile_ms(self, percentile: float) -> float:
        return _percentile(self.samples_ms, percentile)

    def is_noisy(self, threshold_pct: float) -> bool:
        if threshold_pct < 0.0:
            raise ValueError("noise threshold cannot be negative")
        noise = self.range_noise_pct
        return False if noise is None else noise > threshold_pct


class MeasurementBudget(BaseModel):
    warmup: int = Field(default=0, ge=0)
    repetitions: int = Field(default=1, ge=1)
    max_seconds: float | None = Field(default=None, ge=0.0)


def perf_counter_timer(fn: Callable[[], MeasuredValue]) -> float:
    started_at = perf_counter()
    fn()
    return perf_counter() - started_at


def measure_callable(
    fn: Callable[[], MeasuredValue],
    *,
    warmup: int = 0,
    repetitions: int = 1,
    max_seconds: float | None = None,
    budget: MeasurementBudget | None = None,
    timer: MeasurementTimer[MeasuredValue] | None = None,
) -> Measurement:
    if budget is not None:
        warmup = budget.warmup
        repetitions = budget.repetitions
        max_seconds = budget.max_seconds

    if warmup < 0:
        raise ValueError("warmup cannot be negative")
    if repetitions < 1:
        raise ValueError("repetitions must be at least 1")
    if max_seconds is not None and max_seconds < 0.0:
        raise ValueError("max_seconds cannot be negative")

    active_timer = timer or perf_counter_timer
    for _ in range(warmup):
        fn()

    samples: list[float] = []
    started_at = perf_counter()
    timed_out = False
    for repetition_index in range(repetitions):
        duration_seconds = active_timer(fn)
        if duration_seconds < 0.0:
            raise ValueError("measurement timer returned a negative duration")
        samples.append(duration_seconds)

        has_more_repetitions = repetition_index < repetitions - 1
        if max_seconds is not None and has_more_repetitions:
            timed_out = perf_counter() - started_at >= max_seconds
            if timed_out:
                break

    return Measurement(
        samples_seconds=tuple(samples),
        warmup=warmup,
        requested_repetitions=repetitions,
        elapsed_seconds=perf_counter() - started_at,
        timed_out=timed_out,
    )


def _percentile(values: tuple[float, ...], percentile: float) -> float:
    if not 0.0 <= percentile <= 100.0:
        raise ValueError("percentile must be between 0 and 100")
    sorted_values = sorted(values)
    if len(sorted_values) == 1:
        return sorted_values[0]

    rank = (len(sorted_values) - 1) * (percentile / 100.0)
    lower_index = floor(rank)
    upper_index = ceil(rank)
    if lower_index == upper_index:
        return sorted_values[lower_index]

    lower_value = sorted_values[lower_index]
    upper_value = sorted_values[upper_index]
    weight = rank - lower_index
    return lower_value + (upper_value - lower_value) * weight


__all__ = (
    "Measurement",
    "MeasurementBudget",
    "MeasurementTimer",
    "measure_callable",
    "perf_counter_timer",
)
