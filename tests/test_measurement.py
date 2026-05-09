from __future__ import annotations as _annotations

from collections.abc import Callable

import pytest
from pydantic import ValidationError

from autobench.evaluation.measurement import (
    Measurement,
    MeasurementBudget,
    measure_callable,
    perf_counter_timer,
)


def test_measure_callable_collects_warmups_samples_and_summary_stats() -> None:
    calls: list[str] = []
    durations = iter((0.003, 0.001, 0.002, 0.004))

    def fn() -> str:
        calls.append("called")
        return "result"

    def timer(callback: Callable[[], str]) -> float:
        assert callback() == "result"
        return next(durations)

    measurement = measure_callable(fn, warmup=2, repetitions=4, timer=timer)

    assert len(calls) == 6
    assert measurement.repetition_count == 4
    assert measurement.samples_ms == pytest.approx((3.0, 1.0, 2.0, 4.0))
    assert measurement.median_ms == pytest.approx(2.5)
    assert measurement.mean_ms == pytest.approx(2.5)
    assert measurement.min_ms == pytest.approx(1.0)
    assert measurement.max_ms == pytest.approx(4.0)
    assert measurement.p95_ms == pytest.approx(3.85)
    assert measurement.percentile_ms(0.0) == pytest.approx(1.0)
    assert measurement.standard_deviation_ms == pytest.approx(1.1180339887)
    assert measurement.range_noise_pct == pytest.approx(120.0)
    assert measurement.is_noisy(100.0) is True
    assert measurement.is_noisy(150.0) is False
    assert measurement.timed_out is False


def test_measure_callable_timeout_still_keeps_at_least_one_sample() -> None:
    calls = 0

    def fn() -> None:
        nonlocal calls
        calls += 1

    measurement = measure_callable(
        fn,
        repetitions=3,
        max_seconds=0.0,
        timer=lambda callback: _constant_timer(callback, 0.001),
    )

    assert calls == 1
    assert measurement.samples_seconds == (0.001,)
    assert measurement.percentile_ms(95.0) == pytest.approx(1.0)
    assert measurement.timed_out is True


def test_measure_callable_can_run_with_non_exhausted_time_budget() -> None:
    budget = MeasurementBudget(warmup=1, repetitions=2, max_seconds=60.0)
    calls: list[None] = []

    measurement = measure_callable(
        lambda: calls.append(None),
        budget=budget,
        timer=lambda callback: _constant_timer(callback, 0.001),
    )

    assert len(calls) == 3
    assert measurement.samples_seconds == (0.001, 0.001)
    assert measurement.timed_out is False


def test_default_timer_runs_callable_and_returns_non_negative_duration() -> None:
    calls: list[bool] = []

    def fn() -> None:
        calls.append(True)

    duration = perf_counter_timer(fn)

    assert calls == [True]
    assert duration >= 0.0


@pytest.mark.parametrize(
    ("run", "message"),
    [
        (lambda: measure_callable(lambda: None, warmup=-1), "warmup cannot be negative"),
        (lambda: measure_callable(lambda: None, repetitions=0), "repetitions must be at least 1"),
        (
            lambda: measure_callable(lambda: None, max_seconds=-0.1),
            "max_seconds cannot be negative",
        ),
    ],
)
def test_measure_callable_rejects_invalid_budget(
    run: Callable[[], Measurement],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        run()


def test_measure_callable_rejects_negative_timer_duration() -> None:
    with pytest.raises(ValueError, match="negative duration"):
        measure_callable(
            lambda: None,
            timer=lambda callback: _constant_timer(callback, -0.001),
        )


def test_measurement_rejects_bad_samples_and_bad_stat_requests() -> None:
    with pytest.raises(ValidationError, match="at least 1"):
        Measurement(samples_seconds=(), warmup=0, requested_repetitions=1, elapsed_seconds=0.0)
    with pytest.raises(ValidationError, match="cannot be negative"):
        Measurement(
            samples_seconds=(-0.001,),
            warmup=0,
            requested_repetitions=1,
            elapsed_seconds=0.0,
        )

    zero = Measurement(
        samples_seconds=(0.0,),
        warmup=0,
        requested_repetitions=1,
        elapsed_seconds=0.0,
    )
    assert zero.range_noise_pct is None
    assert zero.is_noisy(0.0) is False

    with pytest.raises(ValueError, match="noise threshold"):
        zero.is_noisy(-1.0)
    with pytest.raises(ValueError, match="percentile"):
        zero.percentile_ms(101.0)


def _constant_timer(callback: Callable[[], None], duration: float) -> float:
    callback()
    return duration
