from __future__ import annotations as _annotations

from collections.abc import Sequence
from math import comb, floor, fsum, isfinite, sqrt
from random import Random
from statistics import NormalDist

from autobench.reports.models import (
    ComparisonHypothesis,
    ComparisonReport,
    ComparisonReportSpec,
    HypothesisResult,
    HypothesisStatus,
    PairedBinaryTable,
    PairedDifferenceInterval,
    PairedMeanDifference,
    PairedSignTest,
)

BOOTSTRAP_RESAMPLES = 2_000
BOOTSTRAP_SEED = 0


def build_hypothesis_results(
    specs: Sequence[ComparisonReportSpec],
    comparisons: Sequence[ComparisonReport],
) -> tuple[HypothesisResult, ...]:
    """Evaluate declared comparison hypotheses with a Holm adjustment inside each family.

    A hypothesis without case-paired evidence is reported as untestable. It is not dropped from
    its family, so a missing comparison never makes the remaining tests look less corrected.

    Args:
        specs: Comparison specs in report order.
        comparisons: Comparison results aligned with ``specs``.

    Returns:
        One result per declared hypothesis, in report order.
    """
    if len(specs) != len(comparisons):
        raise ValueError("comparison specs and results must align")
    declared = [
        (spec.label(), spec.hypothesis, comparison)
        for spec, comparison in zip(specs, comparisons, strict=True)
        if spec.hypothesis is not None
    ]
    raw_p_values = [
        _hypothesis_p_value(hypothesis, comparison) for _, hypothesis, comparison in declared
    ]

    families: dict[str, list[int]] = {}
    for index, (_, hypothesis, _) in enumerate(declared):
        families.setdefault(hypothesis.family, []).append(index)
    adjusted_p_values: list[float | None] = [None] * len(declared)
    for indexes in families.values():
        # Untestable members still count toward the family size, which keeps the correction
        # conservative: their p-value is treated as one.
        values = [raw_p_values[index] for index in indexes]
        adjusted = holm_adjusted([1.0 if value is None else value for value in values])
        for index, value, adjusted_value in zip(indexes, values, adjusted, strict=True):
            adjusted_p_values[index] = None if value is None else adjusted_value

    return tuple(
        HypothesisResult(
            comparison=label,
            metric=hypothesis.metric,
            family=hypothesis.family,
            alternative=hypothesis.alternative,
            alpha=hypothesis.alpha,
            status=_hypothesis_status(adjusted_p_values[index], hypothesis.alpha),
            p_value=raw_p_values[index],
            adjusted_p_value=adjusted_p_values[index],
            family_size=len(families[hypothesis.family]),
        )
        for index, (label, hypothesis, _) in enumerate(declared)
    )


def exact_sign_test(wins: int, losses: int) -> PairedSignTest:
    """Return the exact sign test for case-paired wins and losses.

    Under the null hypothesis a discordant pair is equally likely to favor either side, so the
    candidate's win count is binomial with probability one half. Without discordant pairs both
    p-values are one.

    Args:
        wins: Pairs in which the candidate is better.
        losses: Pairs in which the baseline is better.

    Returns:
        The one-sided p-value for a better candidate and the two-sided p-value.
    """
    if wins < 0 or losses < 0:
        raise ValueError("wins and losses must be non-negative")
    discordant = wins + losses
    if discordant == 0:
        return PairedSignTest(discordant_pairs=0, candidate_better_p=1.0, two_sided_p=1.0)
    total = 2**discordant
    upper_tail = sum(comb(discordant, count) for count in range(wins, discordant + 1))
    smaller_tail = sum(comb(discordant, count) for count in range(min(wins, losses) + 1))
    return PairedSignTest(
        discordant_pairs=discordant,
        candidate_better_p=upper_tail / total,
        two_sided_p=min(1.0, 2 * smaller_tail / total),
    )


def paired_success_difference(
    table: PairedBinaryTable,
    *,
    confidence: float = 0.95,
) -> PairedDifferenceInterval | None:
    """Return Newcombe's interval for the paired difference of two success rates.

    This is method 10 of Newcombe (1998), built from Wilson score intervals and the
    correlation of the paired outcomes. It stays inside [-1, 1] and remains usable for the
    small samples and zero cells that benchmark comparisons commonly produce.

    Args:
        table: Case-paired counts of the boolean metric.
        confidence: Two-sided confidence level.

    Returns:
        The candidate minus baseline rate with its interval, or ``None`` without pairs.
    """
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must be between zero and one")
    total = table.both + table.candidate_only + table.baseline_only + table.neither
    if total == 0:
        return None
    z = NormalDist().inv_cdf(0.5 + confidence / 2)
    candidate_successes = table.both + table.candidate_only
    baseline_successes = table.both + table.baseline_only
    candidate_rate = candidate_successes / total
    baseline_rate = baseline_successes / total
    candidate_low, candidate_high = _wilson_interval(candidate_successes, total, z)
    baseline_low, baseline_high = _wilson_interval(baseline_successes, total, z)
    margins = (
        candidate_successes
        * (total - candidate_successes)
        * baseline_successes
        * (total - baseline_successes)
    )
    correlation = (
        (table.both * table.neither - table.candidate_only * table.baseline_only) / sqrt(margins)
        if margins
        else 0.0
    )
    lower_margin = sqrt(
        max(
            0.0,
            (candidate_rate - candidate_low) ** 2
            - 2 * correlation * (candidate_rate - candidate_low) * (baseline_high - baseline_rate)
            + (baseline_high - baseline_rate) ** 2,
        )
    )
    upper_margin = sqrt(
        max(
            0.0,
            (candidate_high - candidate_rate) ** 2
            - 2 * correlation * (candidate_high - candidate_rate) * (baseline_rate - baseline_low)
            + (baseline_rate - baseline_low) ** 2,
        )
    )
    estimate = candidate_rate - baseline_rate
    return PairedDifferenceInterval(
        confidence=confidence,
        estimate=estimate,
        low=estimate - lower_margin,
        high=estimate + upper_margin,
    )


def paired_mean_difference(
    differences: Sequence[float],
    *,
    confidence: float = 0.95,
    resamples: int = BOOTSTRAP_RESAMPLES,
    seed: int = BOOTSTRAP_SEED,
) -> PairedMeanDifference | None:
    """Return a BCa bootstrap interval for the mean of case-paired differences.

    Paired cases are resampled with replacement, and the percentile interval of the resampled
    means is bias-corrected and accelerated (Efron, 1987), with the acceleration taken from the
    jackknife. The generator is seeded, so the same evidence always yields the same interval.

    Args:
        differences: Candidate minus baseline value of each paired case.
        confidence: Two-sided confidence level.
        resamples: Number of bootstrap resamples.
        seed: Seed of the resampling generator.

    Returns:
        The mean difference with its interval, or ``None`` with fewer than two pairs or a
        non-finite difference.
    """
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must be between zero and one")
    if resamples < 1:
        raise ValueError("resamples must be positive")
    count = len(differences)
    if count < 2 or not all(isfinite(value) for value in differences):
        return None
    total = fsum(differences)
    estimate = total / count
    low = high = estimate
    if min(differences) != max(differences):
        generator = Random(seed)
        means = sorted(
            fsum(generator.choices(differences, k=count)) / count for _ in range(resamples)
        )
        normal = NormalDist()
        below = sum(1 for value in means if value < estimate)
        tied = sum(1 for value in means if value == estimate)
        # Ties split evenly, and the proportion stays off zero and one to keep the bias finite.
        proportion = min(max((below + tied / 2) / resamples, 0.5 / resamples), 1 - 0.5 / resamples)
        bias = normal.inv_cdf(proportion)
        leave_one_out = [(total - value) / (count - 1) for value in differences]
        centre = fsum(leave_one_out) / count
        deviations = [centre - value for value in leave_one_out]
        acceleration = fsum(value**3 for value in deviations) / (
            6 * fsum(value**2 for value in deviations) ** 1.5
        )
        tail = (1 - confidence) / 2
        low = _sorted_quantile(means, _bca_level(tail, bias, acceleration, normal))
        high = _sorted_quantile(means, _bca_level(1 - tail, bias, acceleration, normal))
    return PairedMeanDifference(
        confidence=confidence,
        estimate=estimate,
        low=low,
        high=high,
        resamples=resamples,
        seed=seed,
    )


def holm_adjusted(p_values: Sequence[float]) -> tuple[float, ...]:
    """Return Holm step-down adjusted p-values in the order they were given.

    Args:
        p_values: Raw p-values of one hypothesis family.

    Returns:
        Adjusted p-values, each at least as large as its raw value and capped at one.
    """
    if any(not 0.0 <= value <= 1.0 for value in p_values):
        raise ValueError("p-values must be between zero and one")
    order = sorted(range(len(p_values)), key=lambda index: p_values[index])
    adjusted = [0.0] * len(p_values)
    running = 0.0
    for rank, index in enumerate(order):
        running = max(running, min(1.0, (len(p_values) - rank) * p_values[index]))
        adjusted[index] = running
    return tuple(adjusted)


def _hypothesis_p_value(
    hypothesis: ComparisonHypothesis,
    comparison: ComparisonReport,
) -> float | None:
    metric = next(item for item in comparison.metric_results if item.name == hypothesis.metric)
    if metric.sign_test is None:
        return None
    if hypothesis.alternative == "candidate_better":
        return metric.sign_test.candidate_better_p
    return metric.sign_test.two_sided_p


def _hypothesis_status(adjusted_p_value: float | None, alpha: float) -> HypothesisStatus:
    if adjusted_p_value is None:
        return "untestable"
    return "supported" if adjusted_p_value <= alpha else "not_supported"


def _wilson_interval(successes: int, total: int, z: float) -> tuple[float, float]:
    rate = successes / total
    denominator = 1 + z * z / total
    centre = rate + z * z / (2 * total)
    half_width = z * sqrt(rate * (1 - rate) / total + z * z / (4 * total * total))
    return (centre - half_width) / denominator, (centre + half_width) / denominator


def _bca_level(level: float, bias: float, acceleration: float, normal: NormalDist) -> float:
    shifted = bias + normal.inv_cdf(level)
    # A strongly skewed sample can push the denominator to zero; the level then saturates at
    # the edge of the bootstrap distribution instead of wrapping around.
    return normal.cdf(bias + shifted / max(1 - acceleration * shifted, 1e-12))


def _sorted_quantile(values: Sequence[float], level: float) -> float:
    position = level * (len(values) - 1)
    lower = floor(position)
    upper = min(lower + 1, len(values) - 1)
    return values[lower] + (values[upper] - values[lower]) * (position - lower)


__all__ = (
    "BOOTSTRAP_RESAMPLES",
    "BOOTSTRAP_SEED",
    "build_hypothesis_results",
    "exact_sign_test",
    "holm_adjusted",
    "paired_mean_difference",
    "paired_success_difference",
)
