from __future__ import annotations as _annotations

from html import escape

from autobench.reports.models import EvaluationSummaryReport

_GREEN = "#2E7D5B"
_RED = "#C44E52"
_TEAL = "#2F6F73"
_TRACK = "#E8ECEF"
_TEXT = "#213547"
_MUTED = "#68737D"


def render_quality_gate_chart(evaluation: EvaluationSummaryReport) -> str | None:
    if not evaluation.evaluated_count:
        return None
    passed_width = 720 * evaluation.passed_count / evaluation.evaluated_count
    failed_width = 720 - passed_width
    pass_rate = evaluation.pass_rate or 0
    return "\n".join(
        (
            '<svg xmlns="http://www.w3.org/2000/svg" width="920" height="150" '
            'viewBox="0 0 920 150" role="img" aria-labelledby="quality-gate-title" '
            'style="max-width:100%;height:auto">',
            '<title id="quality-gate-title">Quality gate outcome</title>',
            f'<text x="20" y="30" fill="{_TEXT}" font-family="sans-serif" '
            'font-size="18" font-weight="600">Quality gate</text>',
            f'<text x="900" y="30" text-anchor="end" fill="{_TEXT}" '
            f'font-family="sans-serif" font-size="18" font-weight="600">{pass_rate:.0%} passed</text>',
            f'<rect x="100" y="56" width="720" height="30" rx="4" fill="{_TRACK}"/>',
            f'<rect x="100" y="56" width="{passed_width:.2f}" height="30" rx="4" fill="{_GREEN}"/>',
            (
                f'<rect x="{100 + passed_width:.2f}" y="56" width="{failed_width:.2f}" '
                f'height="30" rx="4" fill="{_RED}"/>'
            ),
            f'<circle cx="230" cy="120" r="6" fill="{_GREEN}"/>',
            f'<text x="244" y="125" fill="{_MUTED}" font-family="sans-serif" '
            f'font-size="14">Passed {evaluation.passed_count}</text>',
            f'<circle cx="520" cy="120" r="6" fill="{_RED}"/>',
            f'<text x="534" y="125" fill="{_MUTED}" font-family="sans-serif" '
            f'font-size="14">Failed {evaluation.failed_count}</text>',
            "</svg>",
        )
    )


def render_case_score_chart(evaluation: EvaluationSummaryReport) -> str | None:
    scored = tuple(
        (case, score) for case in evaluation.cases for score in (case.score,) if score is not None
    )
    if not scored:
        return None
    visible = tuple(sorted(scored, key=lambda item: (-item[1], item[0].case_id))[:30])
    scores = tuple(score for _, score in visible)
    normalized = all(0 <= score <= 1 for score in scores)
    lower = 0.0 if min(scores) >= 0 else min(scores)
    upper = 1.0 if normalized else max(scores)
    if upper == lower:
        upper = lower + 1

    width = 920
    label_width = 250
    plot_width = 550
    top = 70
    row_height = 34
    height = top + len(visible) * row_height + 48
    lines = [
        (
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
            f'viewBox="0 0 {width} {height}" role="img" '
            'aria-labelledby="case-score-title" style="max-width:100%;height:auto">'
        ),
        '<title id="case-score-title">Case score ranking</title>',
        f'<text x="20" y="30" fill="{_TEXT}" font-family="sans-serif" '
        'font-size="18" font-weight="600">Case score ranking</text>',
    ]
    for tick in range(5):
        ratio = tick / 4
        x = label_width + ratio * plot_width
        value = lower + ratio * (upper - lower)
        label = f"{value:.0%}" if normalized else f"{value:.4g}"
        lines.extend(
            (
                f'<line x1="{x:.2f}" y1="50" x2="{x:.2f}" y2="{height - 30}" '
                f'stroke="{_TRACK}" stroke-width="1"/>',
                f'<text x="{x:.2f}" y="48" text-anchor="middle" fill="{_MUTED}" '
                f'font-family="sans-serif" font-size="12">{label}</text>',
            )
        )
    for index, (case, score) in enumerate(visible):
        ratio = max(0.0, min(1.0, (score - lower) / (upper - lower)))
        y = top + index * row_height
        color = (
            _GREEN if case.quality_pass is True else _RED if case.quality_pass is False else _TEAL
        )
        value = f"{score:.1%}" if normalized else f"{score:.4g}"
        lines.extend(
            (
                f'<text x="20" y="{y + 18}" fill="{_TEXT}" font-family="sans-serif" '
                f'font-size="13">{escape(case.case_id)}</text>',
                f'<rect x="{label_width}" y="{y + 3}" width="{plot_width}" height="20" '
                f'rx="3" fill="{_TRACK}"/>',
                f'<rect x="{label_width}" y="{y + 3}" width="{ratio * plot_width:.2f}" '
                f'height="20" rx="3" fill="{color}"/>',
                f'<text x="{label_width + plot_width + 16}" y="{y + 18}" fill="{_TEXT}" '
                f'font-family="sans-serif" font-size="13" font-weight="600">{value}</text>',
            )
        )
    if len(scored) > len(visible):
        lines.append(
            f'<text x="20" y="{height - 10}" fill="{_MUTED}" font-family="sans-serif" '
            f'font-size="12">Showing the highest-scoring {len(visible)} of {len(scored)} cases.</text>'
        )
    lines.append("</svg>")
    return "\n".join(lines)


def render_dimension_chart(evaluation: EvaluationSummaryReport) -> str | None:
    metrics = tuple(
        metric
        for metric in evaluation.metrics
        if metric.kind == "score" and metric.minimum >= 0 and metric.maximum <= 1
    )[:10]
    if not metrics:
        return None
    width = 920
    label_width = 280
    plot_width = 500
    top = 65
    row_height = 38
    height = top + len(metrics) * row_height + 32
    lines = [
        (
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
            f'viewBox="0 0 {width} {height}" role="img" '
            'aria-labelledby="dimension-title" style="max-width:100%;height:auto">'
        ),
        '<title id="dimension-title">Average quality by evaluation dimension</title>',
        f'<text x="20" y="30" fill="{_TEXT}" font-family="sans-serif" '
        'font-size="18" font-weight="600">Average quality by dimension</text>',
    ]
    for tick in range(5):
        x = label_width + tick * plot_width / 4
        lines.extend(
            (
                f'<line x1="{x:.2f}" y1="45" x2="{x:.2f}" y2="{height - 20}" '
                f'stroke="{_TRACK}" stroke-width="1"/>',
                f'<text x="{x:.2f}" y="58" text-anchor="middle" fill="{_MUTED}" '
                f'font-family="sans-serif" font-size="12">{tick * 25}%</text>',
            )
        )
    for index, metric in enumerate(metrics):
        y = top + index * row_height
        lines.extend(
            (
                f'<text x="20" y="{y + 20}" fill="{_TEXT}" font-family="sans-serif" '
                f'font-size="13">{escape(metric.label)}</text>',
                f'<rect x="{label_width}" y="{y + 4}" width="{plot_width}" height="22" '
                f'rx="3" fill="{_TRACK}"/>',
                f'<rect x="{label_width}" y="{y + 4}" width="{metric.mean * plot_width:.2f}" '
                f'height="22" rx="3" fill="{_TEAL}"/>',
                f'<text x="{label_width + plot_width + 16}" y="{y + 20}" fill="{_TEXT}" '
                f'font-family="sans-serif" font-size="13" font-weight="600">'
                f"{metric.mean:.1%} · n={metric.sample_count}</text>",
            )
        )
    lines.append("</svg>")
    return "\n".join(lines)


__all__ = (
    "render_case_score_chart",
    "render_dimension_chart",
    "render_quality_gate_chart",
)
