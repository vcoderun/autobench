from __future__ import annotations as _annotations

from typing import Any

from autobench import (
    DurationMetricSpec,
    InstrumentFactorSpec,
    InstrumentMetricSpec,
    RunContext,
    Semantic,
    SpanKind,
    instrument_method,
)
from autobench.data.datasets import Case


class TicketRouter:
    def route(self, message: str, profile: str) -> dict[str, str | int]:
        queue = "billing" if profile == "keyword" and "refund" in message else "technical"
        return {"queue": queue, "profile": profile, "step_count": 2}


_ROUTER = TicketRouter()
_INSTRUMENTATION = instrument_method(
    TicketRouter,
    "route",
    span="support.route",
    span_kind=SpanKind.WORKFLOW,
    metrics=[
        InstrumentMetricSpec(
            name="routing_steps",
            semantic_type="workflow.steps",
            value_path="result.step_count",
        )
    ],
    factors=[
        InstrumentFactorSpec(
            name="routing_profile",
            semantic_type="workflow.profile",
            value_path="result.profile",
        )
    ],
)


def run(ctx: RunContext, case: Case) -> dict[str, Any]:
    payload = case.input if isinstance(case.input, dict) else {}
    message = str(payload.get("message", ""))
    with ctx.span(
        "support.workflow",
        kind=SpanKind.WORKFLOW,
        input=payload,
        duration_metric=DurationMetricSpec(
            name="workflow_latency",
            semantic_type=Semantic.TIME_LATENCY,
            unit="s",
        ),
    ) as workflow:
        result = _ROUTER.route(message, str(ctx.factor("routing_profile")))
        workflow.set_output(result)
        return result


__all__ = ("run",)
