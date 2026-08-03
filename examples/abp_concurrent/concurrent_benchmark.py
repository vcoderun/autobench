from __future__ import annotations as _annotations

import asyncio
from typing import Any

from autobench import InstrumentMetricSpec, RunContext, SpanKind, instrument_method
from autobench.data.datasets import Case


class Worker:
    async def execute(self, units: int) -> dict[str, int]:
        await asyncio.sleep(0)
        return {"units": units}


_WORKER = Worker()
_INSTRUMENTATION = instrument_method(
    Worker,
    "execute",
    span="worker.execute",
    span_kind=SpanKind.WORKFLOW,
    metrics=[
        InstrumentMetricSpec(
            name="completed_units",
            semantic_type="work.units",
            value_path="result.units",
        )
    ],
)


async def run(ctx: RunContext, case: Case) -> dict[str, Any]:
    payload = case.input if isinstance(case.input, dict) else {}
    raw_units = payload.get("units", [])
    units = [int(value) for value in raw_units] if isinstance(raw_units, list) else []
    with ctx.span("batch.workflow", kind=SpanKind.WORKFLOW, input=payload) as workflow:
        completed = await asyncio.gather(*(_WORKER.execute(value) for value in units))
        result = {"total": sum(item["units"] for item in completed)}
        workflow.set_output(result)
        return result


__all__ = ("run",)
