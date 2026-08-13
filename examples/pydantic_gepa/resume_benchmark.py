from __future__ import annotations as _annotations

from pathlib import Path
from tempfile import TemporaryDirectory

from pydantic import BaseModel
from pydantic_gepa import Candidate, Plan, RunConfig, Stage
from pydantic_gepa.orchestration import StageOutput

from autobench import Case, RunContext


class ResumeOutcome(BaseModel):
    candidate: str
    resumed_equal: bool
    metric_calls: int


def run(_ctx: RunContext, case: Case) -> ResumeOutcome:
    target = str(case.expected["candidate"])

    def improve(candidate: Candidate, _limit: int) -> StageOutput:
        return StageOutput(
            candidate=Candidate(values={**candidate.values, "prompt": target}),
            score=1.0,
            metric_calls=1,
        )

    plan = Plan(
        Stage("prompt", components=("prompt",), run=improve, run_id="improve"),
        initial_candidate=Candidate(values={"prompt": str(case.input)}),
    )
    with TemporaryDirectory(prefix="autobench-pydantic-gepa-") as directory:
        path = Path(directory)
        first = plan.run(run=RunConfig(id="checkpoint-example", directory=path))
        resumed = plan.run(
            run=RunConfig(
                id="checkpoint-example",
                directory=path,
                resume="required",
            )
        )
    return ResumeOutcome(
        candidate=resumed.final_candidate.values["prompt"],
        resumed_equal=resumed == first,
        metric_calls=resumed.total_metric_calls,
    )


__all__ = ("run",)
