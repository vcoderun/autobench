from __future__ import annotations as _annotations

from typing import Any

from pydantic import BaseModel, Field

from autobench.metrics.observations import ObservationRole
from autobench.records.recording import RunRecord
from autobench.runtime.context import SpanRecord


class FeedbackRecord(BaseModel):
    score_name: str | None = None
    semantic_type: str | None = None
    score: float | bool | str | None = None
    passed: bool | None = None
    reason: str | None = None
    failure_category: str | None = None
    related_spans: tuple[str, ...] = ()
    related_assets: tuple[str, ...] = ()


class OptimizationFeedbackInput(BaseModel):
    run_id: str
    case_id: str
    variant_id: str
    task_status: str
    evaluation_status: str
    factors: dict[str, Any] = Field(default_factory=dict)
    asset_versions: dict[str, str] = Field(default_factory=dict)
    feedback: tuple[FeedbackRecord, ...] = ()
    trace_excerpt: tuple[dict[str, Any], ...] = ()


def build_feedback_records(record: RunRecord) -> tuple[FeedbackRecord, ...]:
    feedback: list[FeedbackRecord] = []
    for score in record.scores:
        passed = _score_passed(score.value)
        if passed is True:
            continue
        feedback.append(
            FeedbackRecord(
                score_name=score.name,
                semantic_type=score.semantic_type,
                score=score.value,
                passed=passed,
                reason=_score_reason(score.tags),
                failure_category=_score_failure_category(score.value, score.error is not None),
                related_spans=() if score.span_id is None else (score.span_id,),
                related_assets=tuple(asset.asset_id for asset in record.asset_versions),
            )
        )
    for error in record.errors:
        feedback.append(
            FeedbackRecord(
                reason=error.message,
                failure_category="error",
                related_spans=() if error.span_id is None else (error.span_id,),
                related_assets=tuple(asset.asset_id for asset in record.asset_versions),
            )
        )
    for observation in record.observations:
        if observation.role is ObservationRole.CONSTRAINT and observation.value is False:
            feedback.append(
                FeedbackRecord(
                    score_name=observation.name,
                    semantic_type=observation.semantic_type,
                    score=False,
                    passed=False,
                    reason=_score_reason(observation.tags),
                    failure_category="constraint",
                    related_spans=() if observation.span_id is None else (observation.span_id,),
                    related_assets=tuple(asset.asset_id for asset in record.asset_versions),
                )
            )
    return tuple(feedback)


def build_optimization_feedback_input(record: RunRecord) -> OptimizationFeedbackInput:
    return OptimizationFeedbackInput(
        run_id=record.run_id,
        case_id=record.case_id,
        variant_id=record.variant_id,
        task_status=record.task_status.value,
        evaluation_status=record.evaluation_status.value,
        factors={factor.name: factor.value for factor in record.factors},
        asset_versions={asset.asset_id: asset.version for asset in record.asset_versions},
        feedback=build_feedback_records(record),
        trace_excerpt=tuple(_span_excerpt(span) for span in record.spans),
    )


def _score_passed(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    if isinstance(value, int | float):
        return value > 0
    return None


def _score_failure_category(value: Any, has_error: bool) -> str | None:
    if has_error:
        return "error"
    passed = _score_passed(value)
    if passed is False:
        return "low_score"
    return None


def _score_reason(tags: dict[str, Any]) -> str | None:
    reason = tags.get("reason")
    if isinstance(reason, str):
        return reason
    return None


def _span_excerpt(span: SpanRecord) -> dict[str, str | None]:
    return {
        "id": span.id,
        "name": span.name,
        "kind": str(span.kind),
        "parent": span.parent_id,
        "error": None if span.error is None else span.error.message,
    }


__all__ = (
    "FeedbackRecord",
    "OptimizationFeedbackInput",
    "build_feedback_records",
    "build_optimization_feedback_input",
)
