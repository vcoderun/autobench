from __future__ import annotations as _annotations

from datetime import UTC, datetime
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field


class ProgressEventKind(StrEnum):
    BENCHMARK_STARTED = "benchmark_started"
    BENCHMARK_FINISHED = "benchmark_finished"
    RUN_STARTED = "run_started"
    RUN_FINISHED = "run_finished"
    CANDIDATE_DECISION = "candidate_decision"
    POLICY_VIOLATION = "policy_violation"


class ProgressEvent(BaseModel):
    kind: ProgressEventKind
    message: str
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))
    benchmark_id: str | None = None
    experiment_id: str | None = None
    run_id: str | None = None
    case_id: str | None = None
    variant_id: str | None = None
    data: dict[str, Any] = Field(default_factory=dict)


def progress_event(
    kind: ProgressEventKind,
    message: str,
    **data: Any,
) -> ProgressEvent:
    known_fields = {
        "benchmark_id",
        "experiment_id",
        "run_id",
        "case_id",
        "variant_id",
    }
    payload = {key: value for key, value in data.items() if key in known_fields}
    payload["data"] = {key: value for key, value in data.items() if key not in known_fields}
    return ProgressEvent(kind=kind, message=message, **payload)


__all__ = (
    "ProgressEvent",
    "ProgressEventKind",
    "progress_event",
)
