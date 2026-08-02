from __future__ import annotations as _annotations

from datetime import datetime
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field

from autobench.data.datasets import Case
from autobench.runtime.traces import TraceEnvelope


class SampleReason(StrEnum):
    RANDOM = "random"
    FAILURE_ONLY = "failure_only"
    LOW_CONFIDENCE = "low_confidence"
    HIGH_COST = "high_cost"
    HIGH_LATENCY = "high_latency"


class ReviewStatus(StrEnum):
    CANDIDATE = "candidate"
    ACCEPTED = "accepted"
    REJECTED = "rejected"


class ProductionSample(BaseModel):
    id: str
    input: Any = None
    output: Any = None
    expected: Any = None
    trace: TraceEnvelope | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    timestamp: datetime | None = None
    privacy_tags: tuple[str, ...] = ()
    reason: SampleReason = SampleReason.RANDOM
    review_status: ReviewStatus = ReviewStatus.CANDIDATE


class SamplingPolicy(BaseModel):
    reasons: tuple[SampleReason, ...] = (SampleReason.RANDOM,)
    max_samples: int | None = None


def sample_to_case(sample: ProductionSample) -> Case:
    metadata = dict(sample.metadata)
    metadata["source"] = "production"
    metadata["sample_reason"] = sample.reason.value
    metadata["review_status"] = sample.review_status.value
    if sample.timestamp is not None:
        metadata["timestamp"] = sample.timestamp.isoformat()
    if sample.privacy_tags:
        metadata["privacy_tags"] = list(sample.privacy_tags)
    if sample.trace is not None:
        metadata["trace_id"] = sample.trace.trace_id
    return Case(
        id=sample.id,
        input=sample.input,
        expected=sample.expected,
        metadata=metadata,
    )


def samples_to_cases(
    samples: list[ProductionSample],
    *,
    policy: SamplingPolicy | None = None,
) -> list[Case]:
    active_policy = policy or SamplingPolicy()
    selected: list[ProductionSample] = [
        sample for sample in samples if sample.reason in active_policy.reasons
    ]
    if active_policy.max_samples is not None:
        selected = selected[: active_policy.max_samples]
    return [sample_to_case(sample) for sample in selected]


__all__ = (
    "ProductionSample",
    "ReviewStatus",
    "SampleReason",
    "SamplingPolicy",
    "sample_to_case",
    "samples_to_cases",
)
