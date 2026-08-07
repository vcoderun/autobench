from __future__ import annotations as _annotations

import asyncio
import inspect
import warnings
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field

from autobench.errors import AutobenchError
from autobench.runtime.models import ExperimentStatus, RunStatus


class ProgressEventKind(StrEnum):
    BENCHMARK_STARTED = "benchmark_started"
    BENCHMARK_FINISHED = "benchmark_finished"
    RUN_STARTED = "run_started"
    RUN_FINISHED = "run_finished"
    POLICY_VIOLATION = "policy_violation"


class ProgressErrorPolicy(StrEnum):
    STRICT = "strict"
    BEST_EFFORT = "best_effort"


class ProgressEvent(BaseModel):
    kind: ProgressEventKind
    message: str
    sequence: int = Field(default=0, ge=0)
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))
    benchmark_id: str | None = None
    experiment_id: str | None = None
    run_id: str | None = None
    case_id: str | None = None
    variant_id: str | None = None
    run_status: RunStatus | None = None
    experiment_status: ExperimentStatus | None = None
    data: dict[str, Any] = Field(default_factory=dict)


ProgressHandler = Callable[[ProgressEvent], None | Awaitable[None]]


@dataclass(frozen=True, slots=True)
class ProgressHandlerFailure:
    handler_index: int
    sequence: int
    event_kind: ProgressEventKind
    error: Exception


ProgressErrorHandler = Callable[[ProgressHandlerFailure], None]


class ProgressDispatchError(AutobenchError):
    """Raised after strict progress delivery fails and lifecycle cleanup completes."""

    def __init__(self, failures: Sequence[ProgressHandlerFailure]) -> None:
        self.failures = tuple(failures)
        count = len(self.failures)
        super().__init__(f"Progress delivery failed for {count} handler invocation(s).")


def progress_event(
    kind: ProgressEventKind,
    message: str,
    *,
    sequence: int = 0,
    run_status: RunStatus | None = None,
    experiment_status: ExperimentStatus | None = None,
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
    return ProgressEvent(
        kind=kind,
        message=message,
        sequence=sequence,
        run_status=run_status,
        experiment_status=experiment_status,
        **payload,
    )


class _ProgressDispatcher:
    def __init__(
        self,
        handlers: Sequence[ProgressHandler],
        *,
        error_policy: ProgressErrorPolicy,
        error_handler: ProgressErrorHandler | None,
    ) -> None:
        self._handlers = tuple(handlers)
        self._error_policy = error_policy
        self._error_handler = error_handler
        self._failed_handlers: set[int] = set()
        self._failures: list[ProgressHandlerFailure] = []
        self._sequence = 0
        self._lock = asyncio.Lock()

    async def emit(
        self,
        kind: ProgressEventKind,
        message: str,
        *,
        run_status: RunStatus | None = None,
        experiment_status: ExperimentStatus | None = None,
        **data: Any,
    ) -> ProgressEvent:
        async with self._lock:
            self._sequence += 1
            event = progress_event(
                kind,
                message,
                sequence=self._sequence,
                run_status=run_status,
                experiment_status=experiment_status,
                **data,
            )
            for index, handler in enumerate(self._handlers):
                if index in self._failed_handlers:
                    continue
                try:
                    result = handler(event)
                    if inspect.isawaitable(result):
                        await result
                except Exception as error:
                    self._failed_handlers.add(index)
                    failure = ProgressHandlerFailure(
                        handler_index=index,
                        sequence=event.sequence,
                        event_kind=event.kind,
                        error=error,
                    )
                    self._failures.append(failure)
                    if self._error_policy is ProgressErrorPolicy.BEST_EFFORT:
                        self._report(failure)
            return event

    def raise_if_failed(self) -> None:
        if self._error_policy is not ProgressErrorPolicy.STRICT or not self._failures:
            return
        error = ProgressDispatchError(self._failures)
        raise error from self._failures[0].error

    def error(self) -> ProgressDispatchError | None:
        if self._error_policy is not ProgressErrorPolicy.STRICT or not self._failures:
            return None
        return ProgressDispatchError(self._failures)

    def _report(self, failure: ProgressHandlerFailure) -> None:
        if self._error_handler is None:
            warnings.warn(
                (
                    f"Progress handler {failure.handler_index} failed during "
                    f"{failure.event_kind}: {failure.error}"
                ),
                stacklevel=3,
            )
            return
        try:
            self._error_handler(failure)
        except Exception as error:
            warnings.warn(f"Progress error reporter failed: {error}", stacklevel=3)


__all__ = (
    "ProgressDispatchError",
    "ProgressErrorHandler",
    "ProgressErrorPolicy",
    "ProgressEvent",
    "ProgressEventKind",
    "ProgressHandler",
    "ProgressHandlerFailure",
    "progress_event",
)
