from __future__ import annotations as _annotations

import importlib
import sys
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from enum import StrEnum
from inspect import isawaitable
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from autobench.data.datasets import Case
from autobench.errors import ErrorRecord, TaskResolutionError
from autobench.metrics.observations import Observation
from autobench.protocol import EndReason
from autobench.records.artifacts import ArtifactRef
from autobench.runtime.context import RunContext, SpanRecord
from autobench.runtime.instrumentation import reset_active_run_context, set_active_run_context


class TaskStatus(StrEnum):
    PASSED = "passed"
    FAILED = "failed"
    ERRORED = "errored"
    SKIPPED = "skipped"
    CANCELLED = "cancelled"


class TaskResult(BaseModel):
    output: Any = None
    status: TaskStatus
    partial: bool = False
    end_reason: EndReason = EndReason.COMPLETED
    error: ErrorRecord | None = None
    errors: list[ErrorRecord] = Field(default_factory=list)
    observations: list[Observation] = Field(default_factory=list)
    spans: list[SpanRecord] = Field(default_factory=list)
    artifacts: list[ArtifactRef] = Field(default_factory=list)


def resolve_python_callable(
    target: str,
    *,
    search_paths: tuple[str, ...] = (),
) -> Callable[..., Any]:
    module_name, separator, attribute_name = target.partition(":")
    if not separator or not module_name or not attribute_name:
        raise TaskResolutionError("Python task targets must use 'module:function' format.")

    try:
        module = importlib.import_module(module_name)
    except Exception:
        try:
            with _temporary_sys_path(search_paths):
                module = importlib.import_module(module_name)
        except Exception as fallback_exc:
            raise TaskResolutionError(
                f"Could not import task module '{module_name}'."
            ) from fallback_exc

    try:
        task = getattr(module, attribute_name)
    except AttributeError as exc:
        raise TaskResolutionError(
            f"Task target '{target}' does not define '{attribute_name}'."
        ) from exc

    if not callable(task):
        raise TaskResolutionError(f"Task target '{target}' is not callable.")
    return task


async def run_python_task(
    target: str,
    *,
    ctx: RunContext,
    case: Case,
    search_paths: tuple[str, ...] = (),
) -> TaskResult:
    token = set_active_run_context(ctx)
    try:
        task = resolve_python_callable(target, search_paths=search_paths)
        output = task(ctx, case)
        if isawaitable(output):
            output = await output
    except TaskResolutionError as exc:
        error = _error_for_exception(ctx, exc)
        return TaskResult(
            output=None,
            status=TaskStatus.ERRORED,
            end_reason=EndReason.FAILED,
            error=error,
            errors=list(ctx.errors),
            observations=list(ctx.observations),
            spans=list(ctx.spans),
            artifacts=list(ctx.artifacts),
        )
    except Exception as exc:
        error = _error_for_exception(ctx, exc)
        return TaskResult(
            output=None,
            status=TaskStatus.FAILED,
            end_reason=EndReason.FAILED,
            error=error,
            errors=list(ctx.errors),
            observations=list(ctx.observations),
            spans=list(ctx.spans),
            artifacts=list(ctx.artifacts),
        )
    finally:
        reset_active_run_context(token)

    return TaskResult(
        output=output,
        status=TaskStatus.PASSED,
        errors=list(ctx.errors),
        observations=list(ctx.observations),
        spans=list(ctx.spans),
        artifacts=list(ctx.artifacts),
    )


def _error_for_exception(ctx: RunContext, exc: Exception) -> ErrorRecord:
    for error in reversed(ctx.errors):
        if error.error_type == type(exc).__name__ and error.message == str(exc):
            return error
    return ctx.error(exc)


@contextmanager
def _temporary_sys_path(search_paths: tuple[str, ...]) -> Iterator[None]:
    normalized_paths = [
        str(Path(search_path).resolve())
        for search_path in search_paths
        if str(Path(search_path).resolve()) not in sys.path
    ]
    if not normalized_paths:
        yield
        return

    for search_path in reversed(normalized_paths):
        sys.path.insert(0, search_path)
    try:
        yield
    finally:
        for search_path in normalized_paths:
            try:
                sys.path.remove(search_path)
            except ValueError:
                continue


__all__ = (
    "TaskResult",
    "TaskStatus",
    "resolve_python_callable",
    "run_python_task",
)
