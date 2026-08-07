from __future__ import annotations as _annotations

import traceback as traceback_module
from pathlib import Path

from pydantic import BaseModel


class AutobenchError(Exception):
    """Base exception for Autobench."""


class SpecLoadError(AutobenchError):
    """Raised when a YAML spec cannot be loaded."""

    def __init__(
        self,
        message: str,
        *,
        path: Path | None = None,
        line: int | None = None,
        column: int | None = None,
    ) -> None:
        super().__init__(message)
        self.path = path
        self.line = line
        self.column = column


class SpecValidationError(AutobenchError):
    """Raised when a loaded YAML spec does not match the Autobench model."""


class TaskResolutionError(AutobenchError):
    """Raised when a Python task target cannot be resolved."""


class GenerationError(AutobenchError):
    """Raised when generated dataset preparation cannot produce valid evidence."""


class ErrorRecord(BaseModel):
    error_type: str
    message: str
    traceback: str | None = None
    span_id: str | None = None

    @classmethod
    def from_exception(
        cls,
        exc: BaseException,
        *,
        span_id: str | None = None,
        include_traceback: bool = True,
    ) -> ErrorRecord:
        rendered_traceback: str | None = None
        if include_traceback:
            rendered_traceback = "".join(
                traceback_module.format_exception(type(exc), exc, exc.__traceback__)
            )
        return cls(
            error_type=type(exc).__name__,
            message=str(exc),
            traceback=rendered_traceback,
            span_id=span_id,
        )


__all__ = (
    "AutobenchError",
    "ErrorRecord",
    "GenerationError",
    "SpecLoadError",
    "SpecValidationError",
    "TaskResolutionError",
)
