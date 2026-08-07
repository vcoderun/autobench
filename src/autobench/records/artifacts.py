from __future__ import annotations as _annotations

from collections.abc import AsyncIterable, Iterable
from enum import StrEnum
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field, model_validator

from autobench.errors import AutobenchError


class ArtifactSource(StrEnum):
    VALUE = "value"
    FILE = "file"
    STREAM = "stream"


class ArtifactState(StrEnum):
    COMPLETE = "complete"
    TRUNCATED = "truncated"
    PARTIAL = "partial"


class ArtifactOverflow(StrEnum):
    FAIL = "fail"
    TRUNCATE = "truncate"


class SymlinkPolicy(StrEnum):
    FOLLOW = "follow"
    REJECT = "reject"


class ArtifactRef(BaseModel):
    model_config = ConfigDict(frozen=True)

    id: str
    name: str
    media_type: str | None = None
    value: Any = None
    span_id: str | None = None
    tags: dict[str, Any] = Field(default_factory=dict)
    source: ArtifactSource = ArtifactSource.VALUE
    state: ArtifactState = ArtifactState.COMPLETE
    sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    byte_count: int | None = Field(default=None, ge=0)
    filename: str | None = Field(default=None, min_length=1)
    symlink_followed: bool = False

    @model_validator(mode="after")
    def validate_payload_metadata(self) -> ArtifactRef:
        if (self.sha256 is None) != (self.byte_count is None):
            raise ValueError("artifact sha256 and byte_count must be provided together")
        if self.source is ArtifactSource.VALUE and self.state is not ArtifactState.COMPLETE:
            raise ValueError("in-memory artifacts must be complete")
        return self


class ArtifactError(AutobenchError):
    """Raised when a file-backed artifact cannot be prepared safely."""


class ArtifactSinkRequiredError(ArtifactError):
    """Raised before a file or stream is consumed without durable recording."""


class ArtifactTransferError(ArtifactError):
    def __init__(self, message: str, artifact: ArtifactRef) -> None:
        super().__init__(message)
        self.artifact = artifact


@runtime_checkable
class ArtifactSink(Protocol):
    def prepare_file(
        self,
        *,
        run_id: str,
        artifact_id: str,
        name: str,
        source: Path,
        media_type: str | None,
        max_bytes: int,
        overflow: ArtifactOverflow,
        symlinks: SymlinkPolicy,
        filename: str | None,
        span_id: str | None,
        tags: dict[str, Any],
    ) -> ArtifactRef: ...

    def prepare_stream(
        self,
        *,
        run_id: str,
        artifact_id: str,
        name: str,
        source: Iterable[bytes],
        media_type: str | None,
        max_bytes: int,
        overflow: ArtifactOverflow,
        filename: str | None,
        span_id: str | None,
        tags: dict[str, Any],
    ) -> ArtifactRef: ...

    async def prepare_stream_async(
        self,
        *,
        run_id: str,
        artifact_id: str,
        name: str,
        source: AsyncIterable[bytes],
        media_type: str | None,
        max_bytes: int,
        overflow: ArtifactOverflow,
        filename: str | None,
        span_id: str | None,
        tags: dict[str, Any],
    ) -> ArtifactRef: ...

    def prepared_artifact(self, *, run_id: str, artifact_id: str) -> ArtifactRef | None: ...


__all__ = (
    "ArtifactError",
    "ArtifactOverflow",
    "ArtifactRef",
    "ArtifactSink",
    "ArtifactSinkRequiredError",
    "ArtifactSource",
    "ArtifactState",
    "ArtifactTransferError",
    "SymlinkPolicy",
)
