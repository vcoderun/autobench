from __future__ import annotations as _annotations

from typing import Any

from pydantic import BaseModel, Field


class ArtifactRef(BaseModel):
    id: str
    name: str
    media_type: str | None = None
    value: Any = None
    span_id: str | None = None
    tags: dict[str, Any] = Field(default_factory=dict)


__all__ = ("ArtifactRef",)
