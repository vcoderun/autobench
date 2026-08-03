from __future__ import annotations

from enum import StrEnum
from hashlib import sha256
from math import isfinite
from threading import Lock
from typing import Annotated, Any, TypeAlias

from pydantic import BaseModel, BeforeValidator, ConfigDict, Field, JsonValue

from autobench.tracking.models import AssetVersion


def validate_serialized_value(value: Any) -> JsonValue:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not isfinite(value):
            raise ValueError("serialized numbers must be finite")
        return value
    if isinstance(value, list):
        return [validate_serialized_value(item) for item in value]
    if isinstance(value, dict):
        if not all(isinstance(key, str) for key in value):
            raise ValueError("serialized mappings require string keys")
        return {key: validate_serialized_value(item) for key, item in value.items()}
    raise ValueError(f"unsupported serialized value: {type(value).__name__}")


SerializedValue: TypeAlias = Annotated[
    JsonValue,
    BeforeValidator(validate_serialized_value),
]


class ReferenceKind(StrEnum):
    ARTIFACT = "artifact"
    ASSET = "asset"
    DATASET_ITEM = "dataset_item"
    PROMPT = "prompt"
    MESSAGE = "message"
    TOOL = "tool"
    OUTPUT_SCHEMA = "output_schema"
    EXTERNAL_TRACE = "external_trace"
    ERROR = "error"
    CUSTOM = "custom"


class EvidenceRef(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    kind: ReferenceKind
    id: str = Field(min_length=1)
    version: str | None = Field(default=None, min_length=1)
    media_type: str | None = Field(default=None, min_length=1)


class StoredArtifact(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    reference: EvidenceRef
    content: bytes
    content_hash: str = Field(pattern=r"^[0-9a-f]{64}$")

    @property
    def size_bytes(self) -> int:
        return len(self.content)


class ReferenceStore:
    def __init__(self) -> None:
        self._lock = Lock()
        self._artifacts: dict[tuple[str, str], StoredArtifact] = {}
        self._assets: dict[tuple[ReferenceKind, str, str], EvidenceRef] = {}

    @property
    def artifacts(self) -> tuple[StoredArtifact, ...]:
        with self._lock:
            return tuple(self._artifacts.values())

    @property
    def assets(self) -> tuple[EvidenceRef, ...]:
        with self._lock:
            return tuple(self._assets.values())

    def add_artifact(self, content: bytes, *, media_type: str) -> EvidenceRef:
        digest = sha256(content).hexdigest()
        key = (digest, media_type)
        with self._lock:
            existing = self._artifacts.get(key)
            if existing is not None:
                return existing.reference
            reference_digest = sha256(media_type.encode() + b"\0" + content).hexdigest()
            reference = EvidenceRef(
                kind=ReferenceKind.ARTIFACT,
                id=f"artifact_{reference_digest[:24]}",
                media_type=media_type,
            )
            self._artifacts[key] = StoredArtifact(
                reference=reference,
                content=content,
                content_hash=digest,
            )
            return reference

    def add_asset(
        self,
        version: AssetVersion,
        *,
        kind: ReferenceKind = ReferenceKind.ASSET,
    ) -> EvidenceRef:
        if kind not in {
            ReferenceKind.ASSET,
            ReferenceKind.PROMPT,
            ReferenceKind.TOOL,
            ReferenceKind.OUTPUT_SCHEMA,
        }:
            raise ValueError("tracked assets require an asset reference kind")
        key = (kind, version.asset_id, version.version)
        with self._lock:
            existing = self._assets.get(key)
            if existing is not None:
                return existing
            reference = EvidenceRef(
                kind=kind,
                id=version.asset_id,
                version=version.version,
            )
            self._assets[key] = reference
            return reference


__all__ = (
    "EvidenceRef",
    "ReferenceKind",
    "ReferenceStore",
    "SerializedValue",
    "StoredArtifact",
)
