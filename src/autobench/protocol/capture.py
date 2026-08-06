from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from copy import deepcopy
from dataclasses import fields, is_dataclass
from datetime import date, datetime
from decimal import Decimal
from enum import Enum
from fnmatch import fnmatchcase
from hashlib import sha256
from json import dumps
from math import isfinite
from pathlib import Path
from typing import Any, TypeAlias, TypeVar
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, model_validator

from autobench.metrics.semantics import SemanticType
from autobench.protocol.signals import CaptureLevel
from autobench.protocol.traces import Diagnostic, DiagnosticSeverity
from autobench.protocol.values import (
    EvidenceRef,
    ReferenceKind,
    ReferenceStore,
    SerializedValue,
)
from autobench.tracking.models import AssetVersion

T = TypeVar("T")
Redactor: TypeAlias = Callable[[str, tuple[str, ...], SemanticType | None], str]
Encoder: TypeAlias = Callable[[Any], Any]

_HASH_SEMANTICS = (
    "prompt",
    "message",
    "llm.prompt",
    "llm.message",
    "tool.call.arguments",
    "tool.call.result",
    "memory",
    "retrieval.content",
    "retrieval.document",
)
_SAFE_STRING_SEMANTICS = (
    "llm.model",
    "llm.provider",
    "model.name",
    "model.provider",
    "tool.name",
    "tool.type",
    "agent.id",
    "agent.name",
    "conversation.id",
    "workflow.name",
    "operation.name",
    "tool.call.id",
    "abp.logical_operation_id",
    "usage_authority",
    "request_id",
    "response_id",
    "response.status",
    "finish_reason",
    "service_tier",
    "http.request.method",
    "http.request.scheme",
    "http.request.host",
    "http.request.path_hash",
    "network.protocol.version",
    "error.type",
)
_SECRET_NAMES = {
    "api_key",
    "apikey",
    "authorization",
    "cookie",
    "password",
    "secret",
    "set_cookie",
    "token",
}


class CapturePolicy(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    default_level: CaptureLevel = CaptureLevel.METADATA
    asset_default_level: CaptureLevel = CaptureLevel.FULL
    use_semantic_defaults: bool = True
    semantic_overrides: dict[SemanticType, CaptureLevel] = Field(default_factory=dict)
    allow_semantics: tuple[str, ...] = ()
    deny_semantics: tuple[str, ...] = ()
    allow_paths: tuple[str, ...] = ()
    deny_paths: tuple[str, ...] = ()
    secret_names: frozenset[str] = frozenset(_SECRET_NAMES)
    max_inline_bytes: int = Field(default=16_384, ge=1)
    max_artifact_bytes: int = Field(default=4_194_304, ge=1)
    max_collection_items: int = Field(default=100, ge=1)
    max_string_length: int = Field(default=4_096, ge=1)
    max_depth: int = Field(default=8, ge=1)
    store_binary: bool = True
    retain_source_attributes: bool = False

    @model_validator(mode="after")
    def validate_paths(self) -> CapturePolicy:
        if any(
            not pattern.strip()
            for pattern in (
                self.allow_semantics + self.deny_semantics + self.allow_paths + self.deny_paths
            )
        ):
            raise ValueError("capture paths and semantics cannot be empty")
        return self

    @classmethod
    def none(cls, **changes: Any) -> CapturePolicy:
        changes.setdefault("asset_default_level", CaptureLevel.NONE)
        return cls(default_level=CaptureLevel.NONE, use_semantic_defaults=False, **changes)

    @classmethod
    def metadata(cls, **changes: Any) -> CapturePolicy:
        changes.setdefault("asset_default_level", CaptureLevel.METADATA)
        return cls(default_level=CaptureLevel.METADATA, use_semantic_defaults=False, **changes)

    @classmethod
    def hashed(cls, **changes: Any) -> CapturePolicy:
        changes.setdefault("asset_default_level", CaptureLevel.HASH)
        return cls(default_level=CaptureLevel.HASH, use_semantic_defaults=False, **changes)

    @classmethod
    def redacted(cls, **changes: Any) -> CapturePolicy:
        changes.setdefault("asset_default_level", CaptureLevel.REDACTED)
        return cls(default_level=CaptureLevel.REDACTED, use_semantic_defaults=False, **changes)

    @classmethod
    def full(cls, **changes: Any) -> CapturePolicy:
        changes.setdefault("asset_default_level", CaptureLevel.FULL)
        return cls(default_level=CaptureLevel.FULL, use_semantic_defaults=False, **changes)

    def level_for_asset(
        self,
        semantic_type: SemanticType | None,
        *,
        explicit: CaptureLevel | None = None,
    ) -> CaptureLevel:
        if explicit is not None:
            return explicit
        if semantic_type is not None:
            matches = tuple(
                (prefix, level)
                for prefix, level in self.semantic_overrides.items()
                if semantic_type == prefix or semantic_type.startswith(f"{prefix}.")
            )
            if matches:
                return max(matches, key=lambda match: len(match[0]))[1]
        return self.asset_default_level

    def level_for(
        self,
        semantic_type: SemanticType | None,
        value: Any,
        *,
        explicit: CaptureLevel | None = None,
    ) -> CaptureLevel:
        if explicit is not None:
            return explicit
        if semantic_type is not None:
            matches = tuple(
                (prefix, level)
                for prefix, level in self.semantic_overrides.items()
                if semantic_type == prefix or semantic_type.startswith(f"{prefix}.")
            )
            if matches:
                return max(matches, key=lambda match: len(match[0]))[1]
            if self.use_semantic_defaults and isinstance(value, bytes):
                return CaptureLevel.METADATA
            if self.use_semantic_defaults and (
                semantic_type == "environment" or semantic_type.startswith("environment.")
            ):
                return CaptureLevel.NONE
            if self.use_semantic_defaults and any(
                semantic_type == prefix or semantic_type.startswith(f"{prefix}.")
                for prefix in _HASH_SEMANTICS
            ):
                return CaptureLevel.HASH
        if self.use_semantic_defaults and isinstance(value, BaseException):
            return CaptureLevel.REDACTED
        return self.default_level

    def allows_semantic(self, semantic_type: SemanticType | None) -> tuple[bool, str]:
        def matches(pattern: str) -> bool:
            return semantic_type is not None and (
                fnmatchcase(semantic_type, pattern)
                or semantic_type == pattern
                or semantic_type.startswith(f"{pattern}.")
            )

        if any(matches(pattern) for pattern in self.deny_semantics):
            return False, "semantic_denied"
        if not self.allow_semantics or any(matches(pattern) for pattern in self.allow_semantics):
            return True, "allowed"
        return False, "semantic_not_allowed"

    def allows_path(self, path: tuple[str, ...]) -> tuple[bool, str]:
        dotted = ".".join(path)
        if any(fnmatchcase(dotted, pattern) for pattern in self.deny_paths):
            return False, "denied"
        if not self.allow_paths:
            return True, "allowed"
        for pattern in self.allow_paths:
            if (
                fnmatchcase(dotted, pattern)
                or pattern.startswith(f"{dotted}.")
                or dotted.startswith(f"{pattern}.")
            ):
                return True, "allowed"
        return False, "not_allowed"

    def is_secret(self, path: tuple[str, ...]) -> bool:
        if not path:
            return False
        name = path[-1].lower().replace("-", "_")
        return any(
            name == secret or name.startswith(f"{secret}_") or name.endswith(f"_{secret}")
            for secret in self.secret_names
        )


class CaptureResult(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    level: CaptureLevel
    value: SerializedValue = None
    reference: EvidenceRef | None = None
    diagnostics: tuple[Diagnostic, ...] = ()
    omitted: bool = False
    truncated: bool = False


class EncoderRegistry:
    def __init__(self) -> None:
        self._encoders: dict[type[Any], Encoder] = {}

    def register(self, value_type: type[T], encoder: Callable[[T], Any]) -> None:
        def encode(value: Any) -> Any:
            return encoder(value)

        self._encoders[value_type] = encode

    def encoder_for(self, value: Any) -> Encoder | None:
        return self._encoders.get(type(value))


class CaptureSession:
    def __init__(
        self,
        policy: CapturePolicy | None = None,
        *,
        store: ReferenceStore | None = None,
        encoders: EncoderRegistry | None = None,
        redactors: Sequence[Redactor] = (),
    ) -> None:
        self.policy = CapturePolicy() if policy is None else policy
        self.store = ReferenceStore() if store is None else store
        self.encoders = EncoderRegistry() if encoders is None else encoders
        self.redactors = tuple(redactors)

    def capture(
        self,
        value: Any,
        *,
        semantic_type: SemanticType | None = None,
        path: str | tuple[str, ...] | None = None,
        level: CaptureLevel | None = None,
        asset_version: AssetVersion | None = None,
        reference_kind: ReferenceKind | None = None,
        media_type: str | None = None,
    ) -> CaptureResult:
        active_path = self._path(path, semantic_type)
        active_level = self.policy.level_for(semantic_type, value, explicit=level)
        diagnostics: list[Diagnostic] = []
        semantic_allowed, semantic_reason = self.policy.allows_semantic(semantic_type)
        if not semantic_allowed:
            self._diagnose(
                diagnostics,
                f"capture_{semantic_reason}",
                f"capture semantic was {semantic_reason.replace('_', ' ')} by policy",
                active_path,
                semantic_type,
            )
            return CaptureResult(level=active_level, diagnostics=tuple(diagnostics), omitted=True)
        allowed, reason = self.policy.allows_path(active_path)
        if not allowed:
            self._diagnose(
                diagnostics,
                f"capture_{reason}",
                f"capture path was {reason.replace('_', ' ')} by policy",
                active_path,
                semantic_type,
            )
            return CaptureResult(level=active_level, diagnostics=tuple(diagnostics), omitted=True)
        if active_level is CaptureLevel.NONE:
            self._diagnose(
                diagnostics,
                "capture_omitted",
                "capture level omitted the value",
                active_path,
                semantic_type,
            )
            return CaptureResult(level=active_level, diagnostics=tuple(diagnostics), omitted=True)

        if asset_version is not None:
            kind = reference_kind or self._asset_kind(semantic_type)
            reference = self.store.add_asset(asset_version, kind=kind)
            return CaptureResult(
                level=active_level,
                value={"asset_id": asset_version.asset_id, "version": asset_version.version},
                reference=reference,
            )

        if isinstance(value, bytes):
            return self._capture_binary(
                value,
                level=active_level,
                path=active_path,
                semantic_type=semantic_type,
                diagnostics=diagnostics,
                media_type=media_type,
            )

        normalized, supported, truncated = self._normalize(
            value,
            path=active_path,
            semantic_type=semantic_type,
            depth=0,
            diagnostics=diagnostics,
            encoder_types=(),
        )
        if not supported:
            return CaptureResult(level=active_level, diagnostics=tuple(diagnostics), omitted=True)

        if active_level is CaptureLevel.METADATA:
            captured = self._metadata(value, normalized, semantic_type)
            return CaptureResult(
                level=active_level,
                value=captured,
                diagnostics=tuple(diagnostics),
                truncated=truncated,
            )

        content = dumps(
            normalized,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode()
        if active_level is CaptureLevel.HASH:
            return CaptureResult(
                level=active_level,
                value={
                    "sha256": sha256(content).hexdigest(),
                    "type": type(value).__name__,
                    "size_bytes": len(content),
                },
                diagnostics=tuple(diagnostics),
                truncated=truncated,
            )
        if len(content) > self.policy.max_inline_bytes:
            if len(content) > self.policy.max_artifact_bytes:
                self._diagnose(
                    diagnostics,
                    "artifact_too_large",
                    "normalized value exceeded the artifact size limit",
                    active_path,
                    semantic_type,
                    severity=DiagnosticSeverity.ERROR,
                    details={
                        "size_bytes": len(content),
                        "max_artifact_bytes": self.policy.max_artifact_bytes,
                    },
                )
                return CaptureResult(
                    level=active_level,
                    value=self._metadata(value, normalized, semantic_type),
                    diagnostics=tuple(diagnostics),
                    omitted=True,
                    truncated=truncated,
                )
            reference = self.store.add_artifact(
                content,
                media_type=media_type or "application/json",
            )
            self._diagnose(
                diagnostics,
                "capture_referenced",
                "normalized value exceeded the inline limit and became an artifact",
                active_path,
                semantic_type,
                details={
                    "size_bytes": len(content),
                    "max_inline_bytes": self.policy.max_inline_bytes,
                },
            )
            return CaptureResult(
                level=active_level,
                value=self._metadata(value, normalized, semantic_type),
                reference=reference,
                diagnostics=tuple(diagnostics),
                truncated=truncated,
            )
        return CaptureResult(
            level=active_level,
            value=normalized,
            diagnostics=tuple(diagnostics),
            truncated=truncated,
        )

    def _normalize(
        self,
        value: Any,
        *,
        path: tuple[str, ...],
        semantic_type: SemanticType | None,
        depth: int,
        diagnostics: list[Diagnostic],
        encoder_types: tuple[type[Any], ...],
    ) -> tuple[SerializedValue, bool, bool]:
        if depth > self.policy.max_depth:
            self._diagnose(
                diagnostics,
                "capture_depth_limit",
                "value exceeded the capture nesting limit",
                path,
                semantic_type,
                details={"depth": depth, "max_depth": self.policy.max_depth},
            )
            return None, True, True

        encoder = self.encoders.encoder_for(value)
        if encoder is not None:
            value_type = type(value)
            if value_type in encoder_types:
                self._diagnose(
                    diagnostics,
                    "encoder_recursion",
                    "registered encoder returned its own input type",
                    path,
                    semantic_type,
                    severity=DiagnosticSeverity.ERROR,
                )
                return None, False, False
            try:
                encoded = encoder(deepcopy(value))
            except Exception as exc:
                self._diagnose(
                    diagnostics,
                    "encoder_failed",
                    f"registered encoder failed with {type(exc).__name__}",
                    path,
                    semantic_type,
                    severity=DiagnosticSeverity.ERROR,
                )
                return None, False, False
            return self._normalize(
                encoded,
                path=path,
                semantic_type=semantic_type,
                depth=depth,
                diagnostics=diagnostics,
                encoder_types=encoder_types + (value_type,),
            )

        if value is None or isinstance(value, (bool, int)):
            return value, True, False
        if isinstance(value, float):
            if isfinite(value):
                return value, True, False
            self._diagnose(
                diagnostics,
                "non_finite_number",
                "non-finite numbers cannot be captured",
                path,
                semantic_type,
                severity=DiagnosticSeverity.ERROR,
            )
            return None, False, False
        if isinstance(value, str):
            redacted = self._redact(value, path, semantic_type, diagnostics)
            if len(redacted) <= self.policy.max_string_length:
                return redacted, True, redacted != value
            self._diagnose(
                diagnostics,
                "capture_string_truncated",
                "string exceeded the capture length limit",
                path,
                semantic_type,
                details={
                    "original_length": len(redacted),
                    "captured_length": self.policy.max_string_length,
                },
            )
            return redacted[: self.policy.max_string_length], True, True
        if isinstance(value, bytes):
            self._diagnose(
                diagnostics,
                "nested_binary_omitted",
                "nested binary values require a separate artifact reference",
                path,
                semantic_type,
            )
            return None, True, True
        if isinstance(value, BaseException):
            return self._normalize(
                {"type": type(value).__name__, "message": str(value)},
                path=path,
                semantic_type=semantic_type,
                depth=depth + 1,
                diagnostics=diagnostics,
                encoder_types=encoder_types,
            )
        if isinstance(value, BaseModel):
            return self._normalize(
                value.model_dump(mode="python", round_trip=True, warnings=False),
                path=path,
                semantic_type=semantic_type,
                depth=depth + 1,
                diagnostics=diagnostics,
                encoder_types=encoder_types,
            )
        if is_dataclass(value) and not isinstance(value, type):
            return self._normalize(
                {field.name: getattr(value, field.name) for field in fields(value)},
                path=path,
                semantic_type=semantic_type,
                depth=depth + 1,
                diagnostics=diagnostics,
                encoder_types=encoder_types,
            )
        if isinstance(value, Mapping):
            normalized: dict[str, SerializedValue] = {}
            truncated = False
            for index, (key, item) in enumerate(value.items()):
                if index >= self.policy.max_collection_items:
                    self._diagnose(
                        diagnostics,
                        "capture_collection_truncated",
                        "mapping exceeded the capture item limit",
                        path,
                        semantic_type,
                        details={
                            "original_items": len(value),
                            "captured_items": self.policy.max_collection_items,
                        },
                    )
                    truncated = True
                    break
                if not isinstance(key, str):
                    self._diagnose(
                        diagnostics,
                        "non_string_key",
                        "mapping entry was omitted because its key was not a string",
                        path,
                        semantic_type,
                    )
                    truncated = True
                    continue
                child_path = path + (key,)
                allowed, reason = self.policy.allows_path(child_path)
                if not allowed:
                    self._diagnose(
                        diagnostics,
                        f"capture_{reason}",
                        f"capture path was {reason.replace('_', ' ')} by policy",
                        child_path,
                        semantic_type,
                    )
                    truncated = True
                    continue
                child, supported, child_truncated = self._normalize(
                    item,
                    path=child_path,
                    semantic_type=semantic_type,
                    depth=depth + 1,
                    diagnostics=diagnostics,
                    encoder_types=encoder_types,
                )
                if supported:
                    normalized[key] = child
                else:
                    truncated = True
                truncated = truncated or child_truncated
            return normalized, True, truncated
        if isinstance(value, Sequence):
            normalized_items: list[SerializedValue] = []
            truncated = False
            for index, item in enumerate(value):
                if index >= self.policy.max_collection_items:
                    self._diagnose(
                        diagnostics,
                        "capture_collection_truncated",
                        "sequence exceeded the capture item limit",
                        path,
                        semantic_type,
                        details={
                            "original_items": len(value),
                            "captured_items": self.policy.max_collection_items,
                        },
                    )
                    truncated = True
                    break
                child, supported, child_truncated = self._normalize(
                    item,
                    path=path + (str(index),),
                    semantic_type=semantic_type,
                    depth=depth + 1,
                    diagnostics=diagnostics,
                    encoder_types=encoder_types,
                )
                if supported:
                    normalized_items.append(child)
                else:
                    truncated = True
                truncated = truncated or child_truncated
            return normalized_items, True, truncated
        if isinstance(value, Enum):
            return self._normalize(
                value.value,
                path=path,
                semantic_type=semantic_type,
                depth=depth,
                diagnostics=diagnostics,
                encoder_types=encoder_types,
            )
        if isinstance(value, (datetime, date)):
            return value.isoformat(), True, False
        if isinstance(value, (Decimal, UUID, Path)):
            return str(value), True, False

        self._diagnose(
            diagnostics,
            "unknown_capture_type",
            f"no capture encoder is registered for {type(value).__name__}",
            path,
            semantic_type,
            severity=DiagnosticSeverity.ERROR,
        )
        return None, False, False

    def _capture_binary(
        self,
        value: bytes,
        *,
        level: CaptureLevel,
        path: tuple[str, ...],
        semantic_type: SemanticType | None,
        diagnostics: list[Diagnostic],
        media_type: str | None,
    ) -> CaptureResult:
        metadata: dict[str, SerializedValue] = {
            "type": "bytes",
            "size_bytes": len(value),
            "sha256": sha256(value).hexdigest(),
        }
        if self.policy.is_secret(path):
            metadata.pop("sha256")
            self._diagnose(
                diagnostics,
                "secret_redacted",
                "secret-bearing binary field was not persisted",
                path,
                semantic_type,
            )
            return CaptureResult(
                level=level,
                value=metadata,
                diagnostics=tuple(diagnostics),
                omitted=True,
            )
        if level is CaptureLevel.HASH:
            return CaptureResult(level=level, value=metadata)
        if not self.policy.store_binary:
            self._diagnose(
                diagnostics,
                "binary_content_omitted",
                "binary artifact storage is disabled",
                path,
                semantic_type,
            )
            return CaptureResult(
                level=level,
                value=metadata,
                diagnostics=tuple(diagnostics),
                omitted=True,
            )
        if len(value) > self.policy.max_artifact_bytes:
            self._diagnose(
                diagnostics,
                "artifact_too_large",
                "binary value exceeded the artifact size limit",
                path,
                semantic_type,
                severity=DiagnosticSeverity.ERROR,
                details={
                    "size_bytes": len(value),
                    "max_artifact_bytes": self.policy.max_artifact_bytes,
                },
            )
            return CaptureResult(
                level=level,
                value=metadata,
                diagnostics=tuple(diagnostics),
                omitted=True,
            )
        reference = self.store.add_artifact(
            value,
            media_type=media_type or "application/octet-stream",
        )
        self._diagnose(
            diagnostics,
            "binary_content_referenced",
            "binary value was stored as an artifact reference",
            path,
            semantic_type,
            details={"size_bytes": len(value)},
        )
        return CaptureResult(
            level=level,
            value=metadata,
            reference=reference,
            diagnostics=tuple(diagnostics),
        )

    def _redact(
        self,
        value: str,
        path: tuple[str, ...],
        semantic_type: SemanticType | None,
        diagnostics: list[Diagnostic],
    ) -> str:
        if self.policy.is_secret(path):
            self._diagnose(
                diagnostics,
                "secret_redacted",
                "secret-bearing field was redacted",
                path,
                semantic_type,
            )
            return "[REDACTED]"
        redacted = value
        for redactor in self.redactors:
            try:
                redacted = redactor(redacted, path, semantic_type)
            except Exception as exc:
                self._diagnose(
                    diagnostics,
                    "redactor_failed",
                    f"custom redactor failed with {type(exc).__name__}",
                    path,
                    semantic_type,
                    severity=DiagnosticSeverity.ERROR,
                )
        if redacted != value:
            self._diagnose(
                diagnostics,
                "content_redacted",
                "custom redactors changed captured content",
                path,
                semantic_type,
            )
        return redacted

    def _metadata(
        self,
        original: Any,
        normalized: SerializedValue,
        semantic_type: SemanticType | None,
    ) -> SerializedValue:
        if original is None or isinstance(original, (bool, int, float)):
            return normalized
        if (
            isinstance(normalized, str)
            and semantic_type is not None
            and any(
                semantic_type == prefix or semantic_type.startswith(f"{prefix}.")
                for prefix in _SAFE_STRING_SEMANTICS
            )
        ):
            return normalized
        metadata: dict[str, SerializedValue] = {"type": type(original).__name__}
        if isinstance(original, (str, bytes, Mapping, Sequence)):
            metadata["length"] = len(original)
        return metadata

    def _path(
        self,
        path: str | tuple[str, ...] | None,
        semantic_type: SemanticType | None,
    ) -> tuple[str, ...]:
        if isinstance(path, tuple):
            return path
        if isinstance(path, str):
            return tuple(part for part in path.split(".") if part)
        if semantic_type is not None:
            return tuple(semantic_type.split("."))
        return ("value",)

    def _asset_kind(self, semantic_type: SemanticType | None) -> ReferenceKind:
        if semantic_type is None:
            return ReferenceKind.ASSET
        if semantic_type == "prompt" or semantic_type.startswith("prompt."):
            return ReferenceKind.PROMPT
        if semantic_type == "tool" or semantic_type.startswith("tool."):
            return ReferenceKind.TOOL
        if semantic_type == "output_schema" or semantic_type.startswith("output_schema."):
            return ReferenceKind.OUTPUT_SCHEMA
        return ReferenceKind.ASSET

    def _diagnose(
        self,
        diagnostics: list[Diagnostic],
        code: str,
        message: str,
        path: tuple[str, ...],
        semantic_type: SemanticType | None,
        *,
        severity: DiagnosticSeverity = DiagnosticSeverity.WARNING,
        details: dict[str, SerializedValue] | None = None,
    ) -> None:
        diagnostics.append(
            Diagnostic(
                code=code,
                message=message,
                severity=severity,
                path=".".join(path),
                semantic_type=semantic_type,
                details={} if details is None else details,
            )
        )


__all__ = (
    "CapturePolicy",
    "CaptureResult",
    "CaptureSession",
    "EncoderRegistry",
    "Redactor",
)
