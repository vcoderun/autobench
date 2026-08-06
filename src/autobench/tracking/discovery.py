from __future__ import annotations

import inspect
from collections.abc import Callable, Mapping, Sequence
from dataclasses import fields, is_dataclass
from functools import lru_cache
from hashlib import sha256
from json import dumps
from types import BuiltinFunctionType, FunctionType, MethodType
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter

from autobench.metrics.semantics import SemanticType

from .introspection import _normalize_value, _source_hash
from .models import (
    AssetProvenance,
    AssetRepresentation,
    AssetSensitivity,
    AssetVersion,
    SerializedValue,
    TrackedAsset,
)


class AssetCandidate(BaseModel):
    """Normalized asset definition observed at an instrumented SDK boundary."""

    model_config = ConfigDict(frozen=True, extra="forbid", arbitrary_types_allowed=True)

    kind: str = Field(min_length=1)
    local_id: str = Field(min_length=1)
    name: str = Field(min_length=1)
    source_locator: str = Field(min_length=1)
    canonical_content: SerializedValue
    content_fingerprint: str | None = Field(default=None, min_length=64, max_length=64)
    representation: AssetRepresentation = AssetRepresentation.DEFINITION
    semantic_type: SemanticType | None = None
    scope: str | None = Field(default=None, min_length=1)
    python_target: Any = Field(default=None, exclude=True, repr=False)
    explicit_asset_id: str | None = Field(default=None, min_length=1)
    owner_locator: str | None = Field(default=None, min_length=1)
    definition_locator: str | None = Field(default=None, min_length=1)
    provenance: AssetProvenance
    aliases: tuple[str, ...] = ()
    metadata: dict[str, SerializedValue] = Field(default_factory=dict)
    sensitivity: AssetSensitivity = AssetSensitivity.INTERNAL


class AssetUse(BaseModel):
    """One run-local use of a tracked asset version."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    asset_id: str = Field(min_length=1)
    version: str = Field(min_length=1)
    representation: AssetRepresentation
    source_locator: str = Field(min_length=1)
    scope: str | None = Field(default=None, min_length=1)
    span_id: str | None = Field(default=None, min_length=1)
    definition_asset_id: str | None = Field(default=None, min_length=1)
    definition_version: str | None = Field(default=None, min_length=1)
    provenance: AssetProvenance
    aliases: tuple[str, ...] = ()


class RegisteredAsset(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    asset: TrackedAsset
    version: AssetVersion
    use: AssetUse


_SERIALIZED_VALUE = TypeAdapter(SerializedValue)


def canonical_asset_content(value: Any) -> SerializedValue:
    """Convert common SDK asset values into deterministic serialized content."""

    if isinstance(value, BaseModel):
        normalized = value.model_dump(mode="json")
    elif isinstance(value, type) and issubclass(value, BaseModel):
        normalized = _pydantic_schema(value)
    elif is_dataclass(value) and not isinstance(value, type):
        normalized = {
            field.name: canonical_asset_content(getattr(value, field.name))
            for field in fields(value)
        }
    elif isinstance(value, Mapping):
        normalized = {
            str(key): canonical_asset_content(item)
            for key, item in sorted(value.items(), key=lambda item: str(item[0]))
        }
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        normalized = [canonical_asset_content(item) for item in value]
    elif isinstance(value, type):
        normalized = {
            "module": value.__module__,
            "qualname": value.__qualname__,
            "value": _normalize_value(value),
        }
    elif isinstance(value, FunctionType):
        normalized = _function_content(value)
    elif isinstance(value, MethodType):
        normalized = _callable_content(
            value,
            module=value.__module__,
            qualname=value.__qualname__,
            source_target=value.__func__,
        )
    elif isinstance(value, BuiltinFunctionType):
        normalized = _callable_content(
            value,
            module=value.__module__,
            qualname=value.__qualname__,
            source_target=value,
        )
    elif callable(value):
        callable_value: Callable[..., Any] = value
        callable_type = type(callable_value)
        normalized = _callable_content(
            callable_value,
            module=callable_type.__module__,
            qualname=callable_type.__qualname__,
            source_target=callable_type,
        )
    else:
        normalized = _normalize_value(value)
    return _SERIALIZED_VALUE.validate_python(normalized)


def canonical_asset_hash(value: SerializedValue) -> str:
    payload = dumps(value, ensure_ascii=True, separators=(",", ":"), sort_keys=True).encode()
    return sha256(payload).hexdigest()


@lru_cache(maxsize=128)
def _pydantic_schema(model: type[BaseModel]) -> SerializedValue:
    return _SERIALIZED_VALUE.validate_python(model.model_json_schema(mode="validation"))


@lru_cache(maxsize=256)
def _function_content(function: FunctionType) -> SerializedValue:
    return _callable_content(
        function,
        module=function.__module__,
        qualname=function.__qualname__,
        source_target=function,
    )


def _callable_content(
    value: Callable[..., Any],
    *,
    module: str | None,
    qualname: str,
    source_target: type[Any] | Callable[..., Any],
) -> SerializedValue:
    try:
        signature = str(inspect.signature(value))
    except (TypeError, ValueError):
        signature = None
    return _SERIALIZED_VALUE.validate_python(
        {
            "module": module,
            "qualname": qualname,
            "signature": signature,
            "source_hash": _source_hash(source_target),
        }
    )


__all__ = (
    "AssetCandidate",
    "AssetUse",
    "RegisteredAsset",
    "canonical_asset_content",
    "canonical_asset_hash",
)
