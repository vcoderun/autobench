from __future__ import annotations as _annotations

from enum import StrEnum
from typing import Literal, ParamSpec, Protocol, TypeVar

from pydantic import BaseModel, ConfigDict, Field, JsonValue

from autobench.metrics.semantics import SemanticType

_TypeT = TypeVar("_TypeT", bound=type)
_DecoratorParamT = ParamSpec("_DecoratorParamT")
_StructuredTypeKind = Literal["pydantic_model", "dataclass", "typed_class"]
_ParamKind = Literal["positional", "keyword", "var_positional", "var_keyword"]
SerializedValue = JsonValue


class AssetRepresentation(StrEnum):
    DEFINITION = "definition"
    EFFECTIVE = "effective"


class AssetSensitivity(StrEnum):
    PUBLIC = "public"
    INTERNAL = "internal"
    SENSITIVE = "sensitive"


class AssetProvenance(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    system: str = Field(min_length=1)
    key: str = Field(min_length=1)
    path: tuple[str | int, ...] = ()
    instrumentor: str | None = Field(default=None, min_length=1)
    instrumented_library_version: str | None = Field(default=None, min_length=1)


class TypeDecorator(Protocol[_DecoratorParamT]):
    def __call__(
        self,
        cls: _TypeT,
        /,
        *args: _DecoratorParamT.args,
        **kwargs: _DecoratorParamT.kwargs,
    ) -> _TypeT: ...


class ParamAsset(BaseModel):
    model_config = ConfigDict(frozen=True)

    name: str
    annotation: str | None = None
    required: bool
    default: SerializedValue = None
    kind: _ParamKind
    literal_choices: tuple[SerializedValue, ...] = ()


class ParamSchema(BaseModel):
    model_config = ConfigDict(frozen=True)

    params: tuple[ParamAsset, ...] = ()


class FieldAsset(BaseModel):
    model_config = ConfigDict(frozen=True)

    name: str
    annotation: str | None = None
    required: bool
    default: SerializedValue = None
    default_factory: str | None = None
    description: str | None = None
    examples: tuple[SerializedValue, ...] = ()
    alias: str | None = None
    constraints: dict[str, SerializedValue] = Field(default_factory=dict)
    literal_choices: tuple[SerializedValue, ...] = ()
    metadata: dict[str, SerializedValue] = Field(default_factory=dict)
    init: bool | None = None
    kw_only: bool | None = None
    compare: bool | None = None
    repr: bool | None = None


class TrackedAsset(BaseModel):
    model_config = ConfigDict(frozen=True)

    id: str
    kind: str
    name: str
    semantic_type: SemanticType | None = None
    metadata: dict[str, SerializedValue] = Field(default_factory=dict)


class AssetDefinition(TrackedAsset):
    model_config = ConfigDict(frozen=True)

    representation: AssetRepresentation = AssetRepresentation.DEFINITION
    canonical_content: SerializedValue
    scope: str | None = None
    owner_locator: str | None = None
    source_locators: tuple[str, ...] = ()
    aliases: tuple[str, ...] = ()
    sensitivity: AssetSensitivity = AssetSensitivity.INTERNAL


class ToolAsset(TrackedAsset):
    model_config = ConfigDict(frozen=True)

    qualname: str | None = None
    doc: str | None = None
    param_schema: ParamSchema = ParamSchema()
    return_annotation: str | None = None
    return_type_name: str | None = None
    return_type_asset_id: str | None = None


class TypeAsset(TrackedAsset):
    model_config = ConfigDict(frozen=True)

    qualname: str | None = None
    doc: str | None = None
    type_kind: _StructuredTypeKind
    field_assets: tuple[FieldAsset, ...] = ()


class AssetVersion(BaseModel):
    model_config = ConfigDict(frozen=True)

    asset_id: str
    version: str
    content_hash: str
    source_hash: str | None = None
    source_path: str | None = None
    git_commit: str | None = None
    parent_version: str | None = None
    metadata: dict[str, SerializedValue] = Field(default_factory=dict)

    @property
    def hash(self) -> str:
        return self.content_hash


class AssetContentRef(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    asset_id: str = Field(min_length=1)
    version: str = Field(min_length=1)
    path: str = Field(min_length=1)


class AssetDiffRef(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    asset_id: str = Field(min_length=1)
    version: str = Field(min_length=1)
    parent_version: str = Field(min_length=1)
    path: str = Field(min_length=1)


class TrackedPrompt(BaseModel):
    model_config = ConfigDict(frozen=True)

    asset: TrackedAsset
    version: str
    text: str

    @property
    def raw(self) -> str:
        return self.text

    def __str__(self) -> str:
        return self.raw


__all__ = (
    "AssetContentRef",
    "AssetDefinition",
    "AssetDiffRef",
    "AssetProvenance",
    "AssetRepresentation",
    "AssetSensitivity",
    "AssetVersion",
    "FieldAsset",
    "ParamAsset",
    "ParamSchema",
    "SerializedValue",
    "TrackedAsset",
    "TrackedPrompt",
    "ToolAsset",
    "TypeAsset",
    "TypeDecorator",
)
