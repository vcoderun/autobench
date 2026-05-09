from __future__ import annotations as _annotations

import hashlib
import inspect
from collections.abc import Callable, Mapping
from dataclasses import MISSING, is_dataclass
from dataclasses import Field as DataclassField
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, cast, get_args, get_origin, get_type_hints

from pydantic import BaseModel

from autobench.metrics.semantics import SemanticType

from .models import (
    FieldAsset,
    ParamAsset,
    ParamSchema,
    SerializedValue,
    ToolAsset,
    TypeAsset,
)

if TYPE_CHECKING:
    from .registry import TrackingRegistry

_StructuredTypeKind = Literal["pydantic_model", "dataclass", "typed_class"]
_ParamKind = Literal["positional", "keyword", "var_positional", "var_keyword"]


def _build_tool_asset(
    target: Callable[..., Any],
    *,
    name: str,
    semantic_type: SemanticType | None,
    metadata: dict[str, SerializedValue],
    registry: TrackingRegistry,
) -> ToolAsset:
    signature = inspect.signature(target)
    type_hints = _safe_type_hints(target)
    return_type = type_hints.get("return", signature.return_annotation)
    return_type_name = _type_name(return_type)
    return_type_asset_id = _tracked_type_asset_id(return_type, registry)
    return ToolAsset(
        id=f"tool.{name}",
        kind="tool",
        name=name,
        semantic_type=semantic_type,
        metadata=metadata,
        qualname=getattr(target, "__qualname__", None),
        doc=inspect.getdoc(target),
        param_schema=ParamSchema(
            params=tuple(
                _build_param_asset(parameter, type_hints)
                for parameter in signature.parameters.values()
            )
        ),
        return_annotation=_annotation_label(return_type),
        return_type_name=return_type_name,
        return_type_asset_id=return_type_asset_id,
    )


def _build_type_asset(
    target: type[Any],
    *,
    name: str,
    semantic_type: SemanticType | None,
    metadata: dict[str, SerializedValue],
) -> TypeAsset:
    type_kind = _structured_type_kind(target)
    return TypeAsset(
        id=f"type.{name}",
        kind="type",
        name=name,
        semantic_type=semantic_type,
        metadata=metadata,
        qualname=target.__qualname__,
        doc=inspect.getdoc(target),
        type_kind=type_kind,
        field_assets=tuple(_field_assets_for_type(target, type_kind)),
    )


def _build_param_asset(
    parameter: inspect.Parameter,
    type_hints: dict[str, Any],
) -> ParamAsset:
    annotation = type_hints.get(parameter.name, parameter.annotation)
    return ParamAsset(
        name=parameter.name,
        annotation=_annotation_label(annotation),
        required=parameter.default is inspect.Parameter.empty,
        default=(
            None
            if parameter.default is inspect.Parameter.empty
            else _normalize_value(parameter.default)
        ),
        kind=_parameter_kind(parameter.kind),
        literal_choices=tuple(_literal_choices(annotation, {})),
    )


def _field_assets_for_type(
    target: type[Any],
    type_kind: _StructuredTypeKind,
) -> list[FieldAsset]:
    if type_kind == "pydantic_model":
        return _pydantic_field_assets(target)
    if type_kind == "dataclass":
        return _dataclass_field_assets(target)
    return _typed_class_field_assets(target)


def _pydantic_field_assets(target: type[Any]) -> list[FieldAsset]:
    if not issubclass(target, BaseModel):
        return []
    schema = target.model_json_schema()
    required_fields = set(schema.get("required", []))
    property_schemas = schema.get("properties", {})
    field_assets: list[FieldAsset] = []
    for field_name, model_field in target.model_fields.items():
        property_schema = property_schemas.get(field_name, {})
        field_assets.append(
            FieldAsset(
                name=field_name,
                annotation=_annotation_label(model_field.annotation),
                required=field_name in required_fields,
                default=(
                    None
                    if model_field.is_required()
                    else _normalize_value(model_field.get_default(call_default_factory=False))
                ),
                description=(
                    property_schema["description"]
                    if isinstance(property_schema.get("description"), str)
                    else None
                ),
                examples=tuple(
                    _normalize_value(item)
                    for item in property_schema.get("examples", [])
                    if isinstance(property_schema.get("examples"), list)
                ),
                alias=model_field.alias if isinstance(model_field.alias, str) else None,
                constraints=_schema_constraints(property_schema),
                literal_choices=tuple(_literal_choices(model_field.annotation, property_schema)),
            )
        )
    return field_assets


def _dataclass_field_assets(target: type[Any]) -> list[FieldAsset]:
    type_hints = _safe_type_hints(target)
    field_assets: list[FieldAsset] = []
    for dataclass_field in _dataclass_field_values(target):
        annotation = type_hints.get(dataclass_field.name, dataclass_field.type)
        default_factory = None
        if dataclass_field.default_factory is not MISSING:  # type: ignore[comparison-overlap]
            default_factory = _callable_name(dataclass_field.default_factory)
        field_assets.append(
            FieldAsset(
                name=dataclass_field.name,
                annotation=_annotation_label(annotation),
                required=(
                    dataclass_field.default is MISSING
                    and dataclass_field.default_factory is MISSING  # type: ignore[comparison-overlap]
                ),
                default=(
                    None
                    if dataclass_field.default is MISSING
                    else _normalize_value(dataclass_field.default)
                ),
                default_factory=default_factory,
                metadata=_normalize_mapping(dict(dataclass_field.metadata)),
                init=dataclass_field.init,
                kw_only=bool(dataclass_field.kw_only),
                compare=dataclass_field.compare,
                repr=dataclass_field.repr,
                literal_choices=tuple(_literal_choices(annotation, {})),
            )
        )
    return field_assets


def _typed_class_field_assets(target: type[Any]) -> list[FieldAsset]:
    type_hints = _safe_type_hints(target)
    field_assets: list[FieldAsset] = []
    for field_name, annotation in getattr(target, "__annotations__", {}).items():
        resolved_annotation = type_hints.get(field_name, annotation)
        default = getattr(target, field_name, MISSING)
        field_assets.append(
            FieldAsset(
                name=field_name,
                annotation=_annotation_label(resolved_annotation),
                required=default is MISSING,
                default=None if default is MISSING else _normalize_value(default),
                literal_choices=tuple(_literal_choices(resolved_annotation, {})),
            )
        )
    return field_assets


def _structured_type_kind(target: type[Any]) -> _StructuredTypeKind:
    if issubclass(target, BaseModel):
        return "pydantic_model"
    if is_dataclass(target):
        return "dataclass"
    return "typed_class"


def _parameter_kind(kind: inspect._ParameterKind) -> _ParamKind:
    if kind in (
        inspect.Parameter.POSITIONAL_ONLY,
        inspect.Parameter.POSITIONAL_OR_KEYWORD,
    ):
        return "positional"
    if kind is inspect.Parameter.KEYWORD_ONLY:
        return "keyword"
    if kind is inspect.Parameter.VAR_POSITIONAL:
        return "var_positional"
    return "var_keyword"


def _tracked_type_asset_id(
    annotation: Any,
    registry: TrackingRegistry,
) -> str | None:
    if isinstance(annotation, type):
        asset = registry._assets_by_target_id.get(id(annotation))
        if isinstance(asset, TypeAsset):
            return asset.id
    annotation_name = _type_name(annotation)
    if annotation_name is None:
        return None
    named_asset = registry.assets.get(annotation_name)
    if isinstance(named_asset, TypeAsset):
        return named_asset.id
    return None


def _safe_type_hints(target: Any) -> dict[str, Any]:
    globalns = getattr(target, "__globals__", None)
    localns: dict[str, Any] | None = None
    if callable(target):
        try:
            closure_vars = inspect.getclosurevars(target)
        except TypeError:
            closure_vars = None
        if closure_vars is not None:
            localns = {
                **closure_vars.globals,
                **closure_vars.nonlocals,
                **closure_vars.builtins,
            }
    try:
        return get_type_hints(target, globalns=globalns, localns=localns)
    except (NameError, TypeError, AttributeError):
        return {}


def _hash_structured_type(target: type[Any]) -> str:
    if issubclass(target, BaseModel):
        return _hash_serialized(target.model_json_schema())
    if is_dataclass(target):
        return _hash_serialized(_dataclass_snapshot(target))
    return _hash_serialized(_plain_class_snapshot(target))


def _dataclass_snapshot(target: type[Any]) -> dict[str, SerializedValue]:
    type_hints = _safe_type_hints(target)
    return {
        "name": target.__name__,
        "qualname": target.__qualname__,
        "doc": inspect.getdoc(target),
        "fields": [
            {
                "name": dataclass_field.name,
                "annotation": _annotation_label(
                    type_hints.get(dataclass_field.name, dataclass_field.type)
                ),
                "default": (
                    None
                    if dataclass_field.default is MISSING
                    else _normalize_value(dataclass_field.default)
                ),
                "default_factory": (
                    None
                    if dataclass_field.default_factory is MISSING  # type: ignore[comparison-overlap]
                    else _callable_name(dataclass_field.default_factory)
                ),
                "init": dataclass_field.init,
                "kw_only": bool(dataclass_field.kw_only),
                "repr": dataclass_field.repr,
                "compare": dataclass_field.compare,
                "metadata": _normalize_mapping(dict(dataclass_field.metadata)),
            }
            for dataclass_field in _dataclass_field_values(target)
        ],
    }


def _dataclass_field_values(target: type[Any]) -> tuple[DataclassField[Any], ...]:
    dataclass_mapping = getattr(target, "__dataclass_fields__", None)
    if not isinstance(dataclass_mapping, dict):
        return ()
    field_values: list[DataclassField[Any]] = []
    for field_value in dataclass_mapping.values():
        if isinstance(field_value, DataclassField):
            field_values.append(field_value)
    return tuple(field_values)


def _plain_class_snapshot(target: type[Any]) -> dict[str, SerializedValue]:
    init_signature = None
    try:
        init_signature = str(inspect.signature(target.__init__))
    except (TypeError, ValueError):
        init_signature = None
    annotations = getattr(target, "__annotations__", {})
    type_hints = _safe_type_hints(target)
    return {
        "name": target.__name__,
        "qualname": target.__qualname__,
        "doc": inspect.getdoc(target),
        "annotations": {
            field_name: _annotation_label(type_hints.get(field_name, annotation)) or "unknown"
            for field_name, annotation in annotations.items()
        },
        "init_signature": init_signature,
    }


def _source_text(target: Any) -> str | None:
    if not callable(target) and not isinstance(target, type):
        return None
    try:
        return inspect.getsource(cast(Callable[..., Any] | type[Any], target))
    except (OSError, TypeError):
        return None


def _source_hash(target: Any) -> str | None:
    source = _source_text(target)
    if source is None:
        return None
    return _hash_text(source)


def _hash_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _hash_serialized(value: Any) -> str:
    return _hash_text(repr(_normalize_value(value)))


def _safe_filename(value: str) -> str:
    return "".join(char if char.isalnum() or char in ("-", "_") else "_" for char in value)


def _source_path(target: Any) -> str | None:
    if not callable(target) and not isinstance(target, type):
        return None
    try:
        return inspect.getsourcefile(cast(Callable[..., Any] | type[Any], target))
    except TypeError:
        return None


def _annotation_label(annotation: Any) -> str | None:
    if annotation in {inspect.Signature.empty, inspect.Parameter.empty}:
        return None
    if isinstance(annotation, str):
        return annotation
    origin = get_origin(annotation)
    if origin is not None:
        return str(annotation).replace("typing.", "")
    if isinstance(annotation, type):
        return annotation.__name__
    return str(annotation)


def _type_name(annotation: Any) -> str | None:
    if annotation in {inspect.Signature.empty, inspect.Parameter.empty}:
        return None
    if isinstance(annotation, type):
        return annotation.__name__
    return _annotation_label(annotation)


def _literal_choices(
    annotation: Any,
    schema: dict[str, Any],
) -> list[SerializedValue]:
    origin = get_origin(annotation)
    if origin is Literal:
        return [_normalize_value(choice) for choice in get_args(annotation)]
    schema_enum = schema.get("enum")
    if isinstance(schema_enum, list):
        return [_normalize_value(choice) for choice in schema_enum]
    return []


def _schema_constraints(schema: dict[str, Any]) -> dict[str, SerializedValue]:
    excluded = {
        "title",
        "description",
        "examples",
        "default",
        "type",
        "enum",
        "anyOf",
        "oneOf",
        "allOf",
        "items",
        "properties",
        "$ref",
    }
    return {key: _normalize_value(value) for key, value in schema.items() if key not in excluded}


def _normalize_value(value: Any) -> SerializedValue:
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return _normalize_mapping(cast(Mapping[Any, Any], value))
    if isinstance(value, (tuple, list, set, frozenset)):
        return [_normalize_value(item) for item in value]
    if isinstance(value, type):
        return value.__qualname__
    return repr(value)


def _normalize_mapping(value: Mapping[Any, Any]) -> dict[str, SerializedValue]:
    return {
        str(key): _normalize_value(item)
        for key, item in sorted(value.items(), key=lambda item: str(item[0]))
    }


def _callable_name(target: Any) -> str:
    name = getattr(target, "__name__", None)
    if isinstance(name, str):
        return name
    qualname = getattr(target, "__qualname__", None)
    if isinstance(qualname, str):
        return qualname
    return repr(target)


__all__ = (
    "_annotation_label",
    "_build_tool_asset",
    "_build_type_asset",
    "_callable_name",
    "_hash_serialized",
    "_hash_text",
    "_normalize_mapping",
    "_normalize_value",
    "_safe_filename",
    "_source_hash",
    "_source_path",
)
