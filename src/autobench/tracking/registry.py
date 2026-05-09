from __future__ import annotations as _annotations

from collections.abc import Callable
from dataclasses import dataclass as stdlib_dataclass
from dataclasses import field as stdlib_field
from pathlib import Path
from typing import Any, ParamSpec, TypeVar, cast, dataclass_transform, overload

from autobench.io import dump_yaml, load_yaml
from autobench.metrics.semantics import Semantic, SemanticType

from .history import asset_index_to_yaml_view, asset_to_yaml_view
from .introspection import (
    _build_tool_asset,
    _build_type_asset,
    _callable_name,
    _hash_serialized,
    _hash_structured_type,
    _hash_text,
    _normalize_value,
    _safe_filename,
    _source_hash,
    _source_path,
)
from .models import AssetVersion, SerializedValue, TrackedAsset, TrackedPrompt, TypeDecorator

_T = TypeVar("_T")
_TypeT = TypeVar("_TypeT", bound=type)
_ParamT = ParamSpec("_ParamT")
_DecoratorParamT = ParamSpec("_DecoratorParamT")
_ReturnT = TypeVar("_ReturnT")


class TrackingRegistry:
    def __init__(self) -> None:
        self._versions_by_target_id: dict[int, AssetVersion] = {}
        self._assets_by_target_id: dict[int, TrackedAsset] = {}
        self._assets_by_name: dict[str, TrackedAsset] = {}
        self._latest_versions_by_asset_id: dict[str, AssetVersion] = {}
        self._version_history: list[AssetVersion] = []

    @property
    def assets(self) -> dict[str, TrackedAsset]:
        return dict(self._assets_by_name)

    @property
    def versions(self) -> tuple[AssetVersion, ...]:
        return tuple(self._version_history)

    def write_assets(self, directory: Path) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        assets = sorted(self._assets_by_name.values(), key=lambda asset: asset.id)
        versions = [self._version_for_asset_id(asset.id) for asset in assets]
        for asset, version in zip(assets, versions, strict=True):
            asset_path = directory / f"{_safe_filename(asset.id)}.yaml"
            existing = load_yaml(asset_path) if asset_path.exists() else None
            dump_yaml(
                asset_to_yaml_view(asset, version, existing=existing),
                asset_path,
                schema_name="asset",
            )
        dump_yaml(
            asset_index_to_yaml_view(assets, versions),
            directory / "index.yaml",
            schema_name="asset_index",
        )

    def _version_for_asset_id(self, asset_id: str) -> AssetVersion:
        try:
            return self._latest_versions_by_asset_id[asset_id]
        except KeyError as exc:
            raise KeyError(f"Asset version is missing for {asset_id}.") from exc

    def asset(
        self,
        *,
        kind: str,
        name: str,
        semantic_type: SemanticType | None = None,
        version: str | None = None,
        hash: str | None = None,
        source_path: str | Path | None = None,
        parent_version: str | None = None,
        metadata: dict[str, SerializedValue] | None = None,
        source_hash: str | None = None,
    ) -> Callable[[_T], _T]:
        def decorator(target: _T) -> _T:
            asset = TrackedAsset(
                id=f"{kind}.{name}",
                kind=kind,
                name=name,
                semantic_type=semantic_type,
                metadata=dict(metadata or {}),
            )
            content_hash = hash or _source_hash(target) or _hash_text(repr(target))
            version_record = AssetVersion(
                asset_id=asset.id,
                version=version or content_hash[:12],
                content_hash=content_hash,
                source_hash=source_hash or _source_hash(target),
                source_path=str(source_path) if source_path is not None else _source_path(target),
                parent_version=parent_version,
                metadata=dict(metadata or {}),
            )
            self._register(target, asset, version_record)
            return target

        return decorator

    @overload
    def tool(
        self,
        target: Callable[_ParamT, _ReturnT],
        *,
        name: str | None = None,
        semantic_type: SemanticType | None = Semantic.AGENT_TOOL_VERSION,
        version: str | None = None,
        source_path: str | Path | None = None,
        parent_version: str | None = None,
        metadata: dict[str, SerializedValue] | None = None,
    ) -> Callable[_ParamT, _ReturnT]: ...

    @overload
    def tool(
        self,
        target: None = None,
        *,
        name: str | None = None,
        semantic_type: SemanticType | None = Semantic.AGENT_TOOL_VERSION,
        version: str | None = None,
        source_path: str | Path | None = None,
        parent_version: str | None = None,
        metadata: dict[str, SerializedValue] | None = None,
    ) -> Callable[[Callable[_ParamT, _ReturnT]], Callable[_ParamT, _ReturnT]]: ...

    def tool(
        self,
        target: Callable[..., Any] | None = None,
        *,
        name: str | None = None,
        semantic_type: SemanticType | None = Semantic.AGENT_TOOL_VERSION,
        version: str | None = None,
        source_path: str | Path | None = None,
        parent_version: str | None = None,
        metadata: dict[str, SerializedValue] | None = None,
    ) -> Callable[..., Any] | Callable[[Callable[..., Any]], Callable[..., Any]]:
        def decorator(
            tool_target: Callable[_ParamT, _ReturnT],
        ) -> Callable[_ParamT, _ReturnT]:
            if not callable(tool_target):
                raise TypeError("@track.tool can only decorate callables.")
            tool_name = name or _callable_name(tool_target)
            tool_asset = _build_tool_asset(
                tool_target,
                name=tool_name,
                semantic_type=semantic_type,
                metadata=dict(metadata or {}),
                registry=self,
            )
            content_hash = _hash_serialized(tool_asset.model_dump(mode="python"))
            version_record = AssetVersion(
                asset_id=tool_asset.id,
                version=version or content_hash[:12],
                content_hash=content_hash,
                source_hash=_source_hash(tool_target),
                source_path=str(source_path)
                if source_path is not None
                else _source_path(tool_target),
                parent_version=parent_version,
                metadata=tool_asset.metadata,
            )
            self._register(tool_target, tool_asset, version_record)
            return tool_target

        if target is None:
            return decorator
        return decorator(target)

    @overload
    def type(
        self,
        target: _TypeT,
        *,
        name: str | None = None,
        semantic_type: SemanticType | None = None,
        version: str | None = None,
        source_path: str | Path | None = None,
        parent_version: str | None = None,
        metadata: dict[str, SerializedValue] | None = None,
    ) -> _TypeT: ...

    @overload
    def type(
        self,
        target: None = None,
        *,
        name: str | None = None,
        semantic_type: SemanticType | None = None,
        version: str | None = None,
        source_path: str | Path | None = None,
        parent_version: str | None = None,
        metadata: dict[str, SerializedValue] | None = None,
    ) -> Callable[[_TypeT], _TypeT]: ...

    def type(
        self,
        target: _TypeT | None = None,
        *,
        name: str | None = None,
        semantic_type: SemanticType | None = None,
        version: str | None = None,
        source_path: str | Path | None = None,
        parent_version: str | None = None,
        metadata: dict[str, SerializedValue] | None = None,
    ) -> _TypeT | Callable[[_TypeT], _TypeT]:
        def decorator(type_target: _TypeT) -> _TypeT:
            if not isinstance(type_target, type):
                raise TypeError("@track.type can only decorate classes.")
            type_name = name or type_target.__name__
            type_asset = _build_type_asset(
                type_target,
                name=type_name,
                semantic_type=semantic_type,
                metadata=dict(metadata or {}),
            )
            content_hash = _hash_structured_type(type_target)
            version_record = AssetVersion(
                asset_id=type_asset.id,
                version=version or content_hash[:12],
                content_hash=content_hash,
                source_hash=_source_hash(type_target),
                source_path=str(source_path)
                if source_path is not None
                else _source_path(type_target),
                parent_version=parent_version,
                metadata=type_asset.metadata,
            )
            self._register(type_target, type_asset, version_record)
            return type_target

        if target is None:
            return decorator
        return decorator(target)

    def decorate_type(
        self,
        class_decorator: TypeDecorator[_DecoratorParamT],
        /,
        *decorator_args: _DecoratorParamT.args,
        **decorator_kwargs: _DecoratorParamT.kwargs,
    ) -> Callable[[_TypeT], _TypeT]:
        decorator_metadata: dict[str, SerializedValue] = {
            "decorator": {
                "name": _callable_name(class_decorator),
                "module": (
                    class_decorator.__module__
                    if isinstance(class_decorator.__module__, str)
                    else None
                ),
                "args": [_normalize_value(value) for value in decorator_args],
                "kwargs": {
                    key: _normalize_value(value) for key, value in sorted(decorator_kwargs.items())
                },
            }
        }

        def decorator(type_target: _TypeT) -> _TypeT:
            decorated_target = class_decorator(type_target, *decorator_args, **decorator_kwargs)
            if not isinstance(decorated_target, type):
                raise TypeError(
                    "@track.decorate_type requires a class decorator that returns a class."
                )
            return self.type(decorated_target, metadata=decorator_metadata)

        return decorator

    @overload
    def dataclass(
        self,
        target: _TypeT,
        *,
        init: bool = True,
        repr: bool = True,
        eq: bool = True,
        order: bool = False,
        unsafe_hash: bool = False,
        frozen: bool = False,
        match_args: bool = True,
        kw_only: bool = False,
        slots: bool = False,
        weakref_slot: bool = False,
    ) -> _TypeT: ...

    @overload
    def dataclass(
        self,
        target: None = None,
        *,
        init: bool = True,
        repr: bool = True,
        eq: bool = True,
        order: bool = False,
        unsafe_hash: bool = False,
        frozen: bool = False,
        match_args: bool = True,
        kw_only: bool = False,
        slots: bool = False,
        weakref_slot: bool = False,
    ) -> Callable[[_TypeT], _TypeT]: ...

    @dataclass_transform(field_specifiers=(stdlib_field,))
    def dataclass(
        self,
        target: _TypeT | None = None,
        *,
        init: bool = True,
        repr: bool = True,
        eq: bool = True,
        order: bool = False,
        unsafe_hash: bool = False,
        frozen: bool = False,
        match_args: bool = True,
        kw_only: bool = False,
        slots: bool = False,
        weakref_slot: bool = False,
    ) -> _TypeT | Callable[[_TypeT], _TypeT]:
        decorator_metadata: dict[str, SerializedValue] = {
            "decorator": {
                "name": "dataclass",
                "module": "dataclasses",
                "args": [],
                "kwargs": {
                    key: _normalize_value(value)
                    for key, value in sorted(
                        {
                            "init": init,
                            "repr": repr,
                            "eq": eq,
                            "order": order,
                            "unsafe_hash": unsafe_hash,
                            "frozen": frozen,
                            "match_args": match_args,
                            "kw_only": kw_only,
                            "slots": slots,
                            "weakref_slot": weakref_slot,
                        }.items()
                    )
                },
            }
        }

        def decorator(type_target: _TypeT) -> _TypeT:
            dataclass_decorator = stdlib_dataclass(
                init=init,
                repr=repr,
                eq=eq,
                order=order,
                unsafe_hash=unsafe_hash,
                frozen=frozen,
                match_args=match_args,
                kw_only=kw_only,
                slots=slots,
                weakref_slot=weakref_slot,
            )
            decorated_target = dataclass_decorator(cast(type[Any], type_target))
            return self.type(cast(_TypeT, decorated_target), metadata=decorator_metadata)

        if target is None:
            return decorator
        return decorator(target)

    def prompt(
        self,
        *,
        name: str,
        text: str | None = None,
        source: str | Path | None = None,
        semantic_type: SemanticType | None = Semantic.PROMPT_VERSION,
        version: str | None = None,
        hash: str | None = None,
        parent_version: str | None = None,
        metadata: dict[str, SerializedValue] | None = None,
    ) -> TrackedPrompt:
        if (text is None) == (source is None):
            raise ValueError("track.prompt requires exactly one of 'text' or 'source'.")
        prompt_text = text
        prompt_source_path: str | None = None
        if source is not None:
            source_path = Path(source).expanduser().resolve()
            prompt_text = source_path.read_text(encoding="utf-8")
            prompt_source_path = str(source_path)
        assert prompt_text is not None
        prompt_metadata = dict(metadata or {})
        prompt_metadata["raw"] = prompt_text
        asset = TrackedAsset(
            id=f"prompt.{name}",
            kind="prompt",
            name=name,
            semantic_type=semantic_type,
            metadata=prompt_metadata,
        )
        content_hash = hash or _hash_text(prompt_text)
        prompt = TrackedPrompt(asset=asset, version=version or content_hash[:12], text=prompt_text)
        self._register(
            prompt,
            asset,
            AssetVersion(
                asset_id=asset.id,
                version=prompt.version,
                content_hash=content_hash,
                source_hash=content_hash if prompt_source_path is not None else None,
                source_path=prompt_source_path,
                parent_version=parent_version,
                metadata=prompt_metadata,
            ),
        )
        return prompt

    def asset_of(self, target: Any) -> TrackedAsset:
        try:
            return self._assets_by_target_id[id(target)]
        except KeyError as exc:
            raise KeyError("Object is not tracked by Autobench.") from exc

    def version_of(self, target: Any) -> str:
        return self.asset_version_of(target).version

    def asset_version_of(self, target: Any) -> AssetVersion:
        try:
            return self._versions_by_target_id[id(target)]
        except KeyError as exc:
            raise KeyError("Object is not tracked by Autobench.") from exc

    def _register(self, target: Any, asset: TrackedAsset, version: AssetVersion) -> None:
        target_id = id(target)
        self._assets_by_target_id[target_id] = asset
        self._versions_by_target_id[target_id] = version
        self._assets_by_name[asset.name] = asset
        self._latest_versions_by_asset_id[asset.id] = version
        self._version_history.append(version)


__all__ = ("TrackingRegistry", "track")

track = TrackingRegistry()
