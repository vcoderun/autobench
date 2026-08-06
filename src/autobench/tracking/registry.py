from __future__ import annotations as _annotations

from collections.abc import Callable, Collection
from dataclasses import dataclass as stdlib_dataclass
from dataclasses import field as stdlib_field
from pathlib import Path
from threading import RLock
from typing import Any, ParamSpec, TypeVar, cast, dataclass_transform, overload
from uuid import uuid4

from filelock import FileLock

from autobench.io import dump_yaml, load_yaml
from autobench.metrics.semantics import Semantic, SemanticType

from .discovery import AssetCandidate, AssetUse, RegisteredAsset
from .history import (
    _asset_version_changes,
    _asset_version_snapshot,
    asset_index_to_yaml_view,
    asset_to_yaml_view,
)
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
from .models import (
    AssetDefinition,
    AssetRepresentation,
    AssetVersion,
    SerializedValue,
    TrackedAsset,
    TrackedPrompt,
    TypeDecorator,
)
from .store import AssetContentStore

_T = TypeVar("_T")
_TypeT = TypeVar("_TypeT", bound=type)
_ParamT = ParamSpec("_ParamT")
_DecoratorParamT = ParamSpec("_DecoratorParamT")
_ReturnT = TypeVar("_ReturnT")


class TrackingRegistry:
    def __init__(self) -> None:
        self._lock = RLock()
        self._versions_by_target_id: dict[int, AssetVersion] = {}
        self._assets_by_target_id: dict[int, TrackedAsset] = {}
        self._assets_by_name: dict[str, TrackedAsset] = {}
        self._assets_by_id: dict[str, TrackedAsset] = {}
        self._asset_ids_by_locator: dict[str, str] = {}
        self._latest_versions_by_asset_id: dict[str, AssetVersion] = {}
        self._version_history: list[AssetVersion] = []

    @property
    def assets(self) -> dict[str, TrackedAsset]:
        with self._lock:
            return dict(self._assets_by_name)

    @property
    def definitions(self) -> tuple[TrackedAsset, ...]:
        with self._lock:
            return tuple(self._assets_by_id.values())

    @property
    def versions(self) -> tuple[AssetVersion, ...]:
        with self._lock:
            return tuple(self._version_history)

    def write_assets(
        self,
        directory: Path,
        *,
        asset_ids: Collection[str] | None = None,
        content_path: Path | None = None,
        root_dir: Path | None = None,
    ) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        active_content_path = (
            directory / "content.sqlite3" if content_path is None else content_path
        )
        active_root = directory if root_dir is None else root_dir
        try:
            content_reference = active_content_path.relative_to(active_root).as_posix()
        except ValueError as exc:
            raise ValueError("asset content must be stored inside the registry root") from exc
        with self._lock:
            selected_ids = None if asset_ids is None else set(asset_ids)
            assets = sorted(
                (
                    asset
                    for asset in self._assets_by_id.values()
                    if selected_ids is None or asset.id in selected_ids
                ),
                key=lambda asset: asset.id,
            )
            versions = [self._version_for_asset_id(asset.id) for asset in assets]
        with FileLock(directory / ".write.lock"):
            active_content_path.parent.mkdir(parents=True, exist_ok=True)
            with AssetContentStore(active_content_path) as content_store:
                for asset, version in zip(assets, versions, strict=True):
                    asset_path = directory / f"{_safe_filename(asset.id)}.yaml"
                    existing = load_yaml(asset_path) if asset_path.exists() else None
                    existing_asset = existing.get("asset") if isinstance(existing, dict) else None
                    current_version = (
                        existing_asset.get("current_version")
                        if isinstance(existing_asset, dict)
                        else None
                    )
                    previous_version = current_version if isinstance(current_version, str) else None
                    previous_snapshot = (
                        content_store.content(asset_id=asset.id, version=previous_version)
                        if previous_version is not None
                        else None
                    )
                    snapshot = _asset_version_snapshot(asset)
                    content_store.write_content(
                        asset_id=asset.id,
                        version=version.version,
                        content_hash=version.content_hash,
                        snapshot=snapshot,
                    )
                    changes = _asset_version_changes(previous_snapshot, snapshot)
                    diff = changes.get("diff")
                    if previous_version is not None and isinstance(diff, str):
                        content_store.write_diff(
                            asset_id=asset.id,
                            version=version.version,
                            parent_version=previous_version,
                            diff=diff,
                        )
                    _atomic_dump_yaml(
                        asset_to_yaml_view(
                            asset,
                            version,
                            existing=existing,
                            previous_snapshot=previous_snapshot,
                            content_path=content_reference,
                        ),
                        asset_path,
                        schema_name="asset",
                    )
            index_path = directory / "index.yaml"
            index_view = asset_index_to_yaml_view(assets, versions)
            if index_path.exists():
                existing_index = load_yaml(index_path)
                if isinstance(existing_index, dict) and isinstance(
                    existing_index.get("assets"), dict
                ):
                    index_view["assets"] = {
                        **existing_index["assets"],
                        **index_view["assets"],
                    }
            _atomic_dump_yaml(
                index_view,
                index_path,
                schema_name="asset_index",
            )

    def has_asset(self, asset_id: str) -> bool:
        with self._lock:
            return asset_id in self._assets_by_id

    def _version_for_asset_id(self, asset_id: str) -> AssetVersion:
        try:
            return self._latest_versions_by_asset_id[asset_id]
        except KeyError as exc:
            raise KeyError(f"Asset version is missing for {asset_id}.") from exc

    def asset_by_id(self, asset_id: str) -> TrackedAsset:
        with self._lock:
            try:
                return self._assets_by_id[asset_id]
            except KeyError as exc:
                raise KeyError(f"Unknown Autobench asset: {asset_id}") from exc

    def version_by_asset_id(self, asset_id: str) -> AssetVersion:
        with self._lock:
            return self._version_for_asset_id(asset_id)

    def resolve_locator(self, locator: str) -> TrackedAsset:
        with self._lock:
            try:
                asset_id = self._asset_ids_by_locator[locator]
            except KeyError as exc:
                raise KeyError(f"Unknown Autobench asset locator: {locator}") from exc
            return self._assets_by_id[asset_id]

    def register_candidate(
        self,
        candidate: AssetCandidate,
        *,
        span_id: str | None = None,
    ) -> RegisteredAsset:
        with self._lock:
            resolved = self._resolve_candidate(candidate)
            if resolved is None:
                asset = self._asset_from_candidate(candidate)
                identity = self._candidate_identity(candidate)
                if identity is not None:
                    asset = asset.model_copy(update={"id": identity.id})
                    if isinstance(asset, AssetDefinition) and isinstance(identity, AssetDefinition):
                        asset = asset.model_copy(
                            update={
                                "source_locators": tuple(
                                    dict.fromkeys(
                                        (
                                            *identity.source_locators,
                                            *asset.source_locators,
                                        )
                                    )
                                ),
                                "aliases": tuple(
                                    dict.fromkeys((*identity.aliases, *asset.aliases))
                                ),
                            }
                        )
                asset_content = asset.model_dump(mode="python")
                if isinstance(asset, AssetDefinition):
                    for field_name in (
                        "canonical_content",
                        "source_locators",
                        "aliases",
                        "sensitivity",
                    ):
                        asset_content.pop(field_name)
                content_hash = _hash_serialized(
                    {
                        "asset": asset_content,
                        "content_fingerprint": (
                            candidate.content_fingerprint
                            or _hash_serialized(candidate.canonical_content)
                        ),
                    }
                )
                previous = self._latest_versions_by_asset_id.get(asset.id)
                version = AssetVersion(
                    asset_id=asset.id,
                    version=content_hash[:12],
                    content_hash=content_hash,
                    source_hash=(
                        None
                        if candidate.python_target is None
                        else _source_hash(candidate.python_target)
                    ),
                    source_path=(
                        None
                        if candidate.python_target is None
                        else _source_path(candidate.python_target)
                    ),
                    parent_version=(
                        None
                        if previous is None or previous.version == content_hash[:12]
                        else previous.version
                    ),
                    metadata={
                        "representation": candidate.representation.value,
                        "source_locator": candidate.source_locator,
                    },
                )
                self._register(candidate.python_target, asset, version)
            else:
                asset, version = resolved

            locators = (candidate.source_locator, *candidate.aliases)
            for locator in locators:
                self._asset_ids_by_locator[locator] = asset.id

            definition_asset_id: str | None = None
            definition_version: str | None = None
            if candidate.definition_locator is not None:
                definition = self._asset_for_locator(candidate.definition_locator)
                if definition is not None:
                    definition_asset_id = definition.id
                    definition_version = self._version_for_asset_id(definition.id).version

            return RegisteredAsset(
                asset=asset,
                version=version,
                use=AssetUse(
                    asset_id=asset.id,
                    version=version.version,
                    representation=candidate.representation,
                    source_locator=candidate.source_locator,
                    scope=candidate.scope,
                    span_id=span_id,
                    definition_asset_id=definition_asset_id,
                    definition_version=definition_version,
                    provenance=candidate.provenance,
                    aliases=candidate.aliases,
                ),
            )

    def _resolve_candidate(
        self,
        candidate: AssetCandidate,
    ) -> tuple[TrackedAsset, AssetVersion] | None:
        if candidate.representation is AssetRepresentation.DEFINITION:
            target = candidate.python_target
            if target is not None:
                target_asset = self._assets_by_target_id.get(id(target))
                if target_asset is not None:
                    return target_asset, self._version_for_asset_id(target_asset.id)
                python_locator = _python_locator(target)
                if python_locator is not None:
                    located = self._asset_for_locator(python_locator)
                    if located is not None:
                        return located, self._version_for_asset_id(located.id)
        return None

    def _candidate_identity(self, candidate: AssetCandidate) -> TrackedAsset | None:
        if candidate.explicit_asset_id is not None:
            explicit = self._assets_by_id.get(candidate.explicit_asset_id)
            if explicit is not None:
                return explicit
        for locator in (candidate.source_locator, *candidate.aliases):
            located = self._asset_for_locator(locator)
            if located is not None:
                return located
        return None

    def _asset_for_locator(self, locator: str) -> TrackedAsset | None:
        asset_id = self._asset_ids_by_locator.get(locator)
        return None if asset_id is None else self._assets_by_id[asset_id]

    def _asset_from_candidate(self, candidate: AssetCandidate) -> TrackedAsset:
        asset_id = candidate.explicit_asset_id or candidate.source_locator
        target = candidate.python_target
        metadata = dict(candidate.metadata)
        metadata["discovered"] = True
        if (
            candidate.representation is AssetRepresentation.DEFINITION
            and isinstance(target, type)
            and candidate.kind in {"output_schema", "type"}
        ):
            return _build_type_asset(
                target,
                name=candidate.name,
                semantic_type=candidate.semantic_type,
                metadata=metadata,
            ).model_copy(update={"id": asset_id})
        if (
            candidate.representation is AssetRepresentation.DEFINITION
            and callable(target)
            and candidate.kind == "tool"
        ):
            return _build_tool_asset(
                target,
                name=candidate.name,
                semantic_type=candidate.semantic_type,
                metadata=metadata,
                registry=self,
            ).model_copy(update={"id": asset_id})
        return AssetDefinition(
            id=asset_id,
            kind=candidate.kind,
            name=candidate.name,
            semantic_type=candidate.semantic_type,
            metadata=metadata,
            representation=candidate.representation,
            canonical_content=candidate.canonical_content,
            scope=candidate.scope,
            owner_locator=candidate.owner_locator,
            source_locators=(candidate.source_locator,),
            aliases=candidate.aliases,
            sensitivity=candidate.sensitivity,
        )

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
        version_metadata = dict(prompt_metadata)
        version_metadata.pop("raw")
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
                metadata=version_metadata,
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
        with self._lock:
            if target is not None:
                target_id = id(target)
                self._assets_by_target_id[target_id] = asset
                self._versions_by_target_id[target_id] = version
                python_locator = _python_locator(target)
                if python_locator is not None:
                    self._asset_ids_by_locator[python_locator] = asset.id
            self._assets_by_name[asset.name] = asset
            self._assets_by_id[asset.id] = asset
            self._asset_ids_by_locator[asset.id] = asset.id
            self._latest_versions_by_asset_id[asset.id] = version
            if not any(
                item.asset_id == version.asset_id and item.version == version.version
                for item in self._version_history
            ):
                self._version_history.append(version)


def _python_locator(target: Any) -> str | None:
    try:
        module: str = target.__module__
        qualname: str = target.__qualname__
    except AttributeError:
        return None
    return f"python:{module}.{qualname}"


def _atomic_dump_yaml(value: Any, path: Path, *, schema_name: str) -> None:
    rendered = dump_yaml(value, schema_name=schema_name)
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    try:
        temporary.write_text(rendered, encoding="utf-8")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


__all__ = ("TrackingRegistry", "track")

track = TrackingRegistry()
