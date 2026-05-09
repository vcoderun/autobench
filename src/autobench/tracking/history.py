from __future__ import annotations as _annotations

from difflib import unified_diff
from typing import Any

from .introspection import _normalize_value, _safe_filename
from .models import AssetVersion, FieldAsset, ParamAsset, ToolAsset, TrackedAsset, TypeAsset


def asset_index_to_yaml_view(
    assets: list[TrackedAsset],
    versions: list[AssetVersion],
) -> dict[str, Any]:
    return {
        "record": {
            "type": "asset_index",
            "version": 1,
        },
        "assets": {
            asset.id: {
                "kind": _asset_yaml_kind(asset),
                "name": asset.name,
                **({"semantic": asset.semantic_type} if asset.semantic_type is not None else {}),
                "current_version": version.version,
                "file": f"{_safe_filename(asset.id)}.yaml",
            }
            for asset, version in zip(assets, versions, strict=True)
        },
    }


def asset_to_yaml_view(
    asset: TrackedAsset,
    version: AssetVersion,
    *,
    existing: Any = None,
) -> dict[str, Any]:
    existing_versions = _existing_asset_versions(existing)
    current_snapshot = _asset_version_snapshot(asset)
    previous_snapshot = _existing_asset_current_snapshot(existing)
    if previous_snapshot is None and existing_versions:
        previous_snapshot = _version_entry_snapshot(existing_versions[-1])
    version_payload = _asset_version_payload(
        version,
        current_snapshot,
        previous_snapshot=previous_snapshot,
    )
    versions = [
        entry
        for entry in existing_versions
        if isinstance(entry.get("version"), str) and entry["version"] != version.version
    ]
    versions.append(version_payload)
    return {
        "record": {
            "type": "asset",
            "version": 1,
        },
        "asset": _asset_yaml_view(asset, current_version=version.version),
        "versions": versions,
    }


def _asset_yaml_view(asset: TrackedAsset, *, current_version: str) -> dict[str, Any]:
    view: dict[str, Any] = {
        "id": asset.id,
        "kind": _asset_yaml_kind(asset),
        "name": asset.name,
        "current_version": current_version,
    }
    if asset.semantic_type is not None:
        view["semantic"] = asset.semantic_type
    if asset.metadata:
        metadata = dict(asset.metadata)
        metadata.pop("raw", None)
        if metadata:
            view["metadata"] = metadata
    if isinstance(asset, ToolAsset):
        if asset.qualname is not None:
            view["qualname"] = asset.qualname
        if asset.doc is not None:
            view["doc"] = asset.doc
        view["params"] = {
            param.name: _param_asset_yaml_view(param) for param in asset.param_schema.params
        }
        if asset.return_annotation is not None or asset.return_type_asset_id is not None:
            returns: dict[str, Any] = {}
            if asset.return_annotation is not None:
                returns["type"] = asset.return_annotation
            if asset.return_type_asset_id is not None:
                returns["asset_id"] = asset.return_type_asset_id
            view["returns"] = returns
        return view
    if isinstance(asset, TypeAsset):
        if asset.qualname is not None:
            view["qualname"] = asset.qualname
        if asset.doc is not None:
            view["doc"] = asset.doc
        view["fields"] = {
            field_asset.name: _field_asset_yaml_view(field_asset)
            for field_asset in asset.field_assets
        }
        return view
    if asset.kind == "prompt":
        prompt_text = asset.metadata.get("raw")
        if isinstance(prompt_text, str):
            view["raw"] = prompt_text
        return view
    return view


def _asset_yaml_kind(asset: TrackedAsset) -> str:
    if isinstance(asset, TypeAsset):
        return asset.type_kind
    return asset.kind


def _param_asset_yaml_view(param: ParamAsset) -> dict[str, Any]:
    view: dict[str, Any] = {"required": param.required}
    if param.annotation is not None:
        view["type"] = param.annotation
    if param.default is not None:
        view["default"] = param.default
    if param.kind != "positional":
        view["kind"] = param.kind
    if param.literal_choices:
        view["choices"] = list(param.literal_choices)
    return view


def _field_asset_yaml_view(field_asset: FieldAsset) -> dict[str, Any]:
    view: dict[str, Any] = {"required": field_asset.required}
    if field_asset.annotation is not None:
        view["type"] = field_asset.annotation
    if field_asset.default is not None:
        view["default"] = field_asset.default
    if field_asset.default_factory is not None:
        view["default_factory"] = field_asset.default_factory
    if field_asset.description is not None:
        view["description"] = field_asset.description
    if field_asset.examples:
        view["examples"] = list(field_asset.examples)
    if field_asset.alias is not None:
        view["alias"] = field_asset.alias
    if field_asset.constraints:
        view["constraints"] = field_asset.constraints
    if field_asset.literal_choices:
        view["choices"] = list(field_asset.literal_choices)
    if field_asset.metadata:
        view["metadata"] = field_asset.metadata
    if field_asset.init is not None:
        view["init"] = field_asset.init
    if field_asset.kw_only is not None:
        view["kw_only"] = field_asset.kw_only
    if field_asset.compare is not None:
        view["compare"] = field_asset.compare
    if field_asset.repr is not None:
        view["repr"] = field_asset.repr
    return view


def _asset_version_snapshot(asset: TrackedAsset) -> dict[str, Any]:
    if isinstance(asset, ToolAsset):
        tool_snapshot: dict[str, Any] = {
            "kind": "tool",
            "name": asset.name,
            "params": {
                param.name: _param_asset_yaml_view(param) for param in asset.param_schema.params
            },
        }
        if asset.qualname is not None:
            tool_snapshot["qualname"] = asset.qualname
        if asset.doc is not None:
            tool_snapshot["doc"] = asset.doc
        if asset.return_annotation is not None or asset.return_type_asset_id is not None:
            returns: dict[str, Any] = {}
            if asset.return_annotation is not None:
                returns["type"] = asset.return_annotation
            if asset.return_type_asset_id is not None:
                returns["asset_id"] = asset.return_type_asset_id
            tool_snapshot["returns"] = returns
        if asset.metadata:
            tool_snapshot["metadata"] = asset.metadata
        return tool_snapshot
    if isinstance(asset, TypeAsset):
        type_snapshot: dict[str, Any] = {
            "kind": asset.type_kind,
            "name": asset.name,
            "fields": {
                field_asset.name: _field_asset_yaml_view(field_asset)
                for field_asset in asset.field_assets
            },
        }
        if asset.qualname is not None:
            type_snapshot["qualname"] = asset.qualname
        if asset.doc is not None:
            type_snapshot["doc"] = asset.doc
        if asset.metadata:
            type_snapshot["metadata"] = asset.metadata
        return type_snapshot
    if asset.kind == "prompt":
        prompt_snapshot: dict[str, Any] = {"kind": "prompt", "name": asset.name}
        prompt_text = asset.metadata.get("raw")
        if isinstance(prompt_text, str):
            prompt_snapshot["raw"] = prompt_text
        if asset.metadata:
            metadata = dict(asset.metadata)
            metadata.pop("raw", None)
            if metadata:
                prompt_snapshot["metadata"] = metadata
        return prompt_snapshot
    generic_snapshot: dict[str, Any] = {"kind": asset.kind, "name": asset.name}
    if asset.metadata:
        generic_snapshot["metadata"] = asset.metadata
    return generic_snapshot


def _asset_version_payload(
    version: AssetVersion,
    snapshot: dict[str, Any],
    *,
    previous_snapshot: dict[str, Any] | None,
) -> dict[str, Any]:
    changes = _asset_version_changes(previous_snapshot, snapshot)
    payload: dict[str, Any] = {
        "version": version.version,
        "state": snapshot,
        "changes": changes,
        "hashes": {"content": version.content_hash},
    }
    if version.parent_version is not None:
        payload["parent"] = version.parent_version
    if version.source_hash is not None:
        payload["hashes"]["source"] = version.source_hash
    source: dict[str, Any] = {}
    if version.source_path is not None:
        source["path"] = version.source_path
    if version.git_commit is not None:
        source["git_commit"] = version.git_commit
    if source:
        payload["source"] = source
    if version.metadata:
        payload["metadata"] = version.metadata
    return payload


def _asset_version_changes(
    previous_snapshot: dict[str, Any] | None,
    snapshot: dict[str, Any],
) -> dict[str, Any]:
    if previous_snapshot is None:
        return {"fields": ["initial"], "diff": None}
    diff_lines = _asset_version_diff(previous_snapshot, snapshot)
    return {
        "fields": _changed_field_paths(previous_snapshot, snapshot),
        "diff": "\n".join(diff_lines) if diff_lines else None,
    }


def _changed_field_paths(
    previous: dict[str, Any],
    current: dict[str, Any],
) -> list[str]:
    paths = _collect_changed_paths(previous, current)
    return sorted(dict.fromkeys(path or "value" for path in paths))


def _collect_changed_paths(
    previous: Any,
    current: Any,
    *,
    prefix: str = "",
) -> list[str]:
    if isinstance(previous, dict) and isinstance(current, dict):
        dict_paths: list[str] = []
        for key in sorted(set(previous) | set(current)):
            path = f"{prefix}.{key}" if prefix else str(key)
            if key not in previous or key not in current:
                dict_paths.append(path)
                continue
            dict_paths.extend(_collect_changed_paths(previous[key], current[key], prefix=path))
        return dict_paths
    if isinstance(previous, list) and isinstance(current, list):
        if previous == current:
            return []
        if len(previous) != len(current):
            return [prefix]
        list_paths: list[str] = []
        for index, (before_item, after_item) in enumerate(zip(previous, current, strict=True)):
            item_prefix = f"{prefix}[{index}]"
            list_paths.extend(_collect_changed_paths(before_item, after_item, prefix=item_prefix))
        return list_paths or [prefix]
    if previous == current:
        return []
    return [prefix]


def _existing_asset_versions(existing: Any) -> list[dict[str, Any]]:
    if not isinstance(existing, dict):
        return []
    raw_versions = existing.get("versions")
    if not isinstance(raw_versions, list):
        return []
    return [dict(version) for version in raw_versions if isinstance(version, dict)]


def _existing_asset_current_snapshot(existing: Any) -> dict[str, Any] | None:
    if not isinstance(existing, dict):
        return None
    raw_asset = existing.get("asset")
    if not isinstance(raw_asset, dict):
        return None
    current_asset = dict(raw_asset)
    current_asset.pop("id", None)
    current_asset.pop("current_version", None)
    current_asset.pop("semantic", None)
    current_asset.pop("semantic_type", None)
    return current_asset


def _asset_version_diff(
    previous_snapshot: dict[str, Any],
    current_snapshot: dict[str, Any],
) -> list[str]:
    before = repr(_normalize_value(previous_snapshot)).splitlines()
    after = repr(_normalize_value(current_snapshot)).splitlines()
    return list(
        unified_diff(
            before,
            after,
            fromfile="previous",
            tofile="current",
            lineterm="",
        )
    )


def _version_entry_snapshot(entry: dict[str, Any]) -> dict[str, Any] | None:
    state = entry.get("state")
    if isinstance(state, dict):
        return state
    snapshot = entry.get("snapshot")
    if isinstance(snapshot, dict):
        return snapshot
    return None


__all__ = (
    "_asset_version_changes",
    "_asset_version_payload",
    "_asset_version_snapshot",
    "_asset_yaml_view",
    "_existing_asset_current_snapshot",
    "_existing_asset_versions",
    "_version_entry_snapshot",
    "asset_index_to_yaml_view",
    "asset_to_yaml_view",
)
