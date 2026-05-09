from __future__ import annotations as _annotations

import hashlib
import json
from collections.abc import Mapping
from typing import Any

from pydantic import BaseModel, Field

from autobench.records.artifacts import ArtifactRef


class Case(BaseModel):
    id: str = Field(min_length=1)
    input: Any = None
    expected: Any = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    tags: list[str] = Field(default_factory=list)
    attachments: list[ArtifactRef] = Field(default_factory=list)


class CaseDefaults(BaseModel):
    input: Any = None
    expected: Any = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    tags: list[str] = Field(default_factory=list)
    attachments: list[ArtifactRef] = Field(default_factory=list)


class DatasetSpec(BaseModel):
    id: str | None = None
    source: str | None = None
    version: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    cases: list[Case] = Field(default_factory=list)
    case_defaults: CaseDefaults = Field(default_factory=CaseDefaults)


def dataset_to_yaml_view(dataset: DatasetSpec) -> dict[str, Any]:
    dataset_view: dict[str, Any] = {
        "id": dataset.id or "inline",
        "cases": [case_to_yaml_view(case) for case in dataset.cases],
    }
    view: dict[str, Any] = {
        "record": {
            "type": "dataset",
            "version": 1,
        },
        "dataset": dataset_view,
    }
    if dataset.source is not None:
        dataset_view["source"] = dataset.source
    if dataset.version is not None:
        dataset_view["version"] = dataset.version
    if dataset.metadata:
        dataset_view["metadata"] = dataset.metadata
    defaults_view = _case_defaults_yaml_view(dataset.case_defaults)
    if defaults_view:
        dataset_view["defaults"] = defaults_view
    return view


def merge_case_defaults(case: Case, defaults: CaseDefaults) -> Case:
    merged_input = _merge_value(defaults.input, case.input)
    merged_expected = _merge_value(defaults.expected, case.expected)
    merged_metadata = _merge_mapping(defaults.metadata, case.metadata)
    merged_tags = _merge_tags(defaults.tags, case.tags)
    merged_attachments = [*defaults.attachments, *case.attachments]
    return case.model_copy(
        update={
            "input": merged_input,
            "expected": merged_expected,
            "metadata": merged_metadata,
            "tags": merged_tags,
            "attachments": merged_attachments,
        }
    )


def dataset_content_hash(dataset: DatasetSpec) -> str:
    payload = dataset.model_dump(
        mode="json",
        exclude={"source"},
        exclude_none=True,
    )
    rendered = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(rendered.encode("utf-8")).hexdigest()


def _merge_value(base: Any, override: Any) -> Any:
    if override is None:
        return base
    if base is None:
        return override
    if isinstance(base, Mapping) and isinstance(override, Mapping):
        return _merge_mapping(dict(base), dict(override))
    return override


def _merge_mapping(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if key in merged and isinstance(merged[key], Mapping) and isinstance(value, Mapping):
            merged[key] = _merge_mapping(dict(merged[key]), dict(value))
        else:
            merged[key] = value
    return merged


def _merge_tags(base: list[str], override: list[str]) -> list[str]:
    seen: set[str] = set()
    merged: list[str] = []
    for tag in [*base, *override]:
        if tag not in seen:
            seen.add(tag)
            merged.append(tag)
    return merged


def case_to_yaml_view(case: Case) -> dict[str, Any]:
    view: dict[str, Any] = {"id": case.id}
    if case.input is not None:
        view["input"] = case.input
    if case.expected is not None:
        view["expected"] = case.expected
    if case.metadata:
        view["metadata"] = case.metadata
    if case.tags:
        view["tags"] = list(case.tags)
    attachments_view = _attachments_yaml_view(case.attachments)
    if attachments_view:
        view["attachments"] = attachments_view
    return view


def _case_defaults_yaml_view(defaults: CaseDefaults) -> dict[str, Any]:
    view: dict[str, Any] = {}
    if defaults.input is not None:
        view["input"] = defaults.input
    if defaults.expected is not None:
        view["expected"] = defaults.expected
    if defaults.metadata:
        view["metadata"] = defaults.metadata
    if defaults.tags:
        view["tags"] = list(defaults.tags)
    attachments_view = _attachments_yaml_view(defaults.attachments)
    if attachments_view:
        view["attachments"] = attachments_view
    return view


def _attachments_yaml_view(attachments: list[ArtifactRef]) -> list[dict[str, Any]]:
    return [_artifact_ref_yaml_view(attachment) for attachment in attachments]


def _artifact_ref_yaml_view(artifact: ArtifactRef) -> dict[str, Any]:
    view: dict[str, Any] = {
        "id": artifact.id,
        "name": artifact.name,
    }
    if artifact.media_type is not None:
        view["media_type"] = artifact.media_type
    if artifact.value is not None:
        view["value"] = artifact.value
    if artifact.span_id is not None:
        view["span_id"] = artifact.span_id
    if artifact.tags:
        view["tags"] = artifact.tags
    return view


__all__ = (
    "Case",
    "CaseDefaults",
    "DatasetSpec",
    "case_to_yaml_view",
    "dataset_content_hash",
    "dataset_to_yaml_view",
    "merge_case_defaults",
)
