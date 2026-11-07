from __future__ import annotations as _annotations

from pathlib import Path
from typing import Any

import pytest

from autobench.io import load_yaml
from autobench.tracking import AssetCandidate, AssetProvenance, TrackingRegistry
from autobench.tracking.discovery import RegisteredAsset, canonical_asset_hash
from autobench.tracking.store import load_asset_content, load_asset_diff

_LOCATOR = "kedi:form:planner"


def test_every_version_a_run_used_is_persisted_with_parent_diffs(tmp_path: Path) -> None:
    registry = TrackingRegistry()
    forms = [
        registry.register_candidate(_form(text)) for text in ("Plan.", "Plan then act.", "Act.")
    ]

    registry.write_assets(tmp_path, versions=[form.version for form in forms])

    content = tmp_path / "content.sqlite3"
    for form, text in zip(forms, ("Plan.", "Plan then act.", "Act."), strict=True):
        assert (
            load_asset_content(content, asset_id=form.asset.id, version=form.version.version)[
                "content"
            ]
            == text
        )
    for parent, child in zip(forms, forms[1:], strict=False):
        assert child.version.parent_version == parent.version.version
        assert load_asset_diff(
            content,
            asset_id=child.asset.id,
            version=child.version.version,
            parent_version=parent.version.version,
        )
    history = _history(tmp_path, forms[0])
    assert [entry["version"] for entry in history["versions"]] == [
        form.version.version for form in forms
    ]
    assert history["asset"]["current_version"] == forms[-1].version.version


def test_declared_parent_version_drives_lineage_and_diffs(tmp_path: Path) -> None:
    registry = TrackingRegistry()
    first = registry.register_candidate(_form("Plan."))
    second = registry.register_candidate(_form("Plan carefully."))
    revised = registry.register_candidate(
        _form("Plan, then verify.", parent_version=first.version.version)
    )

    assert revised.version.parent_version == first.version.version
    registry.write_assets(tmp_path, versions=[first.version, second.version, revised.version])

    assert "Plan, then verify." in load_asset_diff(
        tmp_path / "content.sqlite3",
        asset_id=revised.asset.id,
        version=revised.version.version,
        parent_version=first.version.version,
    )
    entries = {entry["version"]: entry for entry in _history(tmp_path, first)["versions"]}
    assert entries[revised.version.version]["parent"] == first.version.version


def test_a_version_cannot_be_its_own_parent() -> None:
    registry = TrackingRegistry()
    first = registry.register_candidate(_form("Plan."))
    registry.register_candidate(_form("Act."))

    with pytest.raises(ValueError, match="cannot be its own parent"):
        registry.register_candidate(_form("Plan.", parent_version=first.version.version))


def test_older_versions_recorded_later_keep_the_newest_current(tmp_path: Path) -> None:
    registry = TrackingRegistry()
    first = registry.register_candidate(_form("Plan."))
    second = registry.register_candidate(_form("Act."))

    registry.write_assets(tmp_path, versions=[second.version])
    registry.write_assets(tmp_path, versions=[first.version])
    registry.write_assets(tmp_path, versions=[first.version])

    history = _history(tmp_path, first)
    assert history["asset"]["current_version"] == second.version.version
    assert sorted(entry["version"] for entry in history["versions"]) == sorted(
        (first.version.version, second.version.version)
    )
    index = load_yaml(tmp_path / "index.yaml")
    assert index["assets"][first.asset.id]["current_version"] == second.version.version
    assert (
        load_asset_content(
            tmp_path / "content.sqlite3",
            asset_id=first.asset.id,
            version=first.version.version,
        )["content"]
        == "Plan."
    )


def test_parent_fingerprint_links_a_version_to_a_parent_from_another_record() -> None:
    root = TrackingRegistry().register_candidate(_form("Plan.", root=True))
    # A later generation's record never saw the parent; its fingerprint is enough.
    child = TrackingRegistry().register_candidate(
        _form("Plan, then verify.", parent_fingerprint=canonical_asset_hash("Plan."))
    )

    assert root.version.parent_version is None
    assert child.version.parent_version == root.version.version


def test_root_declares_no_parent_instead_of_the_latest_version() -> None:
    registry = TrackingRegistry()
    first = registry.register_candidate(_form("Plan."))
    unrelated = registry.register_candidate(_form("Act.", root=True))
    chronological = registry.register_candidate(_form("Act again."))

    assert unrelated.version.parent_version is None
    assert chronological.version.parent_version == unrelated.version.version
    assert first.version.parent_version is None


def test_a_version_cannot_be_its_own_parent_by_fingerprint() -> None:
    registry = TrackingRegistry()
    with pytest.raises(ValueError, match="cannot be its own parent"):
        registry.register_candidate(
            _form("Plan.", parent_fingerprint=canonical_asset_hash("Plan."))
        )


@pytest.mark.parametrize(
    "lineage",
    [
        {"parent_version": "abc", "root": True},
        {"parent_version": "abc", "parent_fingerprint": "0" * 64},
        {"parent_fingerprint": "0" * 64, "root": True},
    ],
)
def test_a_candidate_declares_at_most_one_lineage(lineage: dict[str, Any]) -> None:
    with pytest.raises(ValueError, match="Declare at most one of"):
        _form("Plan.", **lineage)


def _form(
    text: str,
    *,
    parent_version: str | None = None,
    parent_fingerprint: str | None = None,
    root: bool = False,
) -> AssetCandidate:
    return AssetCandidate(
        kind="form",
        local_id="planner",
        name="planner",
        source_locator=_LOCATOR,
        canonical_content=text,
        content_fingerprint=canonical_asset_hash(text),
        parent_version=parent_version,
        parent_fingerprint=parent_fingerprint,
        root=root,
        provenance=AssetProvenance(system="kedi", key="planner"),
    )


def _history(directory: Path, registered: RegisteredAsset) -> dict[str, Any]:
    index = load_yaml(directory / "index.yaml")
    history = load_yaml(directory / index["assets"][registered.asset.id]["file"])
    assert isinstance(history, dict)
    return history
