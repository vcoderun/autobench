from __future__ import annotations as _annotations

from pathlib import Path

import pytest

from autobench import __version__
from autobench.errors import SpecLoadError
from autobench.io import (
    dump_yaml,
    load_yaml,
    loose_yaml_schema,
    resolve_file_ref,
    schema_path,
    yaml_schema,
)


def test_yaml_roundtrip(tmp_path: Path) -> None:
    path = tmp_path / "roundtrip.yaml"
    data = {"benchmark": {"id": "demo", "description": "hello"}}

    dump_yaml(data, path)

    assert load_yaml(path) == data
    assert dump_yaml(data).startswith("benchmark:")


def test_dump_yaml_can_write_schema_header_and_cached_schema(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("AUTOBENCH_HOME", str(tmp_path / "home"))
    path = tmp_path / "benchmark.yaml"
    data = {"items": ["one", "two"]}

    rendered = dump_yaml(
        data,
        path,
        schema_name="benchmark",
        schema={"$schema": "https://json-schema.org/draft/2020-12/schema", "type": "object"},
    )

    expected_schema_path = tmp_path / "home" / __version__ / "schemas" / "benchmark_schema.json"
    assert rendered.startswith(f"# yaml-language-server: $schema={expected_schema_path}\n")
    assert schema_path("benchmark") == expected_schema_path
    assert expected_schema_path.exists()
    assert "benchmark" in yaml_schema("benchmark")["properties"]
    assert "payload" in yaml_schema("artifact_payload")["properties"]
    staging_properties = yaml_schema("staging")["properties"]
    assert set(staging_properties) == {
        "staging",
        "experiment",
        "plan",
        "runs",
        "post_processing",
        "environment",
        "semantic_registry",
        "report",
        "benchmark_spec",
        "spec_hash",
        "source_files",
    }
    assert set(yaml_schema("staging_manifest")["properties"]) == {
        "staging",
        "experiment",
        "runs",
        "checkpoints",
        "payloads",
    }
    assert set(yaml_schema("checkpoint")["properties"]) == {
        "checkpoint",
        "run",
        "evidence",
    }
    assert loose_yaml_schema("custom")["title"] == "Autobench Custom YAML"
    assert load_yaml(path) == data
    assert "items:\n  - one\n  - two\n" in rendered


def test_invalid_yaml_reports_path_and_location(tmp_path: Path) -> None:
    path = tmp_path / "broken.yaml"
    path.write_text("benchmark:\n  id: [oops\n", encoding="utf-8")

    with pytest.raises(SpecLoadError) as exc_info:
        load_yaml(path)

    exc = exc_info.value
    assert exc.path == path
    assert exc.line is not None
    assert exc.column is not None


def test_file_ref_resolves_relative_to_spec_file(tmp_path: Path) -> None:
    spec_dir = tmp_path / "specs"
    data_dir = spec_dir / "datasets"
    data_dir.mkdir(parents=True)
    spec_path = spec_dir / "autobench.yaml"
    spec_path.write_text("benchmark:\n  id: demo\n", encoding="utf-8")

    resolved = resolve_file_ref("file://datasets/cases.yaml", base_path=spec_path)

    assert resolved == (data_dir / "cases.yaml").resolve()


def test_plain_file_ref_resolves_relative_to_spec_file(tmp_path: Path) -> None:
    spec_dir = tmp_path / "specs"
    data_dir = spec_dir / "datasets"
    data_dir.mkdir(parents=True)
    spec_path = spec_dir / "autobench.yaml"
    spec_path.write_text("benchmark:\n  id: demo\n", encoding="utf-8")

    resolved = resolve_file_ref("datasets/cases.yaml", base_path=spec_path)

    assert resolved == (data_dir / "cases.yaml").resolve()


def test_remote_refs_are_rejected(tmp_path: Path) -> None:
    spec_path = tmp_path / "autobench.yaml"
    spec_path.write_text("benchmark:\n  id: demo\n", encoding="utf-8")

    with pytest.raises(SpecLoadError):
        resolve_file_ref("https://example.com/spec.yaml", base_path=spec_path)
