from __future__ import annotations as _annotations

from collections.abc import Collection
from io import StringIO
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError
from rich.console import Console

import autobench.records.files as record_files
import autobench.records.recording as recording
import autobench.runtime.pipeline as pipeline
from autobench import (
    AssetProvenance,
    AssetSensitivity,
    BenchmarkPlan,
    Case,
    EndReason,
    EnvironmentMetadata,
    EvaluationStatus,
    ExperimentResult,
    ExperimentStatus,
    ExperimentTermination,
    RecordingError,
    ReplayError,
    RunContext,
    RunRecord,
    RunResult,
    RunStatus,
    TaskResult,
    TaskStatus,
    TrackingRegistry,
    Variant,
    load_experiment_record,
    load_run_record,
    record_experiment,
    replay_experiment,
)
from autobench.io import dump_yaml, load_yaml, yaml_schema
from autobench.records.files import (
    LogicalRecordTarget,
    ManifestEntry,
    RecordFileKind,
    RecordManifest,
    atomic_write_text,
    build_manifest,
    normalize_logical_path,
    sync_directory,
    validate_logical_targets,
    validate_manifest,
)
from autobench.records.views import (
    experiment_record_payload_from_yaml_view,
    manifest_payload_from_yaml_view,
    run_record_to_yaml_view,
)
from autobench.reports.rich import render_experiment_result
from autobench.tracking import AssetCandidate


def test_recording_publishes_complete_directory_with_valid_manifest(tmp_path: Path) -> None:
    result = _experiment_result()
    destination = tmp_path / "record"

    record = record_experiment(result, destination)

    assert record.manifest_path == "manifest.yaml"
    assert sorted(path.name for path in destination.iterdir()) == [
        "artifacts",
        "cases",
        "experiment.yaml",
        "manifest.yaml",
        "summary.yaml",
    ]
    raw_manifest = load_yaml(destination / "manifest.yaml")
    manifest = RecordManifest.model_validate(manifest_payload_from_yaml_view(raw_manifest))
    validate_manifest(destination, manifest)
    assert {entry.kind for entry in manifest.files} >= {
        RecordFileKind.EXPERIMENT,
        RecordFileKind.SUMMARY,
        RecordFileKind.RUN,
        RecordFileKind.ARTIFACT,
    }
    assert (
        (destination / "manifest.yaml")
        .read_text(encoding="utf-8")
        .startswith("# yaml-language-server: $schema=")
    )
    assert yaml_schema("manifest")["properties"]["record"]["properties"]["type"] == {
        "const": "manifest"
    }


def test_recording_supports_empty_destination_and_rejects_nonempty_destination(
    tmp_path: Path,
) -> None:
    result = _experiment_result()
    empty = tmp_path / "empty"
    empty.mkdir()

    record_experiment(result, empty)
    assert (empty / "summary.yaml").is_file()

    with pytest.raises(RecordingError, match="Record target already exists"):
        record_experiment(result, empty)

    target = tmp_path / "symlink-target"
    target.mkdir()
    symlink = tmp_path / "symlink-record"
    symlink.symlink_to(target, target_is_directory=True)
    with pytest.raises(RecordingError, match="Record target already exists"):
        record_experiment(result, symlink)
    assert not list(target.iterdir())


def test_recording_rejects_normalized_run_and_artifact_collisions_before_writing(
    tmp_path: Path,
) -> None:
    first = _run_result(case_id="a-b", run_id="run-1", artifact_id="a-b")
    second = _run_result(case_id="a_b", run_id="run_2", artifact_id="a_b")
    result = _experiment_result(runs=[first, second])
    destination = tmp_path / "collision"

    with pytest.raises(RecordingError, match="Normalized record path collision"):
        record_experiment(result, destination)

    assert not destination.exists()
    assert not list(tmp_path.glob(".collision.finalizing-*"))


@pytest.mark.parametrize(
    "failed_name",
    [
        "payload.yaml",
        "payload.meta.yaml",
        "trace.yaml",
        "run.yaml",
        "experiment.yaml",
        "summary.yaml",
        "manifest.yaml",
    ],
)
def test_recording_failures_never_publish_partial_directory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failed_name: str,
) -> None:
    result = _experiment_result()
    destination = tmp_path / f"failed-{failed_name.replace('.', '-')}"
    original_write_yaml = recording._write_yaml

    def fail_selected_write(
        value: Any,
        path: Path,
        *,
        schema_name: str,
        durability: record_files.RecordDurability,
        schema: dict[str, Any] | None = None,
    ) -> None:
        artifact_failure = (
            failed_name == "payload.yaml" and schema_name == "artifact_payload"
        ) or (failed_name == "payload.meta.yaml" and schema_name == "artifact")
        if path.name == failed_name or artifact_failure:
            raise OSError(f"injected {failed_name} failure")
        original_write_yaml(
            value,
            path,
            schema_name=schema_name,
            durability=durability,
            schema=schema,
        )

    monkeypatch.setattr(recording, "_write_yaml", fail_selected_write)

    with pytest.raises(OSError, match="injected"):
        record_experiment(result, destination, trace_inline_limit_bytes=1)

    assert not destination.exists()
    assert not list(tmp_path.glob(f".{destination.name}.finalizing-*"))


def test_asset_registry_failure_never_publishes_partial_directory(tmp_path: Path) -> None:
    class FailingRegistry(TrackingRegistry):
        def write_assets(
            self,
            directory: Path,
            *,
            asset_ids: Collection[str] | None = None,
            content_path: Path | None = None,
            root_dir: Path | None = None,
        ) -> None:
            raise OSError(f"asset write failed: {directory}:{asset_ids}:{content_path}:{root_dir}")

    registry = FailingRegistry()
    registered = registry.register_candidate(
        AssetCandidate(
            kind="prompt",
            local_id="instructions",
            name="instructions",
            source_locator="test:prompt:instructions",
            canonical_content="Use evidence.",
            provenance=AssetProvenance(system="test", key="instructions"),
            sensitivity=AssetSensitivity.PUBLIC,
        ),
        span_id="span",
    )
    result = _experiment_result()
    result.runs[0].asset_versions = [registered.version]
    destination = tmp_path / "asset-failure"

    with pytest.raises(OSError, match="asset write failed"):
        record_experiment(result, destination, asset_registry=registry)

    assert not destination.exists()
    assert not list(tmp_path.glob(".asset-failure.finalizing-*"))


def test_atomic_file_replace_cleans_temporary_file_and_preserves_previous_content(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    destination = tmp_path / "state.yaml"
    destination.write_text("old", encoding="utf-8")

    def fail_replace(source: Path, target: Path) -> None:
        assert source.parent == target.parent
        raise OSError("replace failed")

    monkeypatch.setattr(record_files.os, "replace", fail_replace)

    with pytest.raises(OSError, match="replace failed"):
        atomic_write_text(destination, "new")

    assert destination.read_text(encoding="utf-8") == "old"
    assert not list(tmp_path.glob(".state.yaml.*.tmp"))


def test_synced_writes_and_directory_publication_fsync_files_and_directories(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    synced_descriptors: list[int] = []
    monkeypatch.setattr(record_files.os, "fsync", synced_descriptors.append)

    destination = tmp_path / "synced.txt"
    atomic_write_text(destination, "durable", durability="synced")
    record_experiment(_experiment_result(), tmp_path / "record", durability="synced")

    assert destination.read_text(encoding="utf-8") == "durable"
    assert len(synced_descriptors) >= 4


def test_synced_mode_reports_unsupported_and_failed_directory_sync(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(record_files.os, "name", "nt")
    with pytest.raises(RecordingError, match="POSIX directory fsync"):
        sync_directory(tmp_path)

    monkeypatch.setattr(record_files.os, "name", "posix")
    real_open = record_files.os.open

    def fail_open(path: Path, flags: int) -> int:
        raise OSError(f"cannot open {path}:{flags}")

    monkeypatch.setattr(record_files.os, "open", fail_open)
    with pytest.raises(RecordingError, match="Could not open directory"):
        sync_directory(tmp_path)

    monkeypatch.setattr(record_files.os, "open", real_open)

    def fail_fsync(descriptor: int) -> None:
        raise OSError(f"cannot sync {descriptor}")

    monkeypatch.setattr(record_files.os, "fsync", fail_fsync)
    with pytest.raises(RecordingError, match="Could not fsync directory"):
        sync_directory(tmp_path)


def test_sync_tree_and_publish_reject_unsupported_or_conflicting_targets(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    staging = tmp_path / "staging"
    staging.mkdir()
    destination = tmp_path / "destination"
    destination.write_text("occupied", encoding="utf-8")
    with pytest.raises(RecordingError, match="already exists"):
        record_files.publish_record_directory(staging, destination)

    destination.unlink()
    symlink_target = tmp_path / "existing-directory"
    symlink_target.mkdir()
    destination.symlink_to(symlink_target, target_is_directory=True)
    with pytest.raises(RecordingError, match="already exists"):
        record_files.publish_record_directory(staging, destination)

    monkeypatch.setattr(record_files.os, "name", "nt")
    with pytest.raises(RecordingError, match="POSIX directory fsync"):
        record_files.sync_tree(staging)


def test_logical_paths_and_manifests_reject_invalid_or_changed_files(tmp_path: Path) -> None:
    with pytest.raises(RecordingError, match="normalized relative path"):
        normalize_logical_path("../outside.yaml")
    with pytest.raises(RecordingError, match="collision"):
        validate_logical_targets(
            (
                LogicalRecordTarget(path="same.yaml", kind=RecordFileKind.RUN, identity="run-1"),
                LogicalRecordTarget(path="same.yaml", kind=RecordFileKind.RUN, identity="run-2"),
            )
        )

    root = tmp_path / "manifest"
    root.mkdir()
    payload = root / "payload.txt"
    payload.write_text("first", encoding="utf-8")
    manifest = build_manifest(root, experiment_id="exp", targets={})
    assert build_manifest(root, experiment_id="exp", targets={}) == manifest
    validate_manifest(root, manifest)
    payload.write_text("other", encoding="utf-8")
    with pytest.raises(RecordingError, match="hash mismatch"):
        validate_manifest(root, manifest)
    (root / "extra.txt").write_text("extra", encoding="utf-8")
    with pytest.raises(RecordingError, match="file set mismatch"):
        validate_manifest(root, manifest)

    payload.unlink()
    (root / "extra.txt").unlink()
    payload.write_text("longer", encoding="utf-8")
    with pytest.raises(RecordingError, match="byte count mismatch"):
        validate_manifest(root, manifest)

    with pytest.raises(ValidationError, match="normalized relative path"):
        ManifestEntry(
            path="../outside.txt",
            sha256="0" * 64,
            byte_count=0,
            kind=RecordFileKind.OTHER,
            identity="outside",
        )

    outside = tmp_path / "outside.txt"
    outside.write_text("outside", encoding="utf-8")
    linked_root = tmp_path / "linked-manifest"
    linked_root.mkdir()
    link = linked_root / "linked.txt"
    link.symlink_to(outside)
    linked_manifest = build_manifest(linked_root, experiment_id="exp", targets={})
    with pytest.raises(RecordingError, match="escapes experiment directory"):
        validate_manifest(linked_root, linked_manifest)


def test_manifest_yaml_parser_rejects_non_manifest_and_invalid_files() -> None:
    raw = {"record": {"type": "run"}}
    assert manifest_payload_from_yaml_view(raw) == raw
    with pytest.raises(RecordingError, match="files must be a list"):
        manifest_payload_from_yaml_view(
            {
                "record": {"type": "manifest", "version": 1},
                "experiment": {"id": "exp"},
                "files": {},
            }
        )


def test_experiment_loading_rejects_wrong_or_non_mapping_manifest(
    tmp_path: Path,
) -> None:
    destination = tmp_path / "record"
    record_experiment(_experiment_result(), destination)
    manifest_path = destination / "manifest.yaml"
    raw = load_yaml(manifest_path)
    raw["experiment"]["id"] = "different"
    dump_yaml(raw, manifest_path, schema_name="manifest")
    with pytest.raises(ReplayError, match="identity does not match"):
        load_experiment_record(destination)

    dump_yaml([], manifest_path, schema_name="manifest")
    with pytest.raises(ReplayError, match="Invalid experiment manifest"):
        load_experiment_record(destination)

    raw_experiment = load_yaml(destination / "experiment.yaml")
    raw_experiment["manifest"] = "../../outside.yaml"
    dump_yaml(raw_experiment, destination / "experiment.yaml", schema_name="experiment")
    with pytest.raises(ReplayError, match="must stay inside"):
        load_experiment_record(destination)


def test_cancelled_partial_records_round_trip_and_legacy_records_default_complete(
    tmp_path: Path,
) -> None:
    run = _run_result(case_id="cancelled", run_id="run_cancelled")
    run.task_result.status = TaskStatus.CANCELLED
    run.task_result.partial = True
    run.task_result.end_reason = EndReason.CANCELLED
    run.status = RunStatus.CANCELLED
    run.evaluation_status = EvaluationStatus.NOT_EVALUATED
    run.partial = True
    run.end_reason = EndReason.CANCELLED
    result = _experiment_result(runs=[run])
    result.termination = ExperimentTermination(
        status=ExperimentStatus.CANCELLED,
        partial=True,
        cross_run_derivation_complete=False,
        policies_complete=False,
        planned_run_ids=("run_cancelled", "run_missing"),
        recorded_run_ids=("run_cancelled",),
        missing_run_ids=("run_missing",),
    )
    destination = tmp_path / "cancelled"

    record_experiment(result, destination)
    record = load_experiment_record(destination)
    run_record = load_run_record(destination / record.run_paths[0])
    replayed = replay_experiment(destination)

    assert record.termination == result.termination
    assert record.cancelled_count == 1
    assert run_record.status is RunStatus.CANCELLED
    assert run_record.partial is True
    assert run_record.end_reason is EndReason.CANCELLED
    assert replayed.termination.status is ExperimentStatus.CANCELLED
    assert replayed.runs[0].task_result.status is TaskStatus.CANCELLED
    assert load_yaml(destination / "summary.yaml")["summary"] == {
        "experiment": "exp",
        "benchmark": "benchmark",
        "status": "cancelled",
        "partial": True,
    }

    legacy = run_record.model_validate(
        run_record.model_dump(
            mode="json",
            exclude={"partial", "end_reason"},
        )
        | {"status": "skipped", "task_status": "skipped"}
    )
    assert legacy.partial is False
    assert legacy.end_reason is EndReason.DEFERRED

    output = StringIO()
    render_experiment_result(
        Console(file=output, force_terminal=False),
        replayed,
        title="Cancelled experiment",
    )
    rendered = output.getvalue()
    assert "cancelled" in rendered
    assert "Cross-run derivation" in rendered
    assert "Policies" in rendered
    assert "run_missing" in rendered


def test_experiment_termination_rejects_inconsistent_run_identity_sets() -> None:
    with pytest.raises(ValidationError, match="must not overlap"):
        ExperimentTermination(
            partial=True,
            planned_run_ids=("run",),
            recorded_run_ids=("run",),
            missing_run_ids=("run",),
        )
    with pytest.raises(ValidationError, match="must be partial"):
        ExperimentTermination(status=ExperimentStatus.ABORTED)

    invalid_sets: tuple[dict[str, Any], ...] = (
        {"planned_run_ids": ("run", "run")},
        {"recorded_run_ids": ("run", "run")},
        {"missing_run_ids": ("run", "run"), "partial": True},
        {"planned_run_ids": ("planned",), "recorded_run_ids": ("other",)},
        {
            "planned_run_ids": ("planned",),
            "missing_run_ids": ("other",),
            "partial": True,
        },
        {"missing_run_ids": ("run",)},
    )
    for values in invalid_sets:
        with pytest.raises(ValidationError):
            ExperimentTermination.model_validate(values)


def test_record_views_reject_invalid_terminal_sections_and_serialize_unusual_values() -> None:
    record = RunRecord(
        run_id="run",
        experiment_id="exp",
        benchmark_id="benchmark",
        case_id="case",
        variant_id="variant",
        status=RunStatus.PASSED,
        evaluation_status=EvaluationStatus.PASSED,
        task_status=TaskStatus.PASSED,
        case=Case(id="case"),
        task_output=Path("output.txt"),
    )
    assert run_record_to_yaml_view(record)["output"] == "output.txt"

    class Opaque:
        def __repr__(self) -> str:
            return "opaque"

    assert (
        run_record_to_yaml_view(record.model_copy(update={"task_output": Opaque()}))["output"]
        == "opaque"
    )

    base = {
        "record": {"type": "experiment", "version": 5},
        "experiment": {"id": "exp", "benchmark": "benchmark"},
        "benchmark": {
            "id": "benchmark",
            "counts": {"cases": 0, "variants": 0, "runs": 0},
        },
        "runs": {"count": 0},
        "environment": {"python": "3.11", "platform": "test", "cwd": "."},
    }
    with pytest.raises(RecordingError, match="termination must be a mapping"):
        experiment_record_payload_from_yaml_view(
            base | {"experiment": base["experiment"] | {"termination": []}}
        )
    with pytest.raises(RecordingError, match="post_processing must be a mapping"):
        experiment_record_payload_from_yaml_view(
            base | {"experiment": base["experiment"] | {"termination": {"post_processing": []}}}
        )
    payload = experiment_record_payload_from_yaml_view(
        base | {"experiment": base["experiment"] | {"termination": {"post_processing": None}}}
    )
    assert payload["termination"]["cross_run_derivation_complete"] is True


def test_cancelled_status_mapping_and_invalid_legacy_status_are_explicit() -> None:
    cancelled = TaskResult(
        status=TaskStatus.CANCELLED,
        partial=True,
        end_reason=EndReason.CANCELLED,
    )
    assert (
        pipeline._evaluation_status_from_task_result(
            cancelled,
            scores=[],
            registry=_experiment_result().semantic_registry,
        )
        is EvaluationStatus.NOT_EVALUATED
    )
    assert (
        pipeline._run_status_from_task_result(
            cancelled,
            evaluation_status=EvaluationStatus.NOT_EVALUATED,
        )
        is RunStatus.CANCELLED
    )

    legacy_cancelled = RunRecord.model_validate(
        {
            "run_id": "cancelled",
            "experiment_id": "exp",
            "benchmark_id": "benchmark",
            "case_id": "case",
            "variant_id": "variant",
            "status": "cancelled",
        }
    )
    assert legacy_cancelled.partial is True
    assert legacy_cancelled.end_reason is EndReason.CANCELLED

    with pytest.raises(ValidationError):
        RunRecord.model_validate(
            {
                "run_id": "run",
                "experiment_id": "exp",
                "benchmark_id": "benchmark",
                "case_id": "case",
                "variant_id": "variant",
                "status": "unknown",
            }
        )


def _experiment_result(*, runs: list[RunResult] | None = None) -> ExperimentResult:
    active_runs = [_run_result()] if runs is None else runs
    return ExperimentResult(
        experiment_id="exp",
        benchmark_id="benchmark",
        plan=BenchmarkPlan(
            benchmark_id="benchmark",
            case_ids=tuple(run.case_id for run in active_runs),
            case_count=len(active_runs),
            variant_count=1,
            planned_run_count=len(active_runs),
        ),
        runs=active_runs,
        environment=EnvironmentMetadata(python_version="3.11", platform="test", cwd="."),
    )


def _run_result(
    *,
    case_id: str = "case",
    run_id: str = "run",
    artifact_id: str = "payload",
) -> RunResult:
    case = Case(id=case_id)
    variant = Variant(id="variant")
    context = RunContext(
        benchmark_id="benchmark",
        experiment_id="exp",
        run_id=run_id,
        case=case,
        variant=variant,
    )
    context.artifact(artifact_id, {"case": case_id})
    trace = context.finalize(output={"ok": True})
    return RunResult(
        run_id=run_id,
        benchmark_id="benchmark",
        experiment_id="exp",
        case_id=case_id,
        variant_id=variant.id,
        status=RunStatus.PASSED,
        evaluation_status=EvaluationStatus.PASSED,
        case=case,
        task_result=TaskResult(
            output={"ok": True},
            status=TaskStatus.PASSED,
            artifacts=list(context.artifacts),
            spans=list(context.spans),
        ),
        trace=trace,
    )
