from __future__ import annotations

import shutil
import sys
from collections.abc import Coroutine
from datetime import UTC, datetime, timedelta
from pathlib import Path
from textwrap import dedent
from typing import Any

import pytest
from click.testing import CliRunner
from pydantic import ValidationError
from rich.console import Console

import autobench.cli as cli_module
import autobench.records.staging as staging_module
import autobench.runtime.pipeline as pipeline_module
from autobench import (
    BenchmarkInfo,
    BenchmarkSpec,
    Case,
    DatasetSpec,
    EndReason,
    ErrorRecord,
    ExecutionSnapshot,
    ExperimentFile,
    ExperimentResult,
    ExperimentStart,
    ExperimentStatus,
    ExperimentTermination,
    FactorValue,
    FileRecorder,
    FileRecordSession,
    MatrixRunSpec,
    PartialRunSnapshot,
    RecordingError,
    RunContext,
    RunPhase,
    RunStatus,
    StagingHealth,
    StagingManifest,
    StagingStatus,
    TaskResult,
    TaskSpec,
    TaskStatus,
    Variant,
    archive_staging,
    build_benchmark_plan,
    capture_environment,
    discard_staging,
    expand_matrix,
    finalize_staging,
    inspect_staging,
    recover_staging,
    replay_experiment,
    run_benchmark_spec,
)
from autobench.cli import cli
from autobench.io import load_yaml
from autobench.records.models import ExperimentRecord
from autobench.records.staging import (
    RecordFileKind,
    RecordSession,
    load_staging_manifest,
    load_staging_state,
    partial_snapshot_from_yaml_view,
)
from autobench.reports.rich import render_staging_inspection


async def test_file_recorder_stages_concurrent_runs_and_publishes_in_plan_order(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spec = durable_spec(
        tmp_path,
        monkeypatch,
        module_name="durable_concurrent",
        reverse_completion=True,
    )
    output = tmp_path / "record"
    recorder = FileRecorder(output, trace_inline_limit_bytes=1)
    completion_order: list[str] = []
    original_stage = FileRecordSession.stage

    async def observe_stage(
        session: FileRecordSession,
        snapshot: ExecutionSnapshot,
    ) -> None:
        completion_order.append(snapshot.run.run_id)
        await original_stage(session, snapshot)

    monkeypatch.setattr(FileRecordSession, "stage", observe_stage)

    result = await run_benchmark_spec(
        spec,
        experiment_id="exp_durable",
        concurrency_limit=4,
        recorder=recorder,
    )

    assert result.total_count == 4
    assert completion_order == [
        "run_0002_0002_case_2__variant_2",
        "run_0002_0001_case_2__variant_1",
        "run_0001_0002_case_1__variant_2",
        "run_0001_0001_case_1__variant_1",
    ]
    assert output.is_dir()
    assert not recorder.staging_dir.exists()
    replayed = replay_experiment(output)
    assert [run.run_id for run in replayed.runs] == [
        "run_0001_0001_case_1__variant_1",
        "run_0001_0002_case_1__variant_2",
        "run_0002_0001_case_2__variant_1",
        "run_0002_0002_case_2__variant_2",
    ]
    manifest = load_yaml(output / "manifest.yaml")
    paths = {entry["path"] for entry in manifest["files"]}
    assert "cases/case_1/variant_1/run.yaml" in paths
    assert any(path.startswith("artifacts/") and path.endswith("trace.yaml") for path in paths)
    assert any(path.startswith("assets/") and path.endswith("index.yaml") for path in paths)

    staging_recorder = FileRecorder(tmp_path / "readable")
    staging_session = await staging_recorder.open(
        experiment_start(spec, experiment_id="exp_readable")
    )
    state_text = (staging_recorder.staging_dir / "staging.yaml").read_text(encoding="utf-8")
    manifest_text = (staging_recorder.staging_dir / "staging-manifest.yaml").read_text(
        encoding="utf-8"
    )
    assert state_text.startswith("# yaml-language-server: $schema=")
    assert "staging:\n  type: experiment" in state_text
    assert manifest_text.startswith("# yaml-language-server: $schema=")
    assert "staging:\n  type: manifest" in manifest_text
    await staging_session.checkpoint(
        partial_snapshot(staging_session.start.runs[0], name="readable", output={"step": 1})
    )
    checkpoint_path = staging_session.manifest.checkpoints[0].path
    checkpoint_text = (staging_recorder.staging_dir / checkpoint_path).read_text(encoding="utf-8")
    assert checkpoint_text.startswith("# yaml-language-server: $schema=")
    assert "checkpoint:\n  version: 1\n  name: readable" in checkpoint_text


async def test_experiment_publisher_cannot_overwrite_record_evidence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spec = durable_spec(tmp_path, monkeypatch, module_name="publication_collision")

    def collide_with_run(
        result: ExperimentResult,
        record: ExperimentRecord,
        experiment_root: Path,
    ) -> tuple[ExperimentFile, ...]:
        del experiment_root, result
        return (
            ExperimentFile(
                path=record.run_paths[0],
                content=b"replacement",
                identity="report:collision",
            ),
        )

    recorder = FileRecorder(
        tmp_path / "record",
        experiment_publishers=(collide_with_run,),
    )

    with pytest.raises(RecordingError, match="publication path already exists"):
        await run_benchmark_spec(spec, recorder=recorder)

    assert recorder.staging_dir.is_dir()


async def test_interrupted_pipeline_leaves_recoverable_partial_staging(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spec = durable_spec(tmp_path, monkeypatch, module_name="durable_interrupted")
    output = tmp_path / "record"
    recorder = FaultingRecorder(FileRecorder(output), fail_at="stage", stage_number=3)

    with pytest.raises(RecordingError, match="injected stage failure"):
        await run_benchmark_spec(
            spec,
            experiment_id="exp_interrupted",
            recorder=recorder,
        )

    staging = recorder.delegate.staging_dir
    orphan = staging / "cases" / "case_2" / "variant_1" / "run.yaml"
    orphan.parent.mkdir(parents=True)
    orphan.write_text("incomplete write", encoding="utf-8")
    inspection = inspect_staging(staging)
    assert inspection.health is StagingHealth.CONFLICTING
    assert inspection.recoverable is True
    assert inspection.status is StagingStatus.ABORTED
    assert inspection.complete_run_ids == (
        "run_0001_0001_case_1__variant_1",
        "run_0001_0002_case_1__variant_2",
    )
    assert len(inspection.missing_run_ids) == 2

    task_module = tmp_path / "durable_interrupted.py"
    task_module.unlink()
    sys.modules.pop("durable_interrupted", None)
    recovered = recover_staging(staging)
    assert len(recovered.runs) == 2

    with pytest.raises(RecordingError, match="allow_partial"):
        finalize_staging(staging, tmp_path / "strict")

    partial_output = tmp_path / "partial"
    record = finalize_staging(staging, partial_output, allow_partial=True)
    assert record.termination.status is ExperimentStatus.ABORTED
    assert record.termination.partial is True
    assert record.termination.recorded_run_ids == inspection.complete_run_ids
    replayed = replay_experiment(partial_output)
    assert replayed.total_count == 2
    assert replayed.termination.missing_run_ids == inspection.missing_run_ids
    assert staging.exists()


async def test_checkpoint_only_recovery_becomes_explicit_cancelled_run(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spec = durable_spec(tmp_path, monkeypatch, module_name="durable_checkpoint")
    output = tmp_path / "record"
    recorder = FileRecorder(output)
    start = experiment_start(spec, experiment_id="exp_checkpoint")
    session = await recorder.open(start)
    run_spec = start.runs[0]
    first = partial_snapshot(run_spec, name="working", output={"step": 1})
    latest = first.model_copy(
        update={
            "name": "latest",
            "captured_at": first.captured_at + timedelta(seconds=1),
            "task_output": {"step": 2},
        }
    )
    older = first.model_copy(
        update={
            "name": "older",
            "captured_at": first.captured_at - timedelta(seconds=1),
            "task_output": {"step": 0},
        }
    )

    await session.checkpoint(first)
    await session.checkpoint(first)
    await session.checkpoint(latest)
    await session.checkpoint(older)
    termination = ExperimentTermination(
        status=ExperimentStatus.CANCELLED,
        partial=True,
        cross_run_derivation_complete=False,
        policies_complete=False,
        planned_run_ids=tuple(run.run_id for run in start.runs),
        recorded_run_ids=(run_spec.run_id,),
        missing_run_ids=tuple(run.run_id for run in start.runs[1:]),
    )
    await session.abort(termination)
    await session.close()

    inspection = inspect_staging(recorder.staging_dir)
    assert inspection.health is StagingHealth.PARTIAL
    assert inspection.checkpointed_run_ids == (run_spec.run_id,)
    recovered = recover_staging(recorder.staging_dir)
    assert [item.name for item in recovered.checkpoints] == ["latest", "older", "working"]

    record = finalize_staging(
        recorder.staging_dir,
        output,
        allow_partial=True,
    )
    assert record.termination.status is ExperimentStatus.CANCELLED
    assert record.termination.recorded_run_ids == (run_spec.run_id,)
    replayed = replay_experiment(output)
    assert replayed.runs[0].status is RunStatus.CANCELLED
    assert replayed.runs[0].task_result.output == {"step": 2}
    assert replayed.runs[0].partial is True


async def test_stage_and_checkpoint_reject_conflicting_or_wrong_identity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spec = durable_spec(tmp_path, monkeypatch, module_name="durable_conflicts", one_run=True)
    result = await run_benchmark_spec(spec, experiment_id="exp_conflicts")
    recorder = FileRecorder(tmp_path / "record")
    session = await recorder.open(experiment_start(spec, experiment_id="exp_conflicts"))
    snapshot = ExecutionSnapshot.from_result(result.runs[0])

    await session.stage(snapshot)
    revision = session.manifest.revision
    await session.stage(snapshot)
    assert session.manifest.revision == revision
    complete_console = Console(record=True, width=100)
    render_staging_inspection(complete_console, inspect_staging(recorder.staging_dir))
    assert "Staging Complete" in complete_console.export_text()
    changed = snapshot.model_copy(
        update={
            "run": snapshot.run.model_copy(
                update={
                    "task_result": snapshot.run.task_result.model_copy(update={"output": "new"})
                }
            )
        }
    )
    with pytest.raises(RecordingError, match="different content"):
        await session.stage(changed)
    with pytest.raises(RecordingError, match="not part"):
        await session.stage(
            snapshot.model_copy(update={"run": snapshot.run.model_copy(update={"run_id": "other"})})
        )
    with pytest.raises(RecordingError, match="identity"):
        await session.stage(
            snapshot.model_copy(
                update={"run": snapshot.run.model_copy(update={"case_id": "wrong"})}
            )
        )

    run_spec = session.start.runs[0]
    checkpoint = partial_snapshot(run_spec, name="safe", output={"step": 1})
    await session.checkpoint(checkpoint)
    await session.checkpoint(checkpoint)
    with pytest.raises(RecordingError, match="conflicting content"):
        await session.checkpoint(checkpoint.model_copy(update={"task_output": {"step": 2}}))
    with pytest.raises(RecordingError, match="not part"):
        await session.checkpoint(checkpoint.model_copy(update={"run_id": "other"}))
    with pytest.raises(RecordingError, match="identity"):
        await session.checkpoint(checkpoint.model_copy(update={"variant_id": "wrong"}))

    await session.close()
    await session.abort(result.termination)
    with pytest.raises(RecordingError, match="closed"):
        await session.stage(snapshot)


async def test_finish_requires_complete_matching_result_and_cannot_repeat(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spec = durable_spec(tmp_path, monkeypatch, module_name="durable_finish", one_run=True)
    result = await run_benchmark_spec(spec, experiment_id="exp_finish")
    recorder = FileRecorder(tmp_path / "record")
    session = await recorder.open(experiment_start(spec, experiment_id="exp_finish"))

    with pytest.raises(RecordingError, match="missing staged runs"):
        await session.finish(result)
    with pytest.raises(RecordingError, match="does not belong"):
        await session.finish(result.model_copy(update={"experiment_id": "wrong"}))
    await session.stage(ExecutionSnapshot.from_result(result.runs[0]))
    record = await session.finish(result)
    assert isinstance(record, ExperimentRecord)
    with pytest.raises(RecordingError, match="already finished"):
        await session.finish(result)
    await session.abort(result.termination)
    await session.close()
    await session.close()


async def test_inspection_detects_corruption_orphaning_and_manifest_conflicts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spec = durable_spec(tmp_path, monkeypatch, module_name="durable_inspect", one_run=True)
    result = await run_benchmark_spec(spec, experiment_id="exp_inspect")

    corrupt_recorder = FileRecorder(tmp_path / "corrupt")
    corrupt_session = await corrupt_recorder.open(
        experiment_start(spec, experiment_id="exp_inspect")
    )
    await corrupt_session.stage(ExecutionSnapshot.from_result(result.runs[0]))
    staged = corrupt_session.manifest.runs[0]
    (corrupt_recorder.staging_dir / staged.record_path).write_text("changed", encoding="utf-8")
    inspection = inspect_staging(corrupt_recorder.staging_dir)
    assert inspection.health is StagingHealth.CORRUPT
    assert inspection.corrupt_run_ids == (result.runs[0].run_id,)
    with pytest.raises(RecordingError, match="corrupt"):
        recover_staging(corrupt_recorder.staging_dir)

    orphan_recorder = FileRecorder(tmp_path / "orphan")
    orphan_session = await orphan_recorder.open(experiment_start(spec, experiment_id="exp_inspect"))
    await orphan_session.stage(ExecutionSnapshot.from_result(result.runs[0]))
    (orphan_recorder.staging_dir / "orphan.txt").write_text("orphan", encoding="utf-8")
    orphaned = inspect_staging(orphan_recorder.staging_dir)
    assert orphaned.health is StagingHealth.CONFLICTING
    assert orphaned.recoverable is True
    assert orphaned.orphaned_files == ("orphan.txt",)
    assert len(recover_staging(orphan_recorder.staging_dir).runs) == 1

    manifest_path = orphan_recorder.staging_dir / "staging-manifest.yaml"
    manifest = load_yaml(manifest_path)
    manifest["runs"].append(manifest["runs"][0])
    from autobench.io import dump_yaml

    manifest_path.write_text(dump_yaml(manifest, schema_name="staging_manifest"), encoding="utf-8")
    conflicting = inspect_staging(orphan_recorder.staging_dir)
    assert conflicting.health is StagingHealth.CONFLICTING
    assert conflicting.recoverable is False
    assert "duplicate run ids" in conflicting.diagnostics[0]
    with pytest.raises(RecordingError, match="conflicting"):
        recover_staging(orphan_recorder.staging_dir)


async def test_checkpoint_corruption_and_uncommitted_run_data_are_not_accepted(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spec = durable_spec(
        tmp_path, monkeypatch, module_name="durable_checkpoint_corrupt", one_run=True
    )
    result = await run_benchmark_spec(spec, experiment_id="exp_checkpoint_corrupt")
    recorder = FileRecorder(tmp_path / "record")
    session = await recorder.open(experiment_start(spec, experiment_id="exp_checkpoint_corrupt"))
    checkpoint = partial_snapshot(session.start.runs[0], name="working", output={"step": 1})
    await session.checkpoint(checkpoint)
    checkpoint_path = recorder.staging_dir / session.manifest.checkpoints[0].path
    checkpoint_path.write_text("corrupt", encoding="utf-8")
    inspection = inspect_staging(recorder.staging_dir)
    assert inspection.health is StagingHealth.CORRUPT
    assert inspection.corrupt_run_ids == (checkpoint.run_id,)

    second_recorder = FileRecorder(tmp_path / "uncommitted")
    second_session = await second_recorder.open(
        experiment_start(spec, experiment_id="exp_checkpoint_corrupt")
    )
    run = result.runs[0]
    uncommitted_path = (
        second_recorder.staging_dir / "cases" / run.case_id / run.variant_id / "run.yaml"
    )
    uncommitted_path.parent.mkdir(parents=True)
    uncommitted_path.write_text("uncommitted", encoding="utf-8")
    with pytest.raises(RecordingError, match="Uncommitted staging data"):
        await second_session.stage(ExecutionSnapshot.from_result(run))


async def test_archive_and_discard_are_explicit_and_validate_staging(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spec = durable_spec(tmp_path, monkeypatch, module_name="durable_archive", one_run=True)
    recorder = FileRecorder(tmp_path / "record")
    session = await recorder.open(experiment_start(spec, experiment_id="exp_archive"))
    await session.close()

    archive = archive_staging(recorder.staging_dir, tmp_path / "archive")
    assert archive.is_dir()
    assert recorder.staging_dir.is_dir()
    assert inspect_staging(archive).health is StagingHealth.MISSING
    discard_staging(recorder.staging_dir)
    assert not recorder.staging_dir.exists()

    plain = tmp_path / "plain"
    plain.mkdir()
    with pytest.raises(RecordingError, match="Not an Autobench"):
        archive_staging(plain, tmp_path / "invalid-archive")
    with pytest.raises(RecordingError, match="Not an Autobench"):
        discard_staging(plain)


@pytest.mark.parametrize("fail_at", ["open", "stage", "finish"])
async def test_pipeline_propagates_required_recorder_failures(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    fail_at: str,
) -> None:
    spec = durable_spec(
        tmp_path,
        monkeypatch,
        module_name=f"durable_failure_{fail_at}",
        one_run=True,
    )
    recorder = FaultingRecorder(FileRecorder(tmp_path / fail_at), fail_at=fail_at)

    with pytest.raises(RecordingError, match=f"injected {fail_at} failure"):
        await run_benchmark_spec(
            spec,
            experiment_id=f"exp_{fail_at}",
            recorder=recorder,
        )

    if fail_at == "open":
        assert not recorder.delegate.staging_dir.exists()
    elif fail_at == "stage":
        assert inspect_staging(recorder.delegate.staging_dir).status is StagingStatus.ABORTED
    else:
        assert not recorder.delegate.output_dir.exists()
        assert inspect_staging(recorder.delegate.staging_dir).status is StagingStatus.ABORTED


async def test_abort_and_close_failures_do_not_replace_primary_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spec = durable_spec(tmp_path, monkeypatch, module_name="durable_cleanup", one_run=True)
    recorder = FaultingRecorder(
        FileRecorder(tmp_path / "record"),
        fail_at="stage",
        fail_abort=True,
        fail_close=True,
    )

    with pytest.raises(RecordingError, match="injected stage failure") as captured:
        await run_benchmark_spec(spec, experiment_id="exp_cleanup", recorder=recorder)

    notes = getattr(captured.value, "__notes__", [])
    assert notes == [
        "recording abort failed: injected abort failure",
        "recording close failed: injected close failure",
    ]


async def test_close_failure_after_publication_is_visible(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spec = durable_spec(tmp_path, monkeypatch, module_name="durable_close", one_run=True)
    recorder = FaultingRecorder(
        FileRecorder(tmp_path / "record"),
        fail_at="none",
        fail_close=True,
    )

    with pytest.raises(RecordingError, match="injected close failure"):
        await run_benchmark_spec(spec, experiment_id="exp_close", recorder=recorder)

    assert recorder.delegate.output_dir.is_dir()


def test_file_recorder_fails_before_task_execution_on_invalid_targets(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spec = durable_spec(tmp_path, monkeypatch, module_name="durable_fail_fast", one_run=True)
    spec_path = tmp_path / "benchmark.yaml"
    from autobench import benchmark_spec_to_yaml_view
    from autobench.io import dump_yaml

    spec_path.write_text(
        dump_yaml(benchmark_spec_to_yaml_view(spec), schema_name="benchmark"),
        encoding="utf-8",
    )
    output = tmp_path / "record"
    output.mkdir()
    (output / "existing").write_text("keep", encoding="utf-8")

    result = CliRunner().invoke(cli, ["run", str(spec_path), "--record", str(output)])

    assert result.exit_code == 1
    assert "Record target already exists" in result.output
    assert not (tmp_path / "task-executed").exists()


def test_file_recorder_validates_configuration_targets_and_open_cleanup(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spec = durable_spec(tmp_path, monkeypatch, module_name="durable_open", one_run=True)
    start = experiment_start(spec, experiment_id="exp_open")
    with pytest.raises(ValueError, match="at least 1"):
        FileRecorder(tmp_path / "invalid", trace_inline_limit_bytes=0)

    recorder = FileRecorder(tmp_path / "record", source_files=(tmp_path / "missing.py",))
    session = recorder.open_sync(start)
    assert session.start.file_hashes == ()
    from autobench.io import dump_yaml

    state_path = recorder.staging_dir / "staging.yaml"
    state_view = load_yaml(state_path)
    state_view["post_processing"] = "invalid"
    state_path.write_text(dump_yaml(state_view, schema_name="staging"), encoding="utf-8")
    loaded_start, _ = load_staging_state(recorder.staging_dir)
    assert loaded_start.requires_cross_run_derivation is False
    assert loaded_start.requires_policies is False
    with pytest.raises(RecordingError, match="Staging target already exists"):
        recorder.open_sync(start)

    symlink_target = tmp_path / "symlink-target"
    symlink_target.mkdir()
    symlink_output = tmp_path / "symlink-output"
    symlink_output.symlink_to(symlink_target, target_is_directory=True)
    with pytest.raises(RecordingError, match="Record target already exists"):
        FileRecorder(symlink_output).open_sync(start)

    file_output = tmp_path / "file-output"
    file_output.write_text("occupied", encoding="utf-8")
    with pytest.raises(RecordingError, match="Record target already exists"):
        FileRecorder(file_output).open_sync(start)

    empty_output = tmp_path / "empty-output"
    empty_output.mkdir()
    empty_recorder = FileRecorder(empty_output)
    empty_recorder.open_sync(start)
    discard_staging(empty_recorder.staging_dir)

    collision_spec = BenchmarkSpec(
        benchmark=BenchmarkInfo(id="collision"),
        dataset=DatasetSpec(cases=[Case(id="a-b"), Case(id="a_b")]),
        task=spec.task,
        variants=[Variant(id="same")],
    )
    with pytest.raises(RecordingError, match="collision"):
        FileRecorder(tmp_path / "collision").open_sync(
            experiment_start(collision_spec, experiment_id="exp_collision")
        )

    cleanup_recorder = FileRecorder(tmp_path / "cleanup")
    original_write = staging_module.atomic_write_text

    def fail_state_write(*args, **kwargs) -> None:
        raise OSError("write failed")

    monkeypatch.setattr(staging_module, "atomic_write_text", fail_state_write)
    with pytest.raises(OSError, match="write failed"):
        cleanup_recorder.open_sync(start)
    assert not cleanup_recorder.staging_dir.exists()
    monkeypatch.setattr(staging_module, "atomic_write_text", original_write)


def test_recording_cli_inspects_archives_finalizes_and_discards(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spec = durable_spec(tmp_path, monkeypatch, module_name="durable_cli", one_run=True)
    recorder = FileRecorder(tmp_path / "record")
    start = experiment_start(spec, experiment_id="exp_cli")
    session = recorder.open_sync(start)
    runner = CliRunner()

    inspected = runner.invoke(cli, ["recording", "inspect", str(recorder.staging_dir)])
    assert inspected.exit_code == 0
    assert "Staging Missing" in inspected.output
    no_confirmation = runner.invoke(
        cli,
        ["recording", "discard", str(recorder.staging_dir)],
    )
    assert no_confirmation.exit_code == 2

    archived = runner.invoke(
        cli,
        [
            "recording",
            "archive",
            str(recorder.staging_dir),
            "--output",
            str(tmp_path / "archive"),
        ],
    )
    assert archived.exit_code == 0
    incomplete = runner.invoke(
        cli,
        [
            "recording",
            "finalize",
            str(recorder.staging_dir),
            "--output",
            str(tmp_path / "strict"),
        ],
    )
    assert incomplete.exit_code == 1
    finalized = runner.invoke(
        cli,
        [
            "recording",
            "finalize",
            str(recorder.staging_dir),
            "--output",
            str(tmp_path / "partial"),
            "--allow-partial",
        ],
    )
    assert finalized.exit_code == 0
    discarded = runner.invoke(
        cli,
        ["recording", "discard", str(recorder.staging_dir), "--yes"],
    )
    assert discarded.exit_code == 0
    assert session.closed is False

    invalid = tmp_path / "invalid-staging"
    invalid.mkdir()
    for command in (
        ["recording", "inspect", str(invalid)],
        ["recording", "finalize", str(invalid), "--output", str(tmp_path / "bad-final")],
        ["recording", "archive", str(invalid), "--output", str(tmp_path / "bad-archive")],
        ["recording", "discard", str(invalid), "--yes"],
    ):
        failed = runner.invoke(cli, command)
        assert failed.exit_code == 1


def test_cli_recording_error_without_recorder_has_no_staging_hint(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spec = durable_spec(tmp_path, monkeypatch, module_name="durable_no_record", one_run=True)
    from autobench import benchmark_spec_to_yaml_view
    from autobench.io import dump_yaml

    spec_path = tmp_path / "no-record.yaml"
    spec_path.write_text(
        dump_yaml(benchmark_spec_to_yaml_view(spec), schema_name="benchmark"),
        encoding="utf-8",
    )

    def fail_run(awaitable: Coroutine[Any, Any, ExperimentResult]) -> None:
        awaitable.close()
        raise RecordingError("no recorder failure")

    monkeypatch.setattr(cli_module, "run_sync_cooperatively", fail_run)
    result = CliRunner().invoke(cli, ["run", str(spec_path), "--no-record"])
    assert result.exit_code == 1
    assert "no recorder failure" in result.output
    assert "Staging path" not in result.output


def test_staging_models_and_yaml_views_reject_invalid_shapes(tmp_path: Path) -> None:
    with pytest.raises(ValidationError, match="planned run ids"):
        start = experiment_start(
            BenchmarkSpec(
                benchmark=BenchmarkInfo(id="invalid"),
                dataset=DatasetSpec(cases=[Case(id="case")]),
                task=TaskSpec(kind="python", target="invalid:run"),
                variants=[Variant(id="variant")],
            ),
            experiment_id="exp_invalid",
        )
        ExperimentStart.model_validate(
            start.model_dump(mode="python")
            | {
                "runs": [start.runs[0], start.runs[0]],
                "plan": start.plan.model_copy(update={"planned_run_count": 2}),
            }
        )
    valid_spec = BenchmarkSpec(
        benchmark=BenchmarkInfo(id="valid"),
        dataset=DatasetSpec(cases=[Case(id="case")]),
        task=TaskSpec(kind="python", target="valid:run"),
        variants=[Variant(id="variant")],
    )
    valid_start = experiment_start(valid_spec, experiment_id="exp_valid")
    for update, message in (
        ({"plan": valid_start.plan.model_copy(update={"planned_run_count": 2})}, "plan count"),
        (
            {"runs": (valid_start.runs[0].model_copy(update={"experiment_id": "other"}),)},
            "belong to the experiment",
        ),
        (
            {"runs": (valid_start.runs[0].model_copy(update={"benchmark_id": "other"}),)},
            "belong to the benchmark",
        ),
    ):
        with pytest.raises(ValidationError, match=message):
            ExperimentStart.model_validate(valid_start.model_dump(mode="python") | update)

    checkpoint_file = {
        "path": "checkpoints/run/safe.yaml",
        "sha256": "0" * 64,
        "byte_count": 0,
        "kind": "run",
        "identity": "run:safe",
    }
    checkpoint = {
        "run_id": "run",
        "name": "safe",
        "path": "checkpoints/run/safe.yaml",
        "snapshot_hash": "1" * 64,
        "captured_at": datetime.now(UTC),
        "signal_sequence_watermark": 0,
        "file": checkpoint_file,
    }
    with pytest.raises(ValidationError, match="duplicate checkpoints"):
        StagingManifest.model_validate(
            {
                "experiment_id": "exp",
                "checkpoints": (checkpoint, checkpoint),
            }
        )
    with pytest.raises(ValidationError, match="conflicting file paths"):
        StagingManifest.model_validate(
            {
                "experiment_id": "exp",
                "runs": (
                    {
                        "run_id": "run",
                        "case_id": "case",
                        "variant_id": "variant",
                        "record_path": "cases/case/variant/run.yaml",
                        "snapshot_hash": "2" * 64,
                        "captured_at": datetime.now(UTC),
                        "signal_sequence_watermark": 0,
                        "files": (checkpoint_file,),
                    },
                ),
                "checkpoints": (checkpoint,),
            }
        )
    with pytest.raises(RecordingError, match="must be a mapping"):
        partial_snapshot_from_yaml_view([])
    with pytest.raises(RecordingError, match="missing checkpoint"):
        partial_snapshot_from_yaml_view({})
    with pytest.raises(RecordingError, match="Not an Autobench"):
        load_staging_state(tmp_path)
    malformed = tmp_path / "malformed"
    malformed.mkdir()
    (malformed / "staging.yaml").write_text("[]\n", encoding="utf-8")
    with pytest.raises(RecordingError, match="must be a mapping"):
        load_staging_state(malformed)
    (malformed / "staging-manifest.yaml").write_text("[]\n", encoding="utf-8")
    with pytest.raises(RecordingError, match="must be a mapping"):
        load_staging_manifest(malformed)
    (malformed / "staging.yaml").write_text("staging: {}\n", encoding="utf-8")
    with pytest.raises(RecordingError, match="missing staging or experiment"):
        load_staging_state(malformed)
    (malformed / "staging-manifest.yaml").write_text("staging: {}\n", encoding="utf-8")
    with pytest.raises(RecordingError, match="missing staging or experiment"):
        load_staging_manifest(malformed)

    staging_target = tmp_path / "staging-target"
    staging_target.mkdir()
    staging_link = tmp_path / "staging-link"
    staging_link.symlink_to(staging_target, target_is_directory=True)
    with pytest.raises(RecordingError, match="Not an Autobench"):
        load_staging_state(staging_link)


async def test_snapshot_without_trace_and_staged_file_integrity_edges(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spec = durable_spec(tmp_path, monkeypatch, module_name="durable_integrity", one_run=True)
    result = await run_benchmark_spec(spec, experiment_id="exp_integrity")
    without_trace = result.runs[0].model_copy(update={"trace": None})
    assert ExecutionSnapshot.from_result(without_trace).signal_sequence_watermark == 0

    missing_recorder = FileRecorder(tmp_path / "missing")
    missing_session = await missing_recorder.open(
        experiment_start(spec, experiment_id="exp_integrity")
    )
    await missing_session.stage(ExecutionSnapshot.from_result(result.runs[0]))
    missing_path = missing_recorder.staging_dir / missing_session.manifest.runs[0].record_path
    missing_path.unlink()
    missing_inspection = inspect_staging(missing_recorder.staging_dir)
    assert "missing staged file" in " ".join(missing_inspection.diagnostics)

    escape_recorder = FileRecorder(tmp_path / "escape")
    escape_session = await escape_recorder.open(
        experiment_start(spec, experiment_id="exp_integrity")
    )
    await escape_session.stage(ExecutionSnapshot.from_result(result.runs[0]))
    staged = escape_session.manifest.runs[0]
    escaped_path = escape_recorder.staging_dir / staged.record_path
    outside = tmp_path / "outside.yaml"
    shutil.copy2(escaped_path, outside)
    escaped_path.unlink()
    escaped_path.symlink_to(outside)
    escaped = inspect_staging(escape_recorder.staging_dir)
    assert "escapes its root" in " ".join(escaped.diagnostics)

    console = Console(record=True, width=100)
    render_staging_inspection(console, escaped)
    assert "escapes its root" in console.export_text()


@pytest.mark.parametrize(
    "file_kind",
    [RecordFileKind.ARTIFACT, RecordFileKind.ASSET, RecordFileKind.TRACE],
)
async def test_inspection_rejects_corrupt_committed_evidence_payloads(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    file_kind: RecordFileKind,
) -> None:
    spec = durable_spec(
        tmp_path,
        monkeypatch,
        module_name=f"durable_{file_kind.value}",
        one_run=True,
    )
    result = await run_benchmark_spec(spec, experiment_id=f"exp_{file_kind.value}")
    recorder = FileRecorder(tmp_path / file_kind.value, trace_inline_limit_bytes=1)
    session = await recorder.open(experiment_start(spec, experiment_id=f"exp_{file_kind.value}"))
    await session.stage(ExecutionSnapshot.from_result(result.runs[0]))
    entry = next(item for item in session.manifest.runs[0].files if item.kind is file_kind)
    (recorder.staging_dir / entry.path).write_bytes(b"corrupt payload")

    inspection = inspect_staging(recorder.staging_dir)

    assert inspection.health is StagingHealth.CORRUPT
    assert inspection.recoverable is False
    assert result.runs[0].run_id in inspection.corrupt_run_ids


async def test_inspection_distinguishes_recoverable_and_fatal_metadata_conflicts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spec = durable_spec(tmp_path, monkeypatch, module_name="durable_metadata", one_run=True)
    result = await run_benchmark_spec(spec, experiment_id="exp_metadata")

    revision_recorder = FileRecorder(tmp_path / "revision")
    revision_session = await revision_recorder.open(
        experiment_start(spec, experiment_id="exp_metadata")
    )
    await revision_session.stage(ExecutionSnapshot.from_result(result.runs[0]))
    state = revision_session.state.model_copy(update={"revision": 2})
    staging_module.atomic_write_text(
        revision_recorder.staging_dir / staging_module.STAGING_STATE_PATH,
        staging_module.dump_yaml(
            staging_module.experiment_start_to_yaml_view(revision_session.start, state),
            schema_name="staging",
        ),
    )
    revision_inspection = inspect_staging(revision_recorder.staging_dir)
    assert revision_inspection.health is StagingHealth.CONFLICTING
    assert revision_inspection.recoverable is True
    assert "revisions differ" in " ".join(revision_inspection.diagnostics)
    assert len(recover_staging(revision_recorder.staging_dir).runs) == 1

    experiment_recorder = FileRecorder(tmp_path / "experiment")
    experiment_session = await experiment_recorder.open(
        experiment_start(spec, experiment_id="exp_metadata")
    )
    await experiment_session.stage(ExecutionSnapshot.from_result(result.runs[0]))
    wrong_experiment = experiment_session.manifest.model_copy(
        update={"experiment_id": "another_experiment"}
    )
    experiment_session.write_state_and_manifest(experiment_session.state, wrong_experiment)
    experiment_inspection = inspect_staging(experiment_recorder.staging_dir)
    assert experiment_inspection.recoverable is False
    assert "does not match staging state" in " ".join(experiment_inspection.diagnostics)


async def test_inspection_rejects_manifest_and_payload_identity_conflicts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spec = durable_spec(tmp_path, monkeypatch, module_name="durable_identity", one_run=True)
    result = await run_benchmark_spec(spec, experiment_id="exp_identity")

    unknown_recorder = FileRecorder(tmp_path / "unknown")
    unknown_session = await unknown_recorder.open(
        experiment_start(spec, experiment_id="exp_identity")
    )
    await unknown_session.stage(ExecutionSnapshot.from_result(result.runs[0]))
    unknown_run = unknown_session.manifest.runs[0].model_copy(
        update={"run_id": "unknown", "record_path": "uncommitted-record.yaml"}
    )
    unknown_manifest = unknown_session.manifest.model_copy(update={"runs": (unknown_run,)})
    unknown_session.write_state_and_manifest(unknown_session.state, unknown_manifest)
    unknown = inspect_staging(unknown_recorder.staging_dir)
    assert unknown.recoverable is False
    assert "not in the experiment plan" in " ".join(unknown.diagnostics)
    assert "not committed by its file list" in " ".join(unknown.diagnostics)
    assert "invalid staged run record" in " ".join(unknown.diagnostics)

    mismatch_recorder = FileRecorder(tmp_path / "mismatch")
    mismatch_session = await mismatch_recorder.open(
        experiment_start(spec, experiment_id="exp_identity")
    )
    await mismatch_session.stage(ExecutionSnapshot.from_result(result.runs[0]))
    mismatched_run = mismatch_session.manifest.runs[0].model_copy(update={"case_id": "wrong"})
    mismatch_manifest = mismatch_session.manifest.model_copy(update={"runs": (mismatched_run,)})
    mismatch_session.write_state_and_manifest(mismatch_session.state, mismatch_manifest)
    mismatch = inspect_staging(mismatch_recorder.staging_dir)
    assert mismatch.recoverable is False
    assert "does not match its plan" in " ".join(mismatch.diagnostics)
    assert "payload identity" in " ".join(mismatch.diagnostics)


async def test_inspection_rejects_checkpoint_manifest_and_payload_conflicts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spec = durable_spec(tmp_path, monkeypatch, module_name="durable_checkpoint_id", one_run=True)
    start = experiment_start(spec, experiment_id="exp_checkpoint_id")

    unknown_recorder = FileRecorder(tmp_path / "unknown-checkpoint")
    unknown_session = await unknown_recorder.open(start)
    await unknown_session.checkpoint(
        partial_snapshot(unknown_session.start.runs[0], name="working", output={"step": 1})
    )
    unknown_checkpoint = unknown_session.manifest.checkpoints[0].model_copy(
        update={"run_id": "unknown", "path": "missing-checkpoint.yaml"}
    )
    unknown_manifest = unknown_session.manifest.model_copy(
        update={"checkpoints": (unknown_checkpoint,)}
    )
    unknown_session.write_state_and_manifest(unknown_session.state, unknown_manifest)
    unknown = inspect_staging(unknown_recorder.staging_dir)
    assert unknown.recoverable is False
    assert "checkpoint run is not in" in " ".join(unknown.diagnostics)
    assert "path is not committed" in " ".join(unknown.diagnostics)
    assert "invalid staged checkpoint" in " ".join(unknown.diagnostics)

    mismatch_recorder = FileRecorder(tmp_path / "mismatch-checkpoint")
    mismatch_session = await mismatch_recorder.open(start)
    await mismatch_session.checkpoint(
        partial_snapshot(mismatch_session.start.runs[0], name="working", output={"step": 1})
    )
    mismatched_checkpoint = mismatch_session.manifest.checkpoints[0].model_copy(
        update={"name": "different"}
    )
    mismatch_manifest = mismatch_session.manifest.model_copy(
        update={"checkpoints": (mismatched_checkpoint,)}
    )
    mismatch_session.write_state_and_manifest(mismatch_session.state, mismatch_manifest)
    mismatch = inspect_staging(mismatch_recorder.staging_dir)
    assert mismatch.recoverable is False
    assert "checkpoint payload identity" in " ".join(mismatch.diagnostics)


class FaultingRecorder:
    def __init__(
        self,
        delegate: FileRecorder,
        *,
        fail_at: str,
        stage_number: int = 1,
        fail_abort: bool = False,
        fail_close: bool = False,
    ) -> None:
        self.delegate = delegate
        self.fail_at = fail_at
        self.stage_number = stage_number
        self.fail_abort = fail_abort
        self.fail_close = fail_close

    async def open(self, start: ExperimentStart) -> RecordSession:
        if self.fail_at == "open":
            raise RecordingError("injected open failure")
        session = await self.delegate.open(start)
        return FaultingSession(
            session,
            fail_at=self.fail_at,
            stage_number=self.stage_number,
            fail_abort=self.fail_abort,
            fail_close=self.fail_close,
        )


class FaultingSession:
    def __init__(
        self,
        delegate: FileRecordSession,
        *,
        fail_at: str,
        stage_number: int,
        fail_abort: bool,
        fail_close: bool,
    ) -> None:
        self.delegate = delegate
        self.fail_at = fail_at
        self.stage_number = stage_number
        self.fail_abort = fail_abort
        self.fail_close = fail_close
        self.stage_count = 0

    @property
    def artifact_sink(self) -> FileRecordSession:
        return self.delegate.artifact_sink

    async def stage(self, snapshot: ExecutionSnapshot) -> None:
        self.stage_count += 1
        if self.fail_at == "stage" and self.stage_count == self.stage_number:
            raise RecordingError("injected stage failure")
        await self.delegate.stage(snapshot)

    async def checkpoint(self, snapshot: PartialRunSnapshot) -> None:
        await self.delegate.checkpoint(snapshot)

    async def finish(self, result: ExperimentResult) -> ExperimentRecord:
        if self.fail_at == "finish":
            raise RecordingError("injected finish failure")
        return await self.delegate.finish(result)

    async def abort(self, termination: ExperimentTermination) -> None:
        if self.fail_abort:
            raise RecordingError("injected abort failure")
        await self.delegate.abort(termination)

    async def close(self) -> None:
        if self.fail_close:
            raise RecordingError("injected close failure")
        await self.delegate.close()


async def test_record_session_can_delegate_artifacts_without_implementing_sink(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "payload.txt"
    source.write_text("durable payload", encoding="utf-8")

    async def artifact_task(
        target: str,
        *,
        ctx: RunContext,
        case: Case,
        search_paths: tuple[str, ...] = (),
    ) -> TaskResult:
        artifact = ctx.artifact_file("payload", source)
        return TaskResult(output={"artifact_id": artifact.id}, status=TaskStatus.PASSED)

    monkeypatch.setattr(pipeline_module, "run_python_task", artifact_task)
    spec = BenchmarkSpec(
        benchmark=BenchmarkInfo(id="delegated-artifacts"),
        dataset=DatasetSpec(cases=[Case(id="case")]),
        task=TaskSpec(kind="python", target="unused:run"),
        variants=[Variant(id="baseline")],
    )
    output = tmp_path / "record"
    recorder = FaultingRecorder(FileRecorder(output), fail_at="never")

    result = await run_benchmark_spec(spec, experiment_id="exp_delegate", recorder=recorder)

    assert result.runs[0].task_result.status is TaskStatus.PASSED
    assert len(result.runs[0].task_result.artifacts) == 1
    assert any(
        path.is_file() and path.suffix == ".txt" for path in (output / "artifacts").rglob("*")
    )


def durable_spec(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    module_name: str,
    one_run: bool = False,
    reverse_completion: bool = False,
) -> BenchmarkSpec:
    async_keyword = "async " if reverse_completion else ""
    delay = (
        """
                delays = {
                    "case_1:variant_1": 0.04,
                    "case_1:variant_2": 0.03,
                    "case_2:variant_1": 0.02,
                    "case_2:variant_2": 0.01,
                }
                await asyncio.sleep(delays[f"{case.id}:{ctx.variant.id}"])
        """
        if reverse_completion
        else ""
    )
    (tmp_path / f"{module_name}.py").write_text(
        dedent(
            f"""
            import asyncio

            from autobench import Semantic, track

            PROMPT = track.prompt(
                name="{module_name}_prompt",
                text="Use durable evidence.",
            )

            {async_keyword}def run(ctx, case):
                {delay}
                ctx.attach_tracked_asset(PROMPT)
                ctx.metric("quality", 1.0, semantic_type=Semantic.QUALITY_SCORE)
                ctx.artifact("details", f"{{case.id}}:{{ctx.variant.id}}", media_type="text/plain")
                return {{"case": case.id, "variant": ctx.variant.id}}
            """
        ).strip()
        + "\n",
        encoding="utf-8",
    )
    monkeypatch.syspath_prepend(str(tmp_path))
    cases = [Case(id="case_1", input={"value": 1})]
    variants = [Variant(id="variant_1", factors=[FactorValue(name="model", value="a")])]
    if not one_run:
        cases.append(Case(id="case_2", input={"value": 2}))
        variants.append(Variant(id="variant_2", factors=[FactorValue(name="model", value="b")]))
    return BenchmarkSpec(
        benchmark=BenchmarkInfo(id=module_name),
        dataset=DatasetSpec(cases=cases),
        task=TaskSpec(kind="python", target=f"{module_name}:run"),
        variants=variants,
    )


def experiment_start(spec: BenchmarkSpec, *, experiment_id: str) -> ExperimentStart:
    return ExperimentStart(
        experiment_id=experiment_id,
        benchmark_id=spec.benchmark.id,
        plan=build_benchmark_plan(spec),
        runs=tuple(expand_matrix(spec, experiment_id=experiment_id)),
        environment=capture_environment(),
        semantic_registry=spec.semantic_registry,
        report_spec_data=spec.reports.model_dump(mode="json"),
        spec_snapshot=spec.model_dump(mode="json"),
    )


def partial_snapshot(
    run_spec: MatrixRunSpec,
    *,
    name: str,
    output: dict[str, int],
) -> PartialRunSnapshot:
    return PartialRunSnapshot(
        run_id=run_spec.run_id,
        experiment_id=run_spec.experiment_id,
        benchmark_id=run_spec.benchmark_id,
        case_id=run_spec.case.id,
        variant_id=run_spec.variant.id,
        name=name,
        phase=RunPhase.EXECUTING,
        captured_at=datetime.now(UTC),
        task_status=TaskStatus.CANCELLED,
        end_reason=EndReason.CANCELLED,
        task_output=output,
        errors=(ErrorRecord(error_type="CancelledError", message="cancelled"),),
    )
