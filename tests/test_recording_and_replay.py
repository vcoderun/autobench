from __future__ import annotations as _annotations

import sys
from pathlib import Path
from textwrap import dedent
from typing import Any

import pytest
from click.testing import CliRunner
from pydantic import ValidationError

from autobench import (
    DEFAULT_SEMANTIC_REGISTRY,
    ArtifactRef,
    AssetVersion,
    BenchmarkInfo,
    BenchmarkPlan,
    BenchmarkSpec,
    Case,
    DatasetSpec,
    EnvironmentMetadata,
    ErrorRecord,
    EvaluationStatus,
    ExperimentRecord,
    RecordingError,
    RunRecord,
    RunResult,
    RunStatus,
    Semantic,
    TaskResult,
    TaskSpec,
    TaskStatus,
    Variant,
    load_experiment_record,
    load_run_record,
    record_experiment,
    replay_experiment,
    run_benchmark_spec,
    run_record_from_result,
)
from autobench.cli import cli
from autobench.data.variants import FactorValue
from autobench.evaluation.scoring import PassFailScorer
from autobench.io import dump_yaml, load_yaml
from autobench.metrics.observations import Observation, ObservationKind, ObservationRole
from autobench.records.recording import (
    _benchmark_spec_snapshot_payload,
    _benchmark_spec_snapshot_view,
    experiment_record_payload_from_yaml_view,
    experiment_record_to_yaml_view,
    run_record_payload_from_yaml_view,
    run_record_to_yaml_view,
)


async def test_record_experiment_writes_successful_and_failed_run_records(
    tmp_path: Path,
    monkeypatch,
) -> None:
    _write_module(
        tmp_path,
        "record_tasks.py",
        """
        def run(ctx, case):
            with ctx.span("task") as span:
                span.factor("model", "demo-model", semantic_type="llm.model.name")
                span.metric("input_tokens", 10, semantic_type="llm.tokens.input")
                span.artifact("payload", {"case": case.id})
            if case.id == "case_fail":
                raise RuntimeError("expected failure")
            return {"success": True, "case_id": case.id}
        """,
    )
    monkeypatch.syspath_prepend(str(tmp_path))
    source_file = tmp_path / "autobench.yaml"
    source_file.write_text("benchmark:\n  id: record-demo\n", encoding="utf-8")
    spec = _recording_spec()

    result = await run_benchmark_spec(spec, experiment_id="exp_fixed")
    record_dir = tmp_path / "runs" / result.experiment_id
    record = record_experiment(result, record_dir, source_files=[source_file])

    assert (record_dir / "experiment.yaml").exists()
    assert (record_dir / "summary.yaml").exists()
    assert record.run_count == 2
    assert record.file_hashes[0].path == str(source_file.resolve())
    raw_experiment = load_yaml(record_dir / "experiment.yaml")
    assert raw_experiment["record"] == {"type": "experiment", "version": 3}
    assert raw_experiment["experiment"]["id"] == result.experiment_id
    assert raw_experiment["benchmark"]["id"] == "record-demo"
    assert raw_experiment["benchmark"]["counts"] == {"cases": 2, "variants": 1, "runs": 2}
    assert raw_experiment["benchmark"]["cases"] == ["case_ok", "case_fail"]
    assert raw_experiment["runs"]["count"] == 2
    assert raw_experiment["runs"]["paths"] == [
        "cases/case_ok/variant_1/run.yaml",
        "cases/case_fail/variant_1/run.yaml",
    ]
    assert raw_experiment["environment"]["python"] == record.environment.python_version
    assert raw_experiment["files"] == {str(source_file.resolve()): record.file_hashes[0].sha256}
    raw_summary = load_yaml(record_dir / "summary.yaml")
    assert raw_summary["record"] == {"type": "summary", "version": 3}
    assert raw_summary["runs"]["failed"] == 1
    assert (record_dir / "cases" / "case_ok" / "variant_1" / "run.yaml").exists()
    assert (record_dir / "cases" / "case_fail" / "variant_1" / "run.yaml").exists()

    failed = load_run_record(record_dir / "cases" / "case_fail" / "variant_1" / "run.yaml")
    assert failed.status is RunStatus.FAILED
    assert failed.evaluation_status is EvaluationStatus.NOT_EVALUATED
    assert failed.errors[0].error_type == "RuntimeError"
    assert failed.factors[0].name == "model"
    assert failed.observations
    assert failed.spans[0].name == "task"
    raw_failed = load_yaml(record_dir / "cases" / "case_fail" / "variant_1" / "run.yaml")
    assert raw_failed["record"] == {"type": "run", "version": 3}
    assert raw_failed["run"]["outcome"]["task"] == "failed"
    assert raw_failed["variant"]["factors"]["model"]["value"] == "demo-model"
    assert raw_failed["metrics"]["factors"]["model"]["semantic"] == Semantic.LLM_MODEL_NAME
    assert raw_failed["metrics"]["measurements"]["input_tokens"]["semantic"] == (
        Semantic.LLM_TOKENS_INPUT
    )
    assert raw_failed["spans"][failed.spans[0].id]["name"] == "task"
    assert raw_failed["artifacts"]["payload"]["path"].startswith("artifacts/")
    assert "record_version" not in raw_failed


def test_experiment_record_yaml_view_round_trips_and_rejects_bad_spec() -> None:
    spec = BenchmarkSpec(
        benchmark=BenchmarkInfo(id="bench_1", description="Recording demo"),
        dataset=DatasetSpec(cases=[Case(id="case_1", input={"message": "hi"})]),
        task=TaskSpec(kind="python", target="app.tasks:run"),
        variants=[
            Variant(
                id="variant_1",
                factors=[
                    FactorValue(
                        name="prompt_version",
                        value="v1",
                        semantic_type=Semantic.PROMPT_VERSION,
                    )
                ],
            )
        ],
        scoring=[
            PassFailScorer(
                name="success",
                semantic_type=Semantic.RESULT_SUCCESS,
                path="output.success",
            )
        ],
    )
    record = ExperimentRecord(
        experiment_id="exp_1",
        benchmark_id="bench_1",
        plan=BenchmarkPlan(
            benchmark_id="bench_1",
            case_count=1,
            variant_count=1,
            planned_run_count=1,
        ),
        environment=EnvironmentMetadata(
            python_version="3.11",
            platform="test",
            cwd="/tmp",
        ),
        run_paths=("cases/case_1/variant_1/run.yaml",),
        run_count=1,
        passed_count=1,
        failed_count=0,
        errored_count=0,
        skipped_count=0,
        spec_hash="abc",
        spec_snapshot=spec.model_dump(mode="json"),
        report_spec_data={"leaderboard": {"metrics": []}},
    )

    view = experiment_record_to_yaml_view(record)
    view_without_reports = experiment_record_to_yaml_view(
        record.model_copy(update={"report_spec_data": None})
    )
    payload = experiment_record_payload_from_yaml_view(view)
    payload_without_reports = experiment_record_payload_from_yaml_view(
        view_without_reports | {"benchmark": view_without_reports["benchmark"] | {"spec": None}}
    )
    round_trip = ExperimentRecord.model_validate(payload)

    assert view["benchmark"]["spec"]["hash"] == "abc"
    assert view["benchmark"]["counts"] == {"cases": 1, "variants": 1, "runs": 1}
    assert view["benchmark"]["spec"]["snapshot"] == {
        "benchmark": {
            "bench_1": {
                "description": "Recording demo",
                "dataset": {
                    "cases": [{"id": "case_1", "input": {"message": "hi"}}],
                },
                "run": {"python": "app.tasks:run"},
                "variants": {
                    "variant_1": {
                        "factors": {
                            "prompt_version": {
                                "value": "v1",
                                "semantic": "prompt.version",
                            }
                        }
                    }
                },
                "score": {
                    "success": {
                        "pass": "output.success",
                        "semantic": "result.success",
                    }
                },
            }
        }
    }
    assert view["environment"] == {
        "python": "3.11",
        "platform": "test",
        "cwd": "/tmp",
    }
    assert "reports" not in view_without_reports
    assert "report_spec_data" not in payload_without_reports
    assert round_trip == record
    assert experiment_record_payload_from_yaml_view({"record": {"type": "run"}}) == {
        "record": {"type": "run"}
    }
    with pytest.raises(RecordingError, match="benchmark.spec must be a mapping"):
        experiment_record_payload_from_yaml_view(
            view | {"benchmark": {"id": "bench_1", "spec": []}}
        )


def test_experiment_record_payload_supports_flattened_dsl_sections() -> None:
    raw = {
        "record": {"type": "experiment", "version": 3},
        "experiment": {"id": "exp_1", "benchmark": "bench_1"},
        "benchmark": {
            "id": "bench_1",
            "dataset": {"id": "tickets", "version": "v2", "hash": "dataset-hash"},
            "cases": None,
            "counts": {"cases": 0, "variants": 2, "runs": 2},
            "warnings": None,
            "spec": None,
        },
        "runs": {
            "count": 2,
            "passed": 1,
            "failed": 1,
            "errored": 0,
            "skipped": 0,
            "paths": ["cases/ticket_1/variant_1/run.yaml"],
        },
        "files": {"/tmp/autobench.yaml": "abc123"},
        "environment": {"python": "3.11.13", "platform": "test", "cwd": "/tmp"},
        "semantic_registry": {
            "version": 1,
            "types": {"custom.metric": {"shape": "number"}},
            "aliases": {"custom.alias": "custom.metric"},
        },
    }

    payload = experiment_record_payload_from_yaml_view(raw)
    record = ExperimentRecord.model_validate(payload)

    assert record.plan.dataset_id == "tickets"
    assert record.plan.dataset_version == "v2"
    assert record.plan.dataset_hash == "dataset-hash"
    assert record.plan.case_ids == ()
    assert record.plan.variant_count == 2
    assert record.plan.planned_run_count == 2
    assert record.plan.warnings == []
    assert record.environment.python_version == "3.11.13"
    assert record.file_hashes[0].path == "/tmp/autobench.yaml"
    assert record.file_hashes[0].sha256 == "abc123"
    assert record.semantic_registry.types["custom.metric"].value_shape == "number"


def test_experiment_record_payload_supports_legacy_plan_and_file_list_shapes() -> None:
    raw = {
        "record": {"type": "experiment", "version": 3},
        "experiment": {"id": "exp_1", "benchmark": "bench_1"},
        "benchmark": {
            "id": "bench_1",
            "plan": {
                "benchmark_id": "bench_1",
                "dataset_id": "tickets",
                "case_count": 1,
                "variant_count": 1,
                "planned_run_count": 1,
            },
            "spec": {},
        },
        "runs": {"count": 1, "passed": 1, "failed": 0, "errored": 0, "skipped": 0},
        "files": [{"path": "/tmp/autobench.yaml", "sha256": "abc123"}],
        "environment": {"python_version": "3.11.13", "platform": "test", "cwd": "/tmp"},
        "semantic_registry": DEFAULT_SEMANTIC_REGISTRY.model_dump(mode="json"),
    }

    record = ExperimentRecord.model_validate(experiment_record_payload_from_yaml_view(raw))

    assert record.plan.benchmark_id == "bench_1"
    assert record.plan.dataset_id == "tickets"
    assert record.file_hashes[0].path == "/tmp/autobench.yaml"
    assert record.environment.python_version == "3.11.13"


def test_benchmark_spec_snapshot_helpers_cover_none_and_passthrough_branches() -> None:
    assert _benchmark_spec_snapshot_view(None) is None
    assert _benchmark_spec_snapshot_payload(None) is None
    assert _benchmark_spec_snapshot_payload("raw-snapshot") == "raw-snapshot"
    assert _benchmark_spec_snapshot_view({"benchmark": "invalid"}) == {"benchmark": "invalid"}


@pytest.mark.parametrize(
    ("patch", "message"),
    [
        ({"benchmark": {"dataset": []}}, "benchmark.dataset must be a mapping"),
        ({"benchmark": {"counts": []}}, "benchmark.counts must be a mapping"),
        ({"benchmark": {"cases": 1}}, "benchmark.cases must be a list"),
        ({"benchmark": {"warnings": 1}}, "benchmark.warnings must be a list"),
        ({"files": 1}, "files must be a mapping or list"),
    ],
)
def test_experiment_record_payload_rejects_invalid_flattened_sections(
    patch: dict[str, Any],
    message: str,
) -> None:
    raw = {
        "record": {"type": "experiment", "version": 3},
        "experiment": {"id": "exp_1", "benchmark": "bench_1"},
        "benchmark": {
            "id": "bench_1",
            "counts": {"cases": 1, "variants": 1, "runs": 1},
            "spec": {},
        },
        "runs": {"count": 1, "passed": 1, "failed": 0, "errored": 0, "skipped": 0},
    }
    merged = dict(raw)
    for key, value in patch.items():
        if key == "benchmark":
            merged["benchmark"] = raw["benchmark"] | value
        else:
            merged[key] = value

    with pytest.raises(RecordingError, match=message):
        experiment_record_payload_from_yaml_view(merged)


async def test_record_experiment_ignores_missing_source_files(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _write_module(
        tmp_path,
        "missing_source_tasks.py",
        """
        def run(ctx, case):
            return {"success": True}
        """,
    )
    monkeypatch.syspath_prepend(str(tmp_path))
    spec = BenchmarkSpec(
        benchmark=BenchmarkInfo(id="missing-source-demo"),
        dataset=DatasetSpec(cases=[Case(id="case_1")]),
        task=TaskSpec(kind="python", target="missing_source_tasks:run"),
        variants=[Variant(id="variant_1")],
    )
    result = await run_benchmark_spec(spec, experiment_id="exp_fixed")

    record = record_experiment(
        result, tmp_path / "recorded", source_files=[tmp_path / "missing.yaml"]
    )

    assert record.file_hashes == ()


async def test_record_experiment_can_write_portable_source_paths(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _write_module(
        tmp_path,
        "portable_tasks.py",
        """
        def run(ctx, case):
            return {"success": True}
        """,
    )
    monkeypatch.syspath_prepend(str(tmp_path))
    source_dir = tmp_path / "examples" / "portable"
    source_dir.mkdir(parents=True)
    source_file = source_dir / "autobench.yaml"
    source_file.write_text("benchmark: portable\n", encoding="utf-8")
    spec = BenchmarkSpec(
        benchmark=BenchmarkInfo(id="portable"),
        dataset=DatasetSpec(cases=[Case(id="case_1")]),
        task=TaskSpec(kind="python", target="portable_tasks:run"),
        variants=[Variant(id="variant_1")],
    )

    result = await run_benchmark_spec(spec, experiment_id="exp_portable")
    record = record_experiment(
        result,
        tmp_path / "recorded",
        source_files=[source_file],
        path_root=tmp_path,
    )

    assert record.file_hashes[0].path == "examples/portable/autobench.yaml"
    assert not Path(record.environment.cwd).is_absolute()


async def test_artifact_refs_are_relative_and_artifact_payloads_are_written(
    tmp_path: Path,
    monkeypatch,
) -> None:
    _write_module(
        tmp_path,
        "artifact_tasks.py",
        """
        def run(ctx, case):
            ctx.artifact("result", {"answer": 42}, media_type="application/x-yaml")
            return {"success": True}
        """,
    )
    monkeypatch.syspath_prepend(str(tmp_path))
    spec = BenchmarkSpec(
        benchmark=BenchmarkInfo(id="artifact-demo"),
        dataset=DatasetSpec(cases=[Case(id="case_1")]),
        task=TaskSpec(kind="python", target="artifact_tasks:run"),
        variants=[Variant(id="variant_1")],
    )

    result = await run_benchmark_spec(spec, experiment_id="exp_fixed")
    record_dir = tmp_path / "runs"
    record_experiment(result, record_dir)

    run_record = load_run_record(record_dir / "cases" / "case_1" / "variant_1" / "run.yaml")
    artifact_ref = run_record.artifacts[0]

    assert isinstance(artifact_ref.value, str)
    assert not Path(artifact_ref.value).is_absolute()
    artifact_payload = load_yaml(record_dir / artifact_ref.value)
    assert artifact_payload["record"] == {"type": "artifact_payload", "version": 1}
    assert artifact_payload["artifact"]["id"] == artifact_ref.id
    assert artifact_payload["artifact"]["name"] == "result"
    assert artifact_payload["artifact"]["media_type"] == "application/x-yaml"
    assert artifact_payload["payload"] == {"answer": 42}
    artifact_meta_path = record_dir / Path(str(artifact_ref.value)).with_suffix(".meta.yaml")
    artifact_meta = load_yaml(artifact_meta_path)
    assert artifact_meta["record"] == {"type": "artifact", "version": 1}
    assert artifact_meta["artifact"]["payload"] == artifact_ref.value


def test_run_record_yaml_view_round_trips_alternate_payload_shapes() -> None:
    raw = {
        "record": {"type": "run", "version": 3},
        "run": {
            "id": "run_1",
            "experiment": "exp_1",
            "benchmark": "record-demo",
            "case": "case_1",
            "variant": "variant_1",
            "status": "passed",
            "outcome": None,
        },
        "case": {"id": "case_1"},
        "scores": [
            {
                "name": "success",
                "semantic_type": Semantic.RESULT_SUCCESS,
                "value": True,
            }
        ],
        "metrics": [
            {
                "id": "metric_1",
                "name": "quality",
                "kind": "metric",
                "semantic_type": Semantic.QUALITY_SCORE,
                "value": 1.0,
            }
        ],
        "spans": [
            {
                "id": "span_1",
                "name": "task",
                "started_at": "2026-01-01T00:00:00Z",
            }
        ],
        "artifacts": [
            {
                "id": "artifact_1",
                "name": "payload",
                "value": "artifacts/run_1/artifact_1.yaml",
            }
        ],
        "variant": {
            "factors": [
                {
                    "name": "model",
                    "value": "demo-model",
                }
            ]
        },
        "assets": [
            {
                "asset_id": "prompt.demo",
                "version": "v1",
                "content_hash": "hash1",
            }
        ],
        "errors": [
            {
                "error_type": "RuntimeError",
                "message": "recoverable",
            }
        ],
    }

    payload = run_record_payload_from_yaml_view(raw)
    record = RunRecord.model_validate(payload)
    view = run_record_to_yaml_view(record)
    round_trip = RunRecord.model_validate(run_record_payload_from_yaml_view(view))

    assert record.evaluation_status is EvaluationStatus.PASSED
    assert record.task_status is TaskStatus.PASSED
    assert record.factors[0].value == "demo-model"
    assert view["variant"]["factors"]["model"] == "demo-model"
    assert view["assets"]["prompt.demo"]["content_hash"] == "hash1"
    assert round_trip.asset_versions[0] == AssetVersion(
        asset_id="prompt.demo",
        version="v1",
        content_hash="hash1",
    )


def test_run_record_yaml_view_supports_primary_and_legacy_errors() -> None:
    raw = {
        "record": {"type": "run", "version": 3},
        "run": {
            "id": "run_1",
            "experiment": "exp_1",
            "benchmark": "record-demo",
            "case": "case_1",
            "variant": "variant_1",
            "status": "failed",
        },
        "case": {"id": "case_1"},
        "errors": {
            "primary": {
                "error_type": "ValueError",
                "message": "primary",
            }
        },
    }
    legacy_error_raw = {
        "record": {"type": "run", "version": 3},
        "run": {
            "id": "run_2",
            "experiment": "exp_1",
            "benchmark": "record-demo",
            "case": "case_2",
            "variant": "variant_1",
            "status": "failed",
        },
        "case": {"id": "case_2"},
        "error": {
            "error_type": "RuntimeError",
            "message": "legacy",
        },
    }

    primary = RunRecord.model_validate(run_record_payload_from_yaml_view(raw))
    legacy = RunRecord.model_validate(run_record_payload_from_yaml_view(legacy_error_raw))
    empty_errors = run_record_payload_from_yaml_view(raw | {"errors": {}})

    assert primary.errors[0].message == "primary"
    assert legacy.errors[0].message == "legacy"
    assert "errors" not in empty_errors


def test_run_record_yaml_view_groups_diagnostic_metrics_and_legacy_snapshot_payload() -> None:
    record = RunRecord(
        run_id="run_1",
        experiment_id="exp_1",
        benchmark_id="record-demo",
        case_id="case_1",
        variant_id="variant_1",
        status=RunStatus.PASSED,
        evaluation_status=EvaluationStatus.PASSED,
        task_status=TaskStatus.PASSED,
        case=Case(id="case_1"),
        observations=(
            Observation(
                id="diagnostic_1",
                name="debug_score",
                kind=ObservationKind.METRIC,
                semantic_type="quality.score",
                value=0.1,
                role=ObservationRole.DIAGNOSTIC,
            ),
        ),
    )

    view = run_record_to_yaml_view(record)

    assert view["metrics"]["diagnostics"]["debug_score"]["name"] == "debug_score"
    assert _benchmark_spec_snapshot_payload({"benchmark": "legacy"}) == {"benchmark": "legacy"}


@pytest.mark.parametrize(
    ("section", "value", "message"),
    [
        ("run", {"outcome": []}, "run.outcome must be a mapping"),
        ("scores", 1, "scores must be a mapping or list"),
        ("scores", {"bad": 1}, "scores.bad must be a mapping"),
        ("metrics", 1, "metrics must be a mapping or list"),
        ("metrics", {"factors": []}, "metrics.factors must be a mapping"),
        ("spans", 1, "spans must be a mapping or list"),
        ("artifacts", 1, "artifacts must be a mapping or list"),
        ("variant", {"factors": 1}, "variant.factors must be a mapping or list"),
        ("assets", 1, "assets must be a mapping or list"),
        ("errors", 1, "errors must be a mapping or list"),
    ],
)
def test_run_record_yaml_view_rejects_invalid_sections(
    section: str,
    value: Any,
    message: str,
) -> None:
    raw = {
        "record": {"type": "run", "version": 3},
        "run": {
            "id": "run_1",
            "experiment": "exp_1",
            "benchmark": "record-demo",
            "case": "case_1",
            "variant": "variant_1",
            "status": "passed",
        },
        "case": {"id": "case_1"},
    }
    if section == "run":
        raw["run"] = raw["run"] | value
    else:
        raw[section] = value

    with pytest.raises(RecordingError, match=message):
        run_record_payload_from_yaml_view(raw)


async def test_replay_loads_without_importing_task_targets_and_does_not_mutate_files(
    tmp_path: Path,
    monkeypatch,
) -> None:
    _write_module(
        tmp_path,
        "replay_tasks.py",
        """
        def run(ctx, case):
            return {"success": True, "case_id": case.id}
        """,
    )
    monkeypatch.syspath_prepend(str(tmp_path))
    spec = BenchmarkSpec(
        benchmark=BenchmarkInfo(id="replay-demo"),
        dataset=DatasetSpec(cases=[Case(id="case_1")]),
        task=TaskSpec(kind="python", target="replay_tasks:run"),
        variants=[Variant(id="variant_1")],
        scoring=[
            PassFailScorer(
                name="success",
                path="output.success",
                semantic_type=Semantic.RESULT_SUCCESS,
            )
        ],
    )
    result = await run_benchmark_spec(spec, experiment_id="exp_fixed")
    record_dir = tmp_path / "recorded"
    record_experiment(result, record_dir)
    before = {
        path.relative_to(record_dir).as_posix(): path.read_bytes()
        for path in record_dir.rglob("*.yaml")
    }
    sys.modules.pop("replay_tasks", None)
    monkeypatch.setattr(sys, "path", [item for item in sys.path if item != str(tmp_path)])

    replayed = replay_experiment(record_dir)

    after = {
        path.relative_to(record_dir).as_posix(): path.read_bytes()
        for path in record_dir.rglob("*.yaml")
    }
    assert replayed.total_count == 1
    assert replayed.runs[0].task_result.output == {"success": True, "case_id": "case_1"}
    assert replayed.runs[0].scores[0].name == "success"
    assert before == after


def test_cli_run_can_record_experiment(tmp_path: Path, monkeypatch) -> None:
    _write_module(
        tmp_path,
        "cli_record_tasks.py",
        """
        def run(ctx, case):
            return {"success": True}
        """,
    )
    monkeypatch.syspath_prepend(str(tmp_path))
    spec_path = tmp_path / "autobench.yaml"
    spec_path.write_text(
        dedent(
            """
            benchmark:
              id: cli-record
            dataset:
              cases:
                - id: case_1
            task:
              kind: python
              target: cli_record_tasks:run
            variants:
              - id: variant_1
                factors: []
            """
        ).strip()
        + "\n",
        encoding="utf-8",
    )
    record_dir = tmp_path / "recorded"
    runner = CliRunner()

    result = runner.invoke(cli, ["run", str(spec_path), "--record", str(record_dir)])

    assert result.exit_code == 0
    assert "Recorded to" in result.output
    assert (record_dir / "experiment.yaml").exists()
    assert (record_dir / "cases" / "case_1" / "variant_1" / "run.yaml").exists()


async def test_recording_is_append_only(
    tmp_path: Path,
    monkeypatch,
) -> None:
    _write_module(
        tmp_path,
        "append_tasks.py",
        """
        def run(ctx, case):
            return {"success": True}
        """,
    )
    monkeypatch.syspath_prepend(str(tmp_path))
    spec = BenchmarkSpec(
        benchmark=BenchmarkInfo(id="append-demo"),
        dataset=DatasetSpec(cases=[Case(id="case_1")]),
        task=TaskSpec(kind="python", target="append_tasks:run"),
        variants=[Variant(id="variant_1")],
    )
    result = await run_benchmark_spec(spec, experiment_id="exp_fixed")
    record_dir = tmp_path / "recorded"
    record_experiment(result, record_dir)

    with pytest.raises(RecordingError, match="already exists"):
        record_experiment(result, record_dir)


async def test_recording_preflight_does_not_write_artifacts_when_run_record_exists(
    tmp_path: Path,
    monkeypatch,
) -> None:
    _write_module(
        tmp_path,
        "preflight_tasks.py",
        """
        def run(ctx, case):
            ctx.artifact("payload", {"case": case.id})
            return {"success": True}
        """,
    )
    monkeypatch.syspath_prepend(str(tmp_path))
    spec = BenchmarkSpec(
        benchmark=BenchmarkInfo(id="preflight-demo"),
        dataset=DatasetSpec(cases=[Case(id="case_1")]),
        task=TaskSpec(kind="python", target="preflight_tasks:run"),
        variants=[Variant(id="variant_1")],
    )
    result = await run_benchmark_spec(spec, experiment_id="exp_fixed")
    record_dir = tmp_path / "recorded"
    existing_run = record_dir / "cases" / "case_1" / "variant_1" / "run.yaml"
    existing_run.parent.mkdir(parents=True)
    existing_run.write_text("already: here\n", encoding="utf-8")

    with pytest.raises(RecordingError):
        record_experiment(result, record_dir)

    assert not (record_dir / "artifacts").exists()


def test_run_record_from_result_serializes_artifacts_and_rejects_collisions(
    tmp_path: Path,
) -> None:
    class StableObject:
        def __repr__(self) -> str:
            return "stable-object"

    task_error = ErrorRecord(error_type="RuntimeError", message="boom")
    runner_error = ErrorRecord(error_type="TaskRunnerError", message="runner")
    other_error = ErrorRecord(error_type="ValueError", message="bad")
    artifact = ArtifactRef(
        id="???",
        name="payload",
        value={
            "path": tmp_path / "source.txt",
            "case": Case(id="nested"),
            "items": (1, StableObject()),
        },
    )
    run = RunResult(
        run_id="!!!",
        benchmark_id="record-demo",
        experiment_id="exp_fixed",
        case_id="case_1",
        variant_id="variant_1",
        status=RunStatus.FAILED,
        evaluation_status=EvaluationStatus.NOT_EVALUATED,
        case=Case(id="case_1"),
        task_result=TaskResult(
            output={"ok": False},
            status=TaskStatus.FAILED,
            error=runner_error,
            errors=[task_error, other_error],
            artifacts=[artifact],
        ),
        factors=[],
        scores=[],
        error=task_error,
    )
    artifacts_dir = tmp_path / "artifacts"

    record = run_record_from_result(run, artifacts_dir=artifacts_dir, root_dir=tmp_path)

    assert [error.error_type for error in record.errors] == [
        "RuntimeError",
        "TaskRunnerError",
        "ValueError",
    ]
    assert record.artifacts[0].value == "artifacts/unnamed/unnamed.yaml"
    artifact_payload = load_yaml(tmp_path / record.artifacts[0].value)
    assert artifact_payload["record"] == {"type": "artifact_payload", "version": 1}
    assert artifact_payload["artifact"]["id"] == "???"
    assert artifact_payload["artifact"]["name"] == "payload"
    assert artifact_payload["payload"]["path"].endswith("source.txt")
    assert artifact_payload["payload"]["case"]["id"] == "nested"
    assert artifact_payload["payload"]["items"] == [1, "stable-object"]
    artifact_meta = load_yaml(tmp_path / "artifacts" / "unnamed" / "unnamed.meta.yaml")
    assert artifact_meta["artifact"]["payload"] == "artifacts/unnamed/unnamed.yaml"
    with pytest.raises(RecordingError, match="Artifact already exists"):
        run_record_from_result(run, artifacts_dir=artifacts_dir, root_dir=tmp_path)


def test_run_record_model_upgrades_legacy_payload_and_rejects_non_mapping() -> None:
    legacy = RunRecord.model_validate(
        {
            "run_id": "run_1",
            "experiment_id": "exp_1",
            "benchmark_id": "demo",
            "case_id": "case_1",
            "variant_id": "variant_1",
            "status": "passed",
        }
    )

    assert legacy.task_status is TaskStatus.PASSED
    assert legacy.evaluation_status is EvaluationStatus.PASSED
    assert legacy.case.id == "case_1"

    with pytest.raises(ValidationError):
        RunRecord.model_validate("not-a-record")


def test_recording_rejects_existing_payload_path_and_writes_markdown_artifacts(
    tmp_path: Path,
) -> None:
    markdown_artifact = ArtifactRef(
        id="artifact", name="report", value="# heading", media_type="text/markdown"
    )
    text_artifact = ArtifactRef(
        id="notes",
        name="notes",
        value="plain text",
        media_type="text/plain",
    )
    json_string_artifact = ArtifactRef(
        id="json",
        name="json",
        value='{"ok": true}',
        media_type="application/json",
    )
    run = RunResult(
        run_id="run_1",
        benchmark_id="record-demo",
        experiment_id="exp_fixed",
        case_id="case_1",
        variant_id="variant_1",
        status=RunStatus.PASSED,
        evaluation_status=EvaluationStatus.PASSED,
        case=Case(id="case_1"),
        task_result=TaskResult(
            output={"ok": True},
            status=TaskStatus.PASSED,
            artifacts=[markdown_artifact, text_artifact, json_string_artifact],
        ),
    )
    artifacts_dir = tmp_path / "artifacts"

    record = run_record_from_result(run, artifacts_dir=artifacts_dir, root_dir=tmp_path)

    assert record.artifacts[0].value.endswith(".md")
    assert (tmp_path / str(record.artifacts[0].value)).read_text(encoding="utf-8") == "# heading"
    assert record.artifacts[1].value.endswith(".txt")
    assert (tmp_path / str(record.artifacts[1].value)).read_text(encoding="utf-8") == "plain text"
    assert record.artifacts[2].value.endswith(".yaml")

    payload_collision = tmp_path / "artifacts" / "run_1" / "artifact.md"
    (tmp_path / "artifacts" / "run_1" / "artifact.meta.yaml").unlink()
    payload_collision.unlink()
    payload_collision.write_text("exists\n", encoding="utf-8")
    with pytest.raises(RecordingError, match="Artifact already exists"):
        run_record_from_result(run, artifacts_dir=artifacts_dir, root_dir=tmp_path)


def test_replay_maps_all_recorded_run_statuses_to_task_statuses(tmp_path: Path) -> None:
    record_dir = tmp_path / "recorded"
    record_dir.mkdir()
    plan = BenchmarkPlan(
        benchmark_id="status-demo",
        case_count=3,
        variant_count=1,
        planned_run_count=3,
    )
    experiment = ExperimentRecord(
        experiment_id="exp_status",
        benchmark_id="status-demo",
        plan=plan,
        environment=EnvironmentMetadata(python_version="3.11", platform="test", cwd="/tmp"),
        run_paths=(
            "cases/case_failed/variant_1/run.yaml",
            "cases/case_skipped/variant_1/run.yaml",
            "cases/case_errored/variant_1/run.yaml",
        ),
        run_count=3,
        passed_count=0,
        failed_count=1,
        errored_count=1,
        skipped_count=1,
    )
    dump_yaml(experiment.model_dump(mode="json"), record_dir / "experiment.yaml")
    for status in (RunStatus.FAILED, RunStatus.SKIPPED, RunStatus.ERRORED):
        run_path = record_dir / "cases" / f"case_{status.value}" / "variant_1" / "run.yaml"
        run_path.parent.mkdir(parents=True)
        dump_yaml(
            RunRecord(
                run_id=f"run_{status.value}",
                experiment_id="exp_status",
                benchmark_id="status-demo",
                case_id=f"case_{status.value}",
                variant_id="variant_1",
                status=status,
                evaluation_status=EvaluationStatus(status.value),
                task_status=TaskStatus(status.value),
                case=Case(id=f"case_{status.value}"),
            ).model_dump(mode="json"),
            run_path,
        )

    replayed = replay_experiment(record_dir)

    assert [run.task_result.status for run in replayed.runs] == [
        TaskStatus.FAILED,
        TaskStatus.SKIPPED,
        TaskStatus.ERRORED,
    ]


def test_cli_replay_prints_recorded_summary(tmp_path: Path) -> None:
    record_dir = tmp_path / "recorded"
    (record_dir / "cases" / "case_1" / "variant_1").mkdir(parents=True)
    (record_dir / "experiment.yaml").write_text(
        dedent(
            """
            record_version: 1
            experiment_id: exp_cli
            benchmark_id: cli-replay
            plan:
              benchmark_id: cli-replay
              case_count: 1
              variant_count: 1
              planned_run_count: 1
              warnings: []
            environment:
              python_version: "3.11"
              platform: test
              cwd: /tmp
            file_hashes: []
            run_paths:
              - cases/case_1/variant_1/run.yaml
            run_count: 1
            passed_count: 1
            failed_count: 0
            errored_count: 0
            skipped_count: 0
            """
        ).strip()
        + "\n",
        encoding="utf-8",
    )
    (record_dir / "cases" / "case_1" / "variant_1" / "run.yaml").write_text(
        dedent(
            """
            record_version: 1
            run_id: run_1
            experiment_id: exp_cli
            benchmark_id: cli-replay
            case_id: case_1
            variant_id: variant_1
            status: passed
            task_output:
              success: true
            observations: []
            scores: []
            spans: []
            artifacts: []
            factors: []
            errors: []
            error: null
            """
        ).strip()
        + "\n",
        encoding="utf-8",
    )
    runner = CliRunner()

    result = runner.invoke(cli, ["replay", str(record_dir)])

    assert result.exit_code == 0
    assert "Replay Loaded" in result.output
    assert "cli-replay" in result.output
    assert "Runs" in result.output


def test_load_run_record_passes_non_mapping_payload_to_model_validation(tmp_path: Path) -> None:
    path = tmp_path / "run.yaml"
    path.write_text("- not\n- a\n- mapping\n", encoding="utf-8")

    with pytest.raises(ValidationError):
        load_run_record(path)


def test_load_experiment_record_passes_non_mapping_payload_to_model_validation(
    tmp_path: Path,
) -> None:
    record_dir = tmp_path / "recorded"
    record_dir.mkdir()
    (record_dir / "experiment.yaml").write_text("[]\n", encoding="utf-8")

    with pytest.raises(ValidationError):
        load_experiment_record(record_dir)


def _recording_spec() -> BenchmarkSpec:
    return BenchmarkSpec(
        benchmark=BenchmarkInfo(id="record-demo"),
        dataset=DatasetSpec(cases=[Case(id="case_ok"), Case(id="case_fail")]),
        task=TaskSpec(kind="python", target="record_tasks:run"),
        variants=[
            Variant(
                id="variant_1",
                factors=[
                    FactorValue(
                        name="model",
                        value="demo-model",
                        semantic_type=Semantic.LLM_MODEL_NAME,
                    )
                ],
            )
        ],
        scoring=[
            PassFailScorer(
                name="success",
                path="output.success",
                semantic_type=Semantic.RESULT_SUCCESS,
                optional=True,
            )
        ],
    )


def _write_module(tmp_path: Path, filename: str, source: str) -> None:
    (tmp_path / filename).write_text(dedent(source).strip() + "\n", encoding="utf-8")
