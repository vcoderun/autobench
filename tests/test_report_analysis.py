from __future__ import annotations as _annotations

from hashlib import sha256
from pathlib import Path

import pytest
from pydantic import BaseModel, JsonValue

from autobench import (
    BenchmarkPlan,
    Case,
    Direction,
    ErrorRecord,
    EvaluationStatus,
    ExperimentResult,
    ExperimentStatus,
    ExperimentTermination,
    Observation,
    ObservationKind,
    ObservationRole,
    ObservationSource,
    RunResult,
    RunStatus,
    ScoreRecord,
    Semantic,
    TaskResult,
    TaskStatus,
)
from autobench.data.variants import FactorValue
from autobench.io import dump_yaml, load_yaml
from autobench.metrics.mappings import SourceSnapshot
from autobench.protocol.capture import CaptureLevel
from autobench.protocol.signals import (
    AbstractionLayer,
    CaptureMechanism,
    InstrumentationScope,
    SpanStatus,
)
from autobench.protocol.traces import Diagnostic, SpanRecord, Trace
from autobench.records.artifacts import ArtifactRef, ArtifactSource, ArtifactState
from autobench.records.models import ExperimentRecord, RunRecord
from autobench.records.storage import EnvironmentMetadata, ResolvedFileHash
from autobench.records.views import run_record_to_yaml_view
from autobench.reports.analysis import build_executive_summary
from autobench.reports.reporting import (
    LeaderboardReportSpec,
    MetricAggregation,
    ReportSpec,
    build_report,
    render_markdown_report,
)
from autobench.tracking import AssetProvenance, AssetRepresentation, AssetUse, AssetVersion
from autobench.tracking.store import AssetContentStore


def test_report_projects_design_health_metrics_and_record_identity() -> None:
    result = _partial_result()
    record = ExperimentRecord(
        experiment_id=result.experiment_id,
        benchmark_id=result.benchmark_id,
        plan=result.plan,
        environment=result.environment,
        termination=result.termination,
        spec_hash=result.spec_hash,
        file_hashes=(ResolvedFileHash(path="benchmark.yaml", sha256="a" * 64),),
        manifest_path="experiment.yaml",
        run_paths=("runs/run_a.yaml", "runs/run_b.yaml"),
        run_count=2,
        passed_count=1,
        failed_count=0,
        errored_count=1,
        skipped_count=0,
        cancelled_count=0,
    )
    report = build_report(result, experiment_record=record)

    assert report.source is not None
    assert report.source.record_version == record.record_version
    assert report.source.manifest_path == "experiment.yaml"
    assert report.source.file_hashes[0].sha256 == "a" * 64

    assert report.design is not None
    assert report.design.description == "Compare extraction models."
    assert report.design.scorer_count == 1
    assert report.design.deriver_count == 1
    assert report.design.policy_count == 1
    assert report.design.instrumentation == ("httpx", "pydantic_ai")
    assert report.design.case_tags == {"case_a": ("invoice",), "case_b": ("receipt",)}

    assert report.health is not None
    assert report.health.experiment_status == "aborted"
    assert report.health.partial is True
    assert report.health.recorded_count == 2
    assert report.health.missing_count == 1
    assert report.health.partial_run_count == 1
    assert report.health.status_by_variant == {
        "baseline": {"passed": 1},
        "candidate": {"errored": 1},
    }
    assert report.health.errors[0].error_type == "ProviderError"
    assert report.health.errors[0].count == 1

    baseline = next(row for row in report.variant_configs if row.variant_id == "baseline")
    assert baseline.factors == {"model": "model-a"}
    assert baseline.factor_details[0].semantic_type == Semantic.LLM_MODEL_NAME
    assert baseline.factor_details[0].optimize is True


def test_metric_catalog_reports_coverage_and_conflicting_metadata() -> None:
    report = build_report(
        _partial_result(),
        report_spec=ReportSpec(
            leaderboard=LeaderboardReportSpec(
                metrics=(
                    MetricAggregation(
                        name="quality",
                        semantic_type=Semantic.QUALITY_SCORE,
                        fn="mean",
                    ),
                ),
            )
        ),
    )
    catalog = {metric.semantic_type: metric for metric in report.metric_catalog}

    quality = catalog[Semantic.QUALITY_SCORE]
    assert quality.observed_count == 2
    assert quality.missing_count == 0
    assert quality.unit is None
    assert quality.direction is None
    assert quality.role == "objective"
    assert quality.aggregation == "mean"
    assert quality.sources == ("score",)

    cost = catalog[Semantic.MONEY_COST]
    assert cost.observed_count == 1
    assert cost.missing_count == 1
    assert cost.unit == "USD"
    assert cost.sources == ("instrumentation",)

    notice_codes = {notice.code for notice in report.notices}
    assert notice_codes == {"metric_direction_conflict", "metric_unit_conflict"}


def test_evaluation_summary_projects_quality_gate_scores_dimensions_and_feedback() -> None:
    class EvaluationOutput(BaseModel):
        hard_pass: bool
        score: float
        metrics: dict[str, JsonValue]
        feedback: list[str]

    result = _partial_result()
    first, second = result.runs
    first_output = EvaluationOutput(
        hard_pass=True,
        score=0.9,
        metrics={
            "semantic_score": 0.8,
            "judge_grounded": True,
            "critical_omissions": 0,
            "output_chars": 1000,
        },
        feedback=["Strong result."],
    )
    second_output = {
        "evaluation": {
            "hard_pass": False,
            "score": 0.3,
            "metrics": {
                "semantic_score": 0.2,
                "judge_grounded": False,
                "critical_omissions": 2,
                "output_chars": 1200,
                "structured": {"ignored": True},
            },
            "feedback": "Remove stale context.",
        }
    }
    result = result.model_copy(
        update={
            "runs": [
                first.model_copy(
                    update={
                        "task_result": first.task_result.model_copy(update={"output": first_output})
                    }
                ),
                second.model_copy(
                    update={
                        "task_result": second.task_result.model_copy(
                            update={"output": second_output}
                        )
                    }
                ),
            ]
        }
    )

    report = build_report(result)

    assert report.evaluation is not None
    assert report.evaluation.case_count == 2
    assert report.evaluation.evaluated_count == 2
    assert report.evaluation.passed_count == 1
    assert report.evaluation.failed_count == 1
    assert report.evaluation.pass_rate == 0.5
    assert report.evaluation.mean_score == pytest.approx(0.6)
    assert report.evaluation.median_score == pytest.approx(0.6)
    assert report.evaluation.minimum_score == 0.3
    assert report.evaluation.maximum_score == 0.9
    metrics = {metric.name: metric for metric in report.evaluation.metrics}
    assert metrics["semantic_score"].kind == "score"
    assert metrics["semantic_score"].mean == 0.5
    assert metrics["judge_grounded"].kind == "score"
    assert metrics["judge_grounded"].mean == 0.5
    assert metrics["critical_omissions"].kind == "count"
    assert metrics["critical_omissions"].total == 2
    assert metrics["output_chars"].kind == "count"
    assert "structured" not in metrics
    cases = {case.case_id: case for case in report.evaluation.cases}
    assert cases["case_a"].feedback == ("Strong result.",)
    assert cases["case_b"].feedback == ("Remove stale context.",)
    assert report.summary is not None
    assert report.summary.health.startswith("1 of 2 evaluated cases met")
    assert "Quality gate outcome" in {finding.title for finding in report.summary.findings}


def test_evaluation_summary_is_absent_without_quality_evidence() -> None:
    result = _comparison_result()
    run = result.runs[0].model_copy(
        update={
            "scores": [],
            "task_result": result.runs[0].task_result.model_copy(
                update={"output": {"answer": "not an evaluation"}}
            ),
        }
    )
    result = result.model_copy(update={"runs": [run]})

    report = build_report(result)

    assert report.evaluation is None


def test_evaluation_summary_uses_numeric_pass_scores_without_inventing_a_score() -> None:
    result = _partial_result()
    first, second = result.runs
    result = result.model_copy(
        update={
            "runs": [
                first.model_copy(
                    update={
                        "task_result": first.task_result.model_copy(
                            update={"output": {"metrics": {"assessment": "clear"}}}
                        ),
                        "scores": [
                            ScoreRecord(
                                name="hard pass",
                                semantic_type="quality.pass",
                                value=1,
                            )
                        ],
                    }
                ),
                second.model_copy(
                    update={
                        "task_result": second.task_result.model_copy(update={"output": {}}),
                        "scores": [
                            ScoreRecord(
                                name="passed",
                                semantic_type="quality.pass",
                                value=2,
                            )
                        ],
                    }
                ),
            ]
        }
    )

    report = build_report(result)

    assert report.evaluation is not None
    assert report.evaluation.evaluated_count == 1
    assert report.evaluation.passed_count == 1
    assert report.evaluation.score_count == 0
    assert report.evaluation.mean_score is None
    assert report.evaluation.metrics == ()
    assert report.summary is not None
    assert report.summary.health == "1 of 1 evaluated cases met the recorded quality gate."
    finding_titles = {finding.title for finding in report.summary.findings}
    assert "Quality gate outcome" in finding_titles
    assert "Score range" not in finding_titles


def test_comparisons_use_direction_paired_cases_and_real_factor_changes() -> None:
    report = build_report(
        _comparison_result(),
        report_spec=ReportSpec.model_validate(
            {
                "leaderboard": {
                    "metrics": [
                        {
                            "name": "quality",
                            "semantic_type": Semantic.QUALITY_SCORE,
                            "fn": "mean",
                        },
                        {
                            "name": "latency",
                            "semantic_type": Semantic.TIME_LATENCY,
                            "fn": "mean",
                        },
                    ]
                },
                "comparisons": [
                    {
                        "baseline": "baseline",
                        "candidate": "candidate",
                        "metrics": [
                            {
                                "name": "quality",
                                "semantic_type": Semantic.QUALITY_SCORE,
                                "fn": "mean",
                            },
                            {
                                "name": "latency",
                                "semantic_type": Semantic.TIME_LATENCY,
                                "fn": "mean",
                            },
                        ],
                    }
                ],
            }
        ),
    )
    comparison = report.comparisons[0]
    metrics = {metric.name: metric for metric in comparison.metric_results}

    assert comparison.paired_count == 2
    assert comparison.missing_pair_count == 0
    assert comparison.confounded is True
    assert set(comparison.factor_deltas) == {"model", "temperature"}
    assert comparison.baseline_factors == {"model": "model-a", "temperature": 0.0}
    assert comparison.candidate_factors == {"model": "model-b", "temperature": 0.2}

    quality = metrics["quality"]
    assert quality.outcome == "improved"
    assert quality.direction == "maximize"
    assert quality.delta == pytest.approx(0.15)
    assert quality.relative_delta == pytest.approx(0.15 / 0.65)
    assert (quality.wins, quality.ties, quality.losses) == (2, 0, 0)

    latency = metrics["latency"]
    assert latency.outcome == "regressed"
    assert latency.direction == "minimize"
    assert latency.delta == 20.0
    assert (latency.wins, latency.ties, latency.losses) == (1, 0, 1)

    leaderboard = {row.variant_id: row for row in report.leaderboard}
    baseline_details = {item.name: item for item in leaderboard["baseline"].metric_details}
    candidate_details = {item.name: item for item in leaderboard["candidate"].metric_details}
    assert baseline_details["latency"].best is True
    assert candidate_details["quality"].best is True
    assert [item.outcome for item in report.regressions] == ["regressed", "improved"]


def test_detailed_evidence_projects_traces_assets_artifacts_and_safe_failures(
    tmp_path: Path,
) -> None:
    result = _result_with_detailed_evidence()
    report = build_report(
        result,
        report_spec=ReportSpec.model_validate({"markdown": {"traces": {"top_slowest": 1}}}),
        experiment_root=tmp_path,
    )

    assert report.run_details[0].span_count == 2
    assert report.run_details[0].asset_count == 1
    assert report.run_details[0].artifact_count == 1
    assert report.failures[0].error_type == "ProviderError"
    assert report.failures[0].traceback is None

    assert report.traces is not None
    assert report.traces.traced_run_count == 1
    assert report.traces.partial_trace_count == 1
    assert report.traces.root_span_count == 1
    assert report.traces.orphan_span_count == 1
    assert report.traces.error_span_count == 1
    assert report.traces.spans_by_kind == {"llm": 1, "tool": 1}
    assert report.traces.diagnostics_by_code == {"accounting_mismatch": 1}
    assert report.traces.slow_spans[0].operation == "agent.run"

    assert report.assets is not None
    assert report.assets.asset_count == 1
    assert report.assets.version_count == 1
    assert report.assets.transition_count == 1
    assert report.assets.versions[0].representations == ("effective",)
    assert report.assets.versions[0].variant_ids == ("baseline",)

    assert report.artifacts is not None
    assert report.artifacts.partial_count == 1
    assert report.artifacts.artifacts[0].path is None
    assert "unsafe_artifact_path" in {notice.code for notice in report.notices}

    assert report.provenance is not None
    assert report.provenance.working_directory == "workspace"
    assert report.provenance.working_directory_redacted is True
    assert report.provenance.source_maps == ("openai:openai@2",)

    audit = build_report(
        result,
        report_spec=ReportSpec.model_validate(
            {
                "markdown": {
                    "profile": "audit",
                    "content": {"include_captured": True},
                }
            }
        ),
    )
    assert audit.failures[0].traceback == "sensitive traceback"
    assert audit.provenance is not None
    assert audit.provenance.working_directory == "/workspace"
    assert audit.provenance.working_directory_redacted is False


def test_report_analysis_handles_inferred_gaps_distinct_errors_and_safe_artifacts(
    tmp_path: Path,
) -> None:
    result = _result_with_detailed_evidence()
    run = result.runs[0]
    assert run.trace is not None
    trace = run.trace.model_copy(
        update={
            "spans": (
                run.trace.spans[0],
                run.trace.spans[1].model_copy(update={"duration_ns": None}),
            )
        }
    )
    task_error = ErrorRecord(error_type="TaskError", message="task failed")
    run_error = ErrorRecord(error_type="RunError", message="run failed")
    score_error = ErrorRecord(error_type="ScoreError", message="score failed")
    source_free = Observation(
        id="source-free",
        name="source free",
        kind=ObservationKind.METRIC,
        semantic_type="custom.source_free",
        value=1,
    )
    safe_payload = tmp_path / "artifacts" / "safe.json"
    safe_payload.parent.mkdir()
    safe_payload.write_bytes(b'{"safe":true}\n')
    safe_artifact = ArtifactRef(
        id="artifact-safe",
        name="safe",
        media_type="application/json",
        source=ArtifactSource.FILE,
        value="artifacts/safe.json",
        sha256=sha256(safe_payload.read_bytes()).hexdigest(),
        byte_count=safe_payload.stat().st_size,
        filename="artifacts/safe.json",
    )
    unhashed_payload = tmp_path / "artifacts" / "unhashed.json"
    unhashed_payload.write_bytes(b'{"recorded":true}\n')
    unhashed_artifact = ArtifactRef(
        id="artifact-unhashed",
        name="unhashed",
        media_type="application/json",
        source=ArtifactSource.FILE,
        value="artifacts/unhashed.json",
    )
    outside = tmp_path.parent / f"{tmp_path.name}-outside"
    outside.mkdir()
    (tmp_path / "linked").symlink_to(outside, target_is_directory=True)
    escaped_artifact = ArtifactRef(
        id="artifact-escaped",
        name="escaped",
        source=ArtifactSource.FILE,
        value="linked/escaped.json",
        sha256="e" * 64,
        byte_count=8,
        filename="linked/escaped.json",
    )
    missing_artifact = ArtifactRef(
        id="artifact-missing",
        name="missing",
        source=ArtifactSource.FILE,
        value="artifacts/missing.json",
        sha256="f" * 64,
        byte_count=4,
    )
    mismatched_payload = tmp_path / "artifacts" / "mismatched.json"
    mismatched_payload.write_bytes(b"different")
    mismatched_artifact = ArtifactRef(
        id="artifact-mismatched",
        name="mismatched",
        source=ArtifactSource.FILE,
        value="artifacts/mismatched.json",
        sha256="0" * 64,
        byte_count=1,
    )
    task_result = run.task_result.model_copy(
        update={
            "error": task_error,
            "errors": [],
            "observations": [*run.task_result.observations, source_free],
            "artifacts": [
                *run.task_result.artifacts,
                safe_artifact,
                unhashed_artifact,
                escaped_artifact,
                missing_artifact,
                mismatched_artifact,
            ],
        }
    )
    error_score = ScoreRecord(
        name="unavailable",
        semantic_type="quality.unavailable",
        error=score_error,
    )
    duplicated_assets = [*run.asset_versions, *run.asset_versions]
    detailed_run = run.model_copy(
        update={
            "task_result": task_result,
            "error": run_error,
            "scores": [*run.scores, error_score],
            "trace": trace,
            "asset_versions": duplicated_assets,
        }
    )
    termination = result.termination.model_copy(
        update={"missing_run_ids": (), "planned_run_ids": (), "recorded_run_ids": ()}
    )
    sparse = result.model_copy(
        update={"runs": [detailed_run, *result.runs[1:3]], "termination": termination}
    )

    report = build_report(sparse, experiment_root=tmp_path)
    unrooted = build_report(sparse)

    assert report.health is not None
    assert report.health.missing_count == 1
    assert {failure.error_type for failure in report.failures} >= {
        "TaskError",
        "RunError",
        "ScoreError",
    }
    source_free_metric = next(
        metric for metric in report.metric_catalog if metric.semantic_type == "custom.source_free"
    )
    assert source_free_metric.sources == ()
    assert report.traces is not None
    assert len(report.traces.slow_spans) == 1
    assert report.assets is not None
    assert report.assets.versions[0].representations == ("effective",)
    assert report.artifacts is not None
    artifact_paths = {
        artifact.artifact_id: artifact.path for artifact in report.artifacts.artifacts
    }
    assert artifact_paths["artifact-safe"] == "artifacts/safe.json"
    assert artifact_paths["artifact-unhashed"] == "artifacts/unhashed.json"
    assert artifact_paths["artifact-escaped"] is None
    assert artifact_paths["artifact-missing"] is None
    assert artifact_paths["artifact-mismatched"] is None
    assert sum(notice.code == "unsafe_artifact_path" for notice in report.notices) == 2
    assert {notice.code for notice in report.notices} >= {
        "missing_artifact_file",
        "artifact_integrity_mismatch",
    }
    assert unrooted.artifacts is not None
    unrooted_paths = {
        artifact.artifact_id: artifact.path for artifact in unrooted.artifacts.artifacts
    }
    assert unrooted_paths["artifact-safe"] is None
    assert unrooted_paths["artifact-escaped"] is None


def test_report_uses_recorded_artifact_paths_and_ignores_invalid_run_files(
    tmp_path: Path,
) -> None:
    result = _result_with_detailed_evidence()
    run = result.runs[0]
    payload = tmp_path / "artifacts" / "recorded.json"
    payload.parent.mkdir()
    payload.write_bytes(b'{"source":"record"}\n')
    recorded_artifact = ArtifactRef(
        id="artifact-recorded",
        name="recorded",
        media_type="application/json",
        source=ArtifactSource.FILE,
        value="artifacts/recorded.json",
    )
    run_record = RunRecord(
        run_id=run.run_id,
        experiment_id=run.experiment_id,
        benchmark_id=run.benchmark_id,
        case_id=run.case_id,
        variant_id=run.variant_id,
        status=run.status,
        evaluation_status=run.evaluation_status,
        task_status=run.task_result.status,
        case=run.case,
        artifacts=(recorded_artifact,),
    )
    runs_dir = tmp_path / "runs"
    runs_dir.mkdir()
    dump_yaml(run_record_to_yaml_view(run_record), runs_dir / "valid.yaml")
    (runs_dir / "malformed.yaml").write_text(":\n  [", encoding="utf-8")
    dump_yaml({"record": {"type": "run", "version": 6}}, runs_dir / "invalid.yaml")
    record = ExperimentRecord(
        experiment_id=result.experiment_id,
        benchmark_id=result.benchmark_id,
        plan=result.plan,
        environment=result.environment,
        termination=result.termination,
        run_paths=(
            "../outside.yaml",
            "runs/missing.yaml",
            "runs/malformed.yaml",
            "runs/invalid.yaml",
            "runs/valid.yaml",
        ),
        run_count=len(result.runs),
        passed_count=1,
        failed_count=0,
        errored_count=0,
        skipped_count=0,
    )

    report = build_report(result, experiment_record=record, experiment_root=tmp_path)

    assert report.artifacts is not None
    recorded = next(
        artifact
        for artifact in report.artifacts.artifacts
        if artifact.artifact_id == "artifact-recorded"
    )
    assert recorded.path == "artifacts/recorded.json"


def test_optional_asset_history_corruption_does_not_erase_run_evidence(tmp_path: Path) -> None:
    result = _result_with_detailed_evidence()
    valid_index: JsonValue = {"assets": {"prompt:system": {"file": "asset.yaml"}}}
    cases: tuple[tuple[JsonValue, JsonValue | None], ...] = (
        ([], None),
        ({"assets": []}, None),
        ({"assets": {"prompt:system": []}}, None),
        ({"assets": {"prompt:system": {"file": 7}}}, None),
        ({"assets": {"prompt:system": {"file": "missing.yaml"}}}, None),
        (valid_index, "malformed"),
        (valid_index, []),
        (valid_index, {"asset": [], "versions": []}),
        (valid_index, {"asset": {}, "versions": []}),
    )

    for index, (index_payload, asset_payload) in enumerate(cases):
        root = tmp_path / str(index)
        assets = root / "assets"
        assets.mkdir(parents=True)
        dump_yaml(index_payload, assets / "index.yaml")
        if asset_payload == "malformed":
            (assets / "asset.yaml").write_text(":\n  [", encoding="utf-8")
        elif asset_payload is not None:
            dump_yaml(asset_payload, assets / "asset.yaml")

        report = build_report(result, experiment_root=root)

        assert report.assets is not None
        assert report.assets.version_count == 1
        assert report.assets.versions[0].content_state == "unavailable"


def test_audit_report_loads_bounded_asset_and_run_evidence_from_the_record(
    tmp_path: Path,
) -> None:
    result = _result_with_detailed_evidence()
    run = result.runs[0]
    score = run.scores[0].model_copy(
        update={"actual_value": {"answer": "candidate"}, "expected_value": {"answer": "gold"}}
    )
    policy = Observation(
        id="policy-quality",
        name="quality-floor",
        kind=ObservationKind.EVENT,
        semantic_type="policy.result",
        value=False,
        role=ObservationRole.CONSTRAINT,
        source=ObservationSource.DERIVED,
        tags={
            "metric": Semantic.QUALITY_SCORE,
            "actual": 0.5,
            "reason": "requirement_failed",
        },
    )
    case = run.case.model_copy(
        update={
            "input": {"request": "extract a deliberately long document"},
            "expected": {"answer": "gold"},
        }
    )
    task_result = run.task_result.model_copy(
        update={
            "output": {"answer": "candidate"},
            "observations": [*run.task_result.observations, policy],
        }
    )
    result = result.model_copy(
        update={
            "runs": [
                run.model_copy(
                    update={"case": case, "task_result": task_result, "scores": [score]}
                ),
                *result.runs[1:],
            ]
        }
    )
    content_database = tmp_path / "artifacts" / "asset-content.sqlite3"
    with AssetContentStore(content_database) as store:
        store.write_content(
            asset_id="prompt:system",
            version="v2",
            content_hash="a" * 64,
            snapshot={"kind": "prompt", "raw": "Extract all requested fields precisely."},
        )
        store.write_diff(
            asset_id="prompt:system",
            version="v2",
            parent_version="v1",
            diff="--- previous\n+++ current\n+Extract all requested fields precisely.",
        )
    assets_dir = tmp_path / "assets"
    assets_dir.mkdir()
    dump_yaml(
        {"assets": {"prompt:system": {"file": "prompt-system.yaml"}}},
        assets_dir / "index.yaml",
    )
    dump_yaml(
        {
            "asset": {
                "id": "prompt:system",
                "kind": "prompt",
                "semantic": "prompt.version",
                "sensitivity": "internal",
            },
            "versions": [
                {
                    "version": "v2",
                    "parent": "v1",
                    "content_ref": {"path": "artifacts/asset-content.sqlite3"},
                    "changes": {
                        "fields": ["raw"],
                        "diff_ref": {"path": "artifacts/asset-content.sqlite3"},
                    },
                }
            ],
        },
        assets_dir / "prompt-system.yaml",
    )
    report_spec = ReportSpec.model_validate(
        {
            "markdown": {
                "profile": "audit",
                "assets": {"diffs": "full"},
                "content": {"include_captured": True},
                "limits": {"value_excerpt_chars": 32},
            }
        }
    )

    report = build_report(result, report_spec=report_spec, experiment_root=tmp_path)
    version = report.assets.versions[0] if report.assets is not None else None

    assert version is not None
    assert version.kind == "prompt"
    assert version.semantic_type == "prompt.version"
    assert version.sensitivity == "internal"
    assert version.changed_fields == ("raw",)
    assert version.content_state == "available"
    assert version.diff_state == "available"
    assert version.content_excerpt is not None and "truncated" in version.content_excerpt
    assert version.diff_excerpt is not None and "truncated" in version.diff_excerpt
    assert report.run_details[0].input_excerpt is not None
    assert "truncated" in report.run_details[0].input_excerpt
    assert report.run_details[0].score_evidence[0].name == "quality"
    assert report.policies[0].passed is False
    assert report.policies[0].metric == Semantic.QUALITY_SCORE
    assert report.summary is not None
    assert "Policy constraints failed" in {finding.title for finding in report.summary.findings}

    rendered = render_markdown_report(report)
    assert "## Captured Audit Evidence" in rendered
    assert "## Policy Outcomes" in rendered
    assert "```diff" in rendered

    no_diffs = build_report(
        result,
        report_spec=ReportSpec.model_validate({"markdown": {"assets": {"diffs": "none"}}}),
        experiment_root=tmp_path,
    )
    assert no_diffs.assets is not None
    assert no_diffs.assets.versions[0].changed_fields == ()
    assert no_diffs.assets.versions[0].diff_state == "omitted"

    asset_payload = load_yaml(assets_dir / "prompt-system.yaml")
    asset_payload["asset"]["sensitivity"] = "sensitive"
    dump_yaml(asset_payload, assets_dir / "prompt-system.yaml")
    sensitive = build_report(result, report_spec=report_spec, experiment_root=tmp_path)
    assert sensitive.assets is not None
    assert sensitive.assets.versions[0].content_state == "omitted"
    assert sensitive.assets.versions[0].diff_state == "omitted"
    assert sensitive.assets.versions[0].content_excerpt is None
    assert sensitive.assets.versions[0].diff_excerpt is None


def test_asset_evidence_states_distinguish_missing_invalid_summary_and_full_content(
    tmp_path: Path,
) -> None:
    result = _result_with_detailed_evidence()
    assets_dir = tmp_path / "assets"
    assets_dir.mkdir()
    dump_yaml(
        {"assets": {"prompt:system": {"file": "prompt-system.yaml"}}},
        assets_dir / "index.yaml",
    )
    asset_path = assets_dir / "prompt-system.yaml"

    def write_asset_refs(
        *,
        content_ref: JsonValue = None,
        diff_ref: JsonValue = None,
    ) -> None:
        changes: dict[str, JsonValue] = {"fields": ["raw"]}
        version: dict[str, JsonValue] = {
            "version": "v2",
            "parent": "v1",
            "changes": changes,
        }
        if content_ref is not None:
            version["content_ref"] = content_ref
        if diff_ref is not None:
            changes["diff_ref"] = diff_ref
        dump_yaml(
            {
                "asset": {
                    "id": "prompt:system",
                    "kind": "prompt",
                    "sensitivity": "internal",
                },
                "versions": [version],
            },
            asset_path,
        )

    write_asset_refs()
    audit = ReportSpec.model_validate(
        {
            "markdown": {
                "profile": "audit",
                "assets": {"diffs": "full"},
                "content": {"include_captured": True},
                "limits": {"value_excerpt_chars": 10_000},
            }
        }
    )

    missing = build_report(result, report_spec=audit, experiment_root=tmp_path)
    assert missing.assets is not None
    assert missing.assets.versions[0].content_state == "unavailable"
    assert missing.assets.versions[0].diff_state == "unavailable"

    write_asset_refs(
        content_ref={"path": 7},
        diff_ref={"path": "../outside.sqlite3"},
    )
    invalid_refs = build_report(result, report_spec=audit, experiment_root=tmp_path)
    assert invalid_refs.assets is not None
    assert invalid_refs.assets.versions[0].content_state == "unavailable"
    assert invalid_refs.assets.versions[0].diff_state == "unavailable"

    outside = tmp_path.parent / f"{tmp_path.name}-asset-outside"
    outside.mkdir()
    (tmp_path / "linked-assets").symlink_to(outside, target_is_directory=True)
    write_asset_refs(
        content_ref={"path": "linked-assets/content.sqlite3"},
        diff_ref={"path": "linked-assets/content.sqlite3"},
    )
    escaped_refs = build_report(result, report_spec=audit, experiment_root=tmp_path)
    assert escaped_refs.assets is not None
    assert escaped_refs.assets.versions[0].content_state == "unavailable"
    assert escaped_refs.assets.versions[0].diff_state == "unavailable"

    invalid_database = tmp_path / "artifacts" / "invalid.sqlite3"
    invalid_database.parent.mkdir()
    invalid_database.write_text("not sqlite", encoding="utf-8")
    write_asset_refs(
        content_ref={"path": "artifacts/invalid.sqlite3"},
        diff_ref={"path": "artifacts/invalid.sqlite3"},
    )
    unreadable = build_report(result, report_spec=audit, experiment_root=tmp_path)
    assert unreadable.assets is not None
    assert unreadable.assets.versions[0].content_state == "unavailable"
    assert unreadable.assets.versions[0].diff_state == "unavailable"

    database = tmp_path / "artifacts" / "content.sqlite3"
    with AssetContentStore(database) as store:
        store.write_content(
            asset_id="prompt:system",
            version="v2",
            content_hash="a" * 64,
            snapshot={"kind": "prompt", "raw": "content"},
        )
        store.write_diff(
            asset_id="prompt:system",
            version="v2",
            parent_version="v1",
            diff="+short diff",
        )
    write_asset_refs(
        content_ref={"path": "artifacts/content.sqlite3"},
        diff_ref={"path": "artifacts/content.sqlite3"},
    )
    summary = build_report(result, experiment_root=tmp_path)
    full = build_report(result, report_spec=audit, experiment_root=tmp_path)

    assert summary.assets is not None
    assert summary.assets.versions[0].diff_state == "available"
    assert summary.assets.versions[0].diff_excerpt is None
    assert full.assets is not None
    assert full.assets.versions[0].content_excerpt == '{"kind":"prompt","raw":"content"}'
    assert full.assets.versions[0].diff_excerpt == "+short diff"


def test_audit_report_marks_non_serializable_captured_values_unavailable() -> None:
    result = _result_with_detailed_evidence()
    run = result.runs[0]
    task_result = run.task_result.model_copy(update={"output": Path("opaque-output")})
    result = result.model_copy(
        update={"runs": [run.model_copy(update={"task_result": task_result}), *result.runs[1:]]}
    )
    audit = ReportSpec.model_validate(
        {
            "markdown": {
                "profile": "audit",
                "content": {"include_captured": True},
            }
        }
    )

    report = build_report(result, report_spec=audit)

    assert report.run_details[0].output_excerpt == "<unavailable:pathlib.PosixPath>"


def test_executive_findings_are_deterministic_and_evidence_linked() -> None:
    partial = build_report(_partial_result())
    assert partial.summary is not None
    assert partial.summary.leader_variant is None
    assert partial.summary.findings[0].severity == "error"
    assert {finding.title for finding in partial.summary.findings} >= {
        "Experiment evidence is incomplete",
        "High unsuccessful run rate",
        "No defensible overall winner",
    }
    assert all(finding.evidence for finding in partial.summary.findings)

    report_spec = ReportSpec.model_validate(
        {
            "leaderboard": {
                "metrics": [
                    {
                        "name": "quality",
                        "semantic_type": Semantic.QUALITY_SCORE,
                        "fn": "mean",
                    }
                ]
            },
            "comparisons": [
                {
                    "baseline": "baseline",
                    "candidate": "candidate",
                    "metrics": [
                        {
                            "name": "quality",
                            "semantic_type": Semantic.QUALITY_SCORE,
                            "fn": "mean",
                        }
                    ],
                }
            ],
        }
    )
    compared = build_report(_comparison_result(), report_spec=report_spec)
    assert compared.summary is not None
    assert compared.summary.leader_variant == "candidate"
    assert compared.summary.objective_metric == "quality"
    assert {finding.title for finding in compared.summary.findings} >= {
        "Comparison is confounded",
        "Objective leader",
        "quality improved",
    }
    assert compared.summary == build_report(_comparison_result(), report_spec=report_spec).summary

    without_health = build_executive_summary(
        compared.model_copy(update={"health": None, "summary": None})
    )
    assert without_health.health == "4 recorded runs; lifecycle metadata unavailable."

    comparison = compared.comparisons[0]
    metric = comparison.metric_results[0].model_copy(
        update={"delta": None, "outcome": "indeterminate"}
    )
    assert compared.health is not None
    no_comparison_claim = build_executive_summary(
        compared.model_copy(
            update={
                "health": compared.health.model_copy(update={"planned_count": 0}),
                "comparisons": [comparison.model_copy(update={"metric_results": (metric,)})],
                "summary": None,
            }
        )
    )
    assert "quality improved" not in {finding.title for finding in no_comparison_claim.findings}


def _partial_result() -> ExperimentResult:
    provider_error = ErrorRecord(error_type="ProviderError", message="upstream unavailable")
    runs = [
        _run(
            run_id="run_a",
            case_id="case_a",
            variant_id="baseline",
            model="model-a",
            status=RunStatus.PASSED,
            evaluation_status=EvaluationStatus.PASSED,
            quality=0.7,
            quality_unit="points",
            direction=Direction.MAXIMIZE,
            tags=["invoice"],
            cost=0.01,
        ),
        _run(
            run_id="run_b",
            case_id="case_b",
            variant_id="candidate",
            model="model-b",
            status=RunStatus.ERRORED,
            evaluation_status=EvaluationStatus.ERRORED,
            quality=0.6,
            quality_unit="ratio",
            direction=Direction.MINIMIZE,
            tags=["receipt"],
            error=provider_error,
            partial=True,
        ),
    ]
    termination = ExperimentTermination(
        status=ExperimentStatus.ABORTED,
        partial=True,
        cross_run_derivation_complete=False,
        policies_complete=False,
        planned_run_ids=("run_a", "run_b", "run_c"),
        recorded_run_ids=("run_a", "run_b"),
        missing_run_ids=("run_c",),
        error=provider_error,
    )
    return ExperimentResult(
        experiment_id="exp_report",
        benchmark_id="report-analysis",
        plan=BenchmarkPlan(
            benchmark_id="report-analysis",
            dataset_id="receipts",
            dataset_version="v2",
            dataset_hash="dataset-hash",
            case_ids=("case_a", "case_b"),
            case_count=2,
            variant_count=2,
            planned_run_count=3,
            warnings=["Matrix stopped before completion."],
        ),
        runs=runs,
        environment=EnvironmentMetadata(
            python_version="3.11.13",
            platform="test",
            cwd="/workspace",
        ),
        termination=termination,
        report_spec_data={
            "leaderboard": {
                "metrics": [
                    {
                        "name": "quality",
                        "semantic_type": Semantic.QUALITY_SCORE,
                        "fn": "mean",
                    }
                ]
            }
        },
        spec_snapshot={
            "benchmark": {
                "id": "report-analysis",
                "description": "Compare extraction models.",
            },
            "capture": {"level": "full"},
            "scoring": [{"kind": "output"}],
            "derive": [{"kind": "token_cost"}],
            "policies": [{"name": "quality-floor"}],
            "instrumentation": [
                {"kind": "pydantic_ai"},
                {"kind": "httpx"},
            ],
        },
        spec_hash="spec-hash",
    )


def _comparison_result() -> ExperimentResult:
    runs: list[RunResult] = []
    values = {
        ("case_a", "baseline"): (0.5, 100.0),
        ("case_a", "candidate"): (0.7, 90.0),
        ("case_b", "baseline"): (0.8, 200.0),
        ("case_b", "candidate"): (0.9, 250.0),
    }
    for (case_id, variant_id), (quality, latency) in values.items():
        candidate = variant_id == "candidate"
        runs.append(
            RunResult(
                run_id=f"{case_id}_{variant_id}",
                benchmark_id="comparison",
                experiment_id="exp_comparison",
                case_id=case_id,
                variant_id=variant_id,
                status=RunStatus.PASSED,
                evaluation_status=EvaluationStatus.PASSED,
                case=Case(id=case_id),
                task_result=TaskResult(output={}, status=TaskStatus.PASSED),
                scores=[
                    ScoreRecord(
                        name="quality",
                        semantic_type=Semantic.QUALITY_SCORE,
                        value=quality,
                        direction=Direction.MAXIMIZE,
                        role=ObservationRole.OBJECTIVE,
                    ),
                    ScoreRecord(
                        name="latency",
                        semantic_type=Semantic.TIME_LATENCY,
                        value=latency,
                        unit="ms",
                        direction=Direction.MINIMIZE,
                        role=ObservationRole.DIAGNOSTIC,
                    ),
                ],
                factors=[
                    FactorValue(name="model", value="model-b" if candidate else "model-a"),
                    FactorValue(name="temperature", value=0.2 if candidate else 0.0),
                ],
            )
        )
    return ExperimentResult(
        experiment_id="exp_comparison",
        benchmark_id="comparison",
        plan=BenchmarkPlan(
            benchmark_id="comparison",
            case_ids=("case_a", "case_b"),
            case_count=2,
            variant_count=2,
            planned_run_count=4,
        ),
        runs=runs,
        environment=EnvironmentMetadata(
            python_version="3.11.13",
            platform="test",
            cwd="/workspace",
        ),
    )


def _result_with_detailed_evidence() -> ExperimentResult:
    result = _comparison_result()
    run = result.runs[0]
    trace_id = "1" * 32
    root_span_id = "2" * 16
    orphan_span_id = "3" * 16
    scope = InstrumentationScope(
        instrumentor_name="autobench.pydantic_ai",
        instrumentor_version="1",
        package_name="pydantic-ai",
        package_version="2.0",
        mechanism=CaptureMechanism.PATCH,
        layer=AbstractionLayer.FRAMEWORK,
    )
    trace = Trace(
        trace_id=trace_id,
        root_span_ids=(root_span_id,),
        spans=(
            SpanRecord(
                trace_id=trace_id,
                span_id=root_span_id,
                operation="agent.run",
                kind="llm",
                scope=scope,
                capture=CaptureLevel.FULL,
                duration_ns=20_000_000,
                status=SpanStatus.ERROR,
                partial=True,
            ),
            SpanRecord(
                trace_id=trace_id,
                span_id=orphan_span_id,
                parent_span_id="4" * 16,
                operation="tool.call",
                kind="tool",
                scope=scope,
                capture=CaptureLevel.METADATA,
                duration_ns=5_000_000,
                status=SpanStatus.OK,
            ),
        ),
        diagnostics=(
            Diagnostic(
                code="accounting_mismatch",
                message="token totals disagree",
                span_id=root_span_id,
            ),
        ),
        partial=True,
    )
    version = AssetVersion(
        asset_id="prompt:system",
        version="v2",
        parent_version="v1",
        content_hash="a" * 64,
        source_hash="b" * 64,
        source_path="prompts/system.md",
        git_commit="abc123",
    )
    use = AssetUse(
        asset_id=version.asset_id,
        version=version.version,
        representation=AssetRepresentation.EFFECTIVE,
        source_locator="Agent.__init__.instructions",
        scope="agent:extractor",
        span_id=root_span_id,
        provenance=AssetProvenance(
            system="pydantic_ai",
            key="instructions",
            instrumentor="autobench.pydantic_ai",
        ),
    )
    artifact = ArtifactRef(
        id="artifact-timing",
        name="timings",
        media_type="application/json",
        source=ArtifactSource.FILE,
        state=ArtifactState.PARTIAL,
        value="../../secret.json",
        sha256="c" * 64,
        byte_count=128,
        filename="../../secret.json",
    )
    error = ErrorRecord(
        error_type="ProviderError",
        message="request failed",
        traceback="sensitive traceback",
        span_id=root_span_id,
    )
    task_result = run.task_result.model_copy(
        update={"artifacts": [artifact], "error": error, "errors": [error]}
    )
    detailed_run = run.model_copy(
        update={
            "task_result": task_result,
            "error": error,
            "trace": trace,
            "asset_versions": [version],
            "asset_uses": [use],
            "source_snapshots": (
                SourceSnapshot(
                    system="openai",
                    convention_version="1.0",
                    source_map_id="openai",
                    source_map_version=2,
                ),
            ),
        }
    )
    return result.model_copy(update={"runs": [detailed_run, *result.runs[1:]]})


def _run(
    *,
    run_id: str,
    case_id: str,
    variant_id: str,
    model: str,
    status: RunStatus,
    evaluation_status: EvaluationStatus,
    quality: float,
    quality_unit: str,
    direction: Direction,
    tags: list[str],
    cost: float | None = None,
    error: ErrorRecord | None = None,
    partial: bool = False,
) -> RunResult:
    observations = []
    if cost is not None:
        observations.append(
            Observation(
                id=f"{run_id}_cost",
                name="cost",
                kind=ObservationKind.METRIC,
                semantic_type=Semantic.MONEY_COST,
                value=cost,
                unit="USD",
                direction=Direction.MINIMIZE,
                role=ObservationRole.DIAGNOSTIC,
                source=ObservationSource.INSTRUMENTATION,
            )
        )
    task_result = TaskResult(
        output={"quality": quality},
        status=TaskStatus.ERRORED if error is not None else TaskStatus.PASSED,
        partial=partial,
        error=error,
        errors=[] if error is None else [error],
        observations=observations,
    )
    return RunResult(
        run_id=run_id,
        benchmark_id="report-analysis",
        experiment_id="exp_report",
        case_id=case_id,
        variant_id=variant_id,
        status=status,
        evaluation_status=evaluation_status,
        partial=partial,
        case=Case(id=case_id, tags=tags),
        task_result=task_result,
        scores=[
            ScoreRecord(
                name="quality",
                semantic_type=Semantic.QUALITY_SCORE,
                value=quality,
                unit=quality_unit,
                direction=direction,
                role=ObservationRole.OBJECTIVE,
            )
        ],
        factors=[
            FactorValue(
                name="model",
                value=model,
                semantic_type=Semantic.LLM_MODEL_NAME,
                optimize=True,
            )
        ],
        error=error,
    )
