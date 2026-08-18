from __future__ import annotations as _annotations

from pathlib import Path

import pytest

import autobench.reports.markdown as markdown_module
from autobench import ExecutionCorrelation
from autobench.reports.analysis import build_executive_summary
from autobench.reports.markdown import (
    ReportPublicationError,
    render_markdown_bundle,
    render_markdown_report,
    write_markdown_report,
)
from autobench.reports.models import (
    ArtifactInventoryReport,
    ArtifactReport,
    AssetLineageReport,
    AssetVersionReport,
    BenchmarkReport,
    CaseMatrix,
    ComparisonReport,
    ErrorSummaryReport,
    EvaluationCaseReport,
    EvaluationMetricReport,
    EvaluationSummaryReport,
    ExperimentDesignReport,
    FactorReport,
    FailureReport,
    FileHashReport,
    LeaderboardMetricReport,
    LeaderboardRow,
    MarkdownReportConfig,
    PolicyOutcomeReport,
    ProvenanceReport,
    RunDetailReport,
    RunHealthReport,
    SourceIdentityReport,
    TraceSummaryReport,
    VariantConfigRow,
)


def test_single_markdown_publication_is_deterministic_and_protects_records(
    tmp_path: Path,
) -> None:
    report = _report()
    path = tmp_path / "report.md"

    publication = write_markdown_report(report, path)

    assert publication.layout == "single"
    assert publication.requested_layout == "auto"
    assert publication.files[0].path == path
    assert publication.files[0].byte_count == len(path.read_bytes())
    assert path.read_text(encoding="utf-8") == render_markdown_report(report)
    with pytest.raises(ReportPublicationError, match="already exists"):
        write_markdown_report(report, path)
    replaced = write_markdown_report(report, path, overwrite=True)
    assert replaced.files[0].sha256 == publication.files[0].sha256

    record_root = tmp_path / "record"
    record_root.mkdir()
    with pytest.raises(ReportPublicationError, match="outside the immutable"):
        write_markdown_report(report, record_root / "report.md", immutable_root=record_root)
    with pytest.raises(ReportPublicationError, match="outside the immutable"):
        write_markdown_report(report, record_root, immutable_root=record_root)


def test_bundle_publication_separates_reader_and_audit_pages(
    tmp_path: Path,
) -> None:
    report = _report(layout="bundle")
    pages = render_markdown_bundle(report)
    destination = tmp_path / "report"

    publication = write_markdown_report(report, destination)

    assert publication.layout == "bundle"
    assert {path.split("/", maxsplit=1)[0] for path in pages} == {
        "cases",
        "index.md",
        "variants",
    }
    assert "## Report Pages" in pages["index.md"]
    assert all(
        (destination / relative_path).read_text(encoding="utf-8") == content
        for relative_path, content in pages.items()
    )
    (destination / "stale.txt").write_text("stale", encoding="utf-8")
    replaced = write_markdown_report(report, destination, overwrite=True)
    assert not (destination / "stale.txt").exists()
    assert len(replaced.files) == 3

    audit = report.model_copy(
        update={"markdown": report.markdown.model_copy(update={"profile": "audit"})}
    )
    audit_pages = render_markdown_bundle(audit)
    assert {path.split("/", maxsplit=1)[0] for path in audit_pages} == {
        "assets",
        "cases",
        "index.md",
        "runs",
        "variants",
    }


def test_artifact_links_are_rebased_for_single_and_bundle_publications(tmp_path: Path) -> None:
    record_root = tmp_path / "record"
    payload = record_root / "artifacts" / "payload.json"
    payload.parent.mkdir(parents=True)
    payload.write_text('{"ok":true}\n', encoding="utf-8")
    base = _report()
    report = base.model_copy(
        update={
            "markdown": base.markdown.model_copy(update={"profile": "audit"}),
            "artifacts": ArtifactInventoryReport(
                artifact_count=1,
                complete_count=1,
                partial_count=0,
                truncated_count=0,
                artifacts=(
                    ArtifactReport(
                        artifact_id="artifact-1",
                        name="payload",
                        run_id="run-1",
                        media_type="application/json",
                        source="file",
                        state="complete",
                        path="artifacts/payload.json",
                    ),
                ),
            ),
        }
    )
    single = tmp_path / "analysis" / "report.md"
    bundle = tmp_path / "bundle"

    write_markdown_report(report, single, immutable_root=record_root)
    write_markdown_report(report, bundle, layout="bundle", immutable_root=record_root)

    assert "../record/artifacts/payload.json" in single.read_text(encoding="utf-8")
    assert "../record/artifacts/payload.json" in (bundle / "index.md").read_text(encoding="utf-8")
    run_page = next((bundle / "runs").glob("*.md"))
    assert "../../record/artifacts/payload.json" in run_page.read_text(encoding="utf-8")
    assert "## Artifact Inventory" in (bundle / "index.md").read_text(encoding="utf-8")


def test_bundle_publication_rejects_unsafe_destinations_and_rolls_back(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    report = _report(layout="bundle")
    occupied_file = tmp_path / "occupied"
    occupied_file.write_text("original", encoding="utf-8")
    with pytest.raises(ReportPublicationError, match="already exists"):
        write_markdown_report(report, occupied_file, overwrite=True)

    occupied_directory = tmp_path / "existing"
    occupied_directory.mkdir()
    with pytest.raises(ReportPublicationError, match="already exists"):
        write_markdown_report(report, occupied_directory)

    backup = tmp_path / ".backup-conflict.replaced"
    backup.write_text("leftover", encoding="utf-8")
    with pytest.raises(ReportPublicationError, match="backup path"):
        write_markdown_report(report, tmp_path / "backup-conflict")
    assert not tuple(tmp_path.glob(".backup-conflict.finalizing-*"))

    destination = tmp_path / "rollback"
    destination.mkdir()
    original = destination / "original.md"
    original.write_text("original", encoding="utf-8")
    replace = markdown_module.os.replace

    class InterruptedReplacement:
        def __init__(self, *, fail_at: int, message: str) -> None:
            self.calls = 0
            self.fail_at = fail_at
            self.message = message

        def replace(self, source: Path, target: Path) -> None:
            self.calls += 1
            if self.calls == self.fail_at:
                raise OSError(self.message)
            replace(source, target)

    monkeypatch.setattr(
        markdown_module,
        "os",
        InterruptedReplacement(fail_at=2, message="publication interrupted"),
    )
    with pytest.raises(OSError, match="publication interrupted"):
        write_markdown_report(report, destination, overwrite=True)
    assert original.read_text(encoding="utf-8") == "original"

    new_destination = tmp_path / "new-failure"
    monkeypatch.setattr(
        markdown_module,
        "os",
        InterruptedReplacement(fail_at=1, message="first publication failed"),
    )
    with pytest.raises(OSError, match="first publication failed"):
        write_markdown_report(report, new_destination)
    assert not new_destination.exists()


def test_markdown_publication_rejects_symlinks_and_normalized_page_collisions(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    report = _report(layout="bundle")
    target = tmp_path / "target"
    target.mkdir()
    symlink = tmp_path / "report-link"
    symlink.symlink_to(target, target_is_directory=True)
    with pytest.raises(ReportPublicationError, match="cannot be a symlink"):
        write_markdown_report(report, symlink)

    monkeypatch.setattr(markdown_module, "_page_slug", lambda value: "same")
    colliding = report.model_copy(
        update={
            "markdown": report.markdown.model_copy(update={"profile": "audit"}),
            "run_details": (
                *report.run_details,
                report.run_details[0].model_copy(update={"run_id": "run-2"}),
            ),
        }
    )
    with pytest.raises(ReportPublicationError, match="paths collide"):
        render_markdown_bundle(colliding)


def test_auto_layout_uses_record_size_not_terminal_state(tmp_path: Path) -> None:
    report = _report()
    expanded = report.model_copy(
        update={
            "run_count": 51,
            "run_details": tuple(
                report.run_details[0].model_copy(update={"run_id": f"run-{index}"})
                for index in range(51)
            ),
        }
    )

    publication = write_markdown_report(expanded, tmp_path / "large")

    assert publication.requested_layout == "auto"
    assert publication.layout == "bundle"
    assert (tmp_path / "large" / "index.md").is_file()

    assert report.assets is not None
    detailed = report.model_copy(
        update={
            "markdown": report.markdown.model_copy(update={"profile": "audit"}),
            "assets": report.assets.model_copy(
                update={
                    "versions": tuple(
                        report.assets.versions[0].model_copy(update={"asset_id": f"prompt:{index}"})
                        for index in range(101)
                    )
                }
            ),
        }
    )
    detailed_publication = write_markdown_report(detailed, tmp_path / "detailed")
    assert detailed_publication.layout == "bundle"

    matrix = report.model_copy(
        update={
            "case_matrix": CaseMatrix(
                metric="quality.score",
                rows={f"case-{index}": {"baseline": 1.0} for index in range(1_001)},
            )
        }
    )
    matrix_publication = write_markdown_report(matrix, tmp_path / "matrix")
    assert matrix_publication.layout == "bundle"


def test_markdown_renderer_neutralizes_untrusted_structure() -> None:
    report = _report().model_copy(
        update={
            "benchmark_id": "unsafe | <script>alert(1)</script>\nheading",
            "variant_configs": [
                VariantConfigRow(
                    variant_id="candidate|x",
                    label="`quoted`\nline",
                    factors={"prompt": "<img src=x onerror=alert(1)>"},
                )
            ],
        }
    )

    rendered = render_markdown_report(report)

    assert "<script>" not in rendered
    assert "&lt;script&gt;" in rendered
    assert "candidate\\|x" in rendered
    assert "`quoted`<br>line" in rendered
    assert "<img" not in rendered


def test_reader_report_prioritizes_quality_outcomes_and_hides_technical_evidence() -> None:
    evaluation = EvaluationSummaryReport(
        case_count=3,
        evaluated_count=3,
        passed_count=1,
        failed_count=2,
        unevaluated_count=0,
        pass_rate=1 / 3,
        score_count=3,
        mean_score=0.5,
        median_score=0.4,
        minimum_score=0.2,
        maximum_score=0.9,
        metrics=(
            EvaluationMetricReport(
                name="semantic_score",
                label="Semantic fidelity",
                kind="score",
                sample_count=3,
                missing_count=0,
                mean=0.5,
                median=0.4,
                minimum=0.2,
                maximum=0.9,
                total=1.5,
            ),
            EvaluationMetricReport(
                name="critical_omissions",
                label="Critical omissions",
                kind="count",
                sample_count=3,
                missing_count=0,
                mean=1,
                median=1,
                minimum=0,
                maximum=2,
                total=3,
            ),
            EvaluationMetricReport(
                name="output_chars",
                label="Output chars",
                kind="count",
                sample_count=3,
                missing_count=0,
                mean=1200,
                median=1200,
                minimum=1000,
                maximum=1400,
                total=3600,
            ),
        ),
        cases=(
            EvaluationCaseReport(
                run_id="run-fail-low",
                case_id="unsafe <script>",
                variant_id="baseline",
                quality_pass=False,
                score=0.2,
                metrics={
                    "semantic_score": 0.2,
                    "critical_omissions": 2,
                    "output_chars": 1000,
                },
                feedback=("Remove <img src=x> from the answer.",),
            ),
            EvaluationCaseReport(
                run_id="run-fail-high",
                case_id="case-fail-high",
                variant_id="baseline",
                quality_pass=False,
                score=0.4,
                metrics={
                    "semantic_score": 0.4,
                    "critical_omissions": 1,
                    "output_chars": 1200,
                },
                feedback=("The quality gate was not met.",),
            ),
            EvaluationCaseReport(
                run_id="run-pass",
                case_id="case-pass",
                variant_id="baseline",
                quality_pass=True,
                score=0.9,
                metrics={
                    "semantic_score": 0.9,
                    "critical_omissions": 0,
                    "output_chars": 1400,
                },
            ),
        ),
    )
    report = _report().model_copy(update={"evaluation": evaluation})
    report = report.model_copy(update={"summary": build_executive_summary(report)})

    rendered = render_markdown_report(report)
    summary = render_markdown_report(
        report.model_copy(
            update={"markdown": report.markdown.model_copy(update={"profile": "summary"})}
        )
    )
    audit = render_markdown_report(
        report.model_copy(
            update={"markdown": report.markdown.model_copy(update={"profile": "audit"})}
        )
    )

    assert "1 of 3 evaluated cases met the recorded quality gate" in rendered
    assert "## Benchmark Outcome" in rendered
    assert "Quality gate</text>" in rendered
    assert "Case score ranking" in rendered
    assert "Average quality by dimension" in rendered
    assert "Critical omissions: 2" in rendered
    assert "Output chars" not in rendered
    assert "Remove &lt;img src=x&gt; from the answer." in rendered
    assert "unsafe &lt;script&gt;" in rendered
    assert "<script>" not in rendered
    assert "## Asset Lineage" not in rendered
    assert "## Technical Evidence" not in rendered
    assert "## Where Quality Broke Down" not in summary
    assert "## Asset Lineage" in audit
    assert "## Technical Evidence" in audit


def test_reader_report_handles_sparse_quality_evidence_and_bundle_drill_downs() -> None:
    cases = (
        EvaluationCaseReport(
            run_id="run-1",
            case_id="case-1",
            variant_id="baseline",
            quality_pass=False,
            score=0.2,
            metrics={"assessment": "review required"},
            feedback=("Revise the unsupported conclusion.",),
        ),
        EvaluationCaseReport(
            run_id="run-2",
            case_id="case-2",
            variant_id="baseline",
            quality_pass=True,
            score=0.8,
        ),
    )
    evaluation = EvaluationSummaryReport(
        case_count=2,
        evaluated_count=2,
        passed_count=1,
        failed_count=1,
        unevaluated_count=0,
        pass_rate=0.5,
        score_count=2,
        mean_score=0.5,
        median_score=0.5,
        minimum_score=0.2,
        maximum_score=0.8,
        metrics=(
            EvaluationMetricReport(
                name="assessment",
                label="Assessment",
                kind="value",
                sample_count=0,
                missing_count=2,
                mean=0,
                median=0,
                minimum=0,
                maximum=0,
                total=0,
            ),
            EvaluationMetricReport(
                name="unbounded_score",
                label="Unbounded score",
                kind="score",
                sample_count=2,
                missing_count=0,
                mean=1.2,
                median=1.2,
                minimum=1.1,
                maximum=1.3,
                total=2.4,
            ),
        ),
        cases=cases,
    )
    base = _report(layout="bundle")
    report = base.model_copy(
        update={
            "design": ExperimentDesignReport(
                dataset_id="support-cases",
                case_count=2,
                variant_count=1,
                planned_run_count=2,
            ),
            "evaluation": evaluation,
            "run_count": 3,
            "run_details": (
                base.run_details[0],
                base.run_details[0].model_copy(update={"run_id": "run-2", "case_id": "case-2"}),
                base.run_details[0].model_copy(
                    update={"run_id": "run-3", "case_id": "case-without-evaluation"}
                ),
            ),
        }
    )

    rendered = render_markdown_report(report)
    pages = render_markdown_bundle(report)
    case_one = next(page for path, page in pages.items() if "case-1" in path)
    case_two = next(page for path, page in pages.items() if "case-2" in path)
    case_without_evaluation = next(
        page for path, page in pages.items() if "case-without-evaluation" in path
    )
    variant = next(page for path, page in pages.items() if path.startswith("variants/"))

    assert "Quality gate not met" in rendered
    assert "Average quality by dimension" not in rendered
    assert "| Dataset | support-cases |" in rendered
    assert "## Evaluator Feedback" in case_one
    assert "Revise the unsupported conclusion." in case_one
    assert "## Evaluator Feedback" not in case_two
    assert "| Variant | Result | Score |" not in case_without_evaluation
    assert "## Case Outcomes" in variant

    many_cases = tuple(
        EvaluationCaseReport(
            run_id=f"run-{index}",
            case_id=f"case-{index:02d}",
            variant_id="baseline",
            quality_pass=index % 2 == 0,
            score=index / 20,
        )
        for index in range(21)
    )
    many = evaluation.model_copy(
        update={
            "case_count": 21,
            "evaluated_count": 21,
            "passed_count": 11,
            "failed_count": 10,
            "pass_rate": 11 / 21,
            "score_count": 21,
            "cases": many_cases,
        }
    )
    summary_report = report.model_copy(
        update={
            "evaluation": many,
            "markdown": report.markdown.model_copy(update={"profile": "summary"}),
        }
    )
    assert "Showing 20 of 21 case results" in render_markdown_report(summary_report)

    empty = evaluation.model_copy(
        update={
            "case_count": 0,
            "evaluated_count": 0,
            "passed_count": 0,
            "failed_count": 0,
            "pass_rate": None,
            "score_count": 0,
            "cases": (),
        }
    )
    assert "### Case Results" not in render_markdown_report(
        report.model_copy(update={"evaluation": empty})
    )


def test_full_and_audit_reports_render_bounded_evidence_and_numeric_edges() -> None:
    report = _report()
    detailed = report.model_copy(
        update={
            "markdown": MarkdownReportConfig.model_validate(
                {
                    "profile": "audit",
                    "limits": {"table_rows": 1, "run_details": 1, "failure_details": 1},
                }
            ),
            "health": RunHealthReport(
                experiment_status="aborted",
                partial=True,
                planned_count=2,
                recorded_count=1,
                missing_count=1,
                partial_run_count=1,
                status_by_variant={"baseline": {"errored": 1}},
                missing_run_ids=("run-missing",),
                cross_run_derivation_complete=False,
                policies_complete=False,
                errors=(
                    ErrorSummaryReport(
                        error_type="ValueError",
                        count=1,
                        run_ids=("run-1",),
                    ),
                ),
            ),
            "design": ExperimentDesignReport(
                description="A bounded report rendering exercise.",
                case_count=2,
                case_tags={"case-1": ("critical",), "case-2": ()},
                variant_count=1,
                planned_run_count=2,
                capture_enabled=True,
                warnings=("Case coverage is intentionally partial.",),
            ),
            "case_matrix": CaseMatrix(
                metric="quality.score",
                rows={"case-1": {"baseline": 0.75}, "case-2": {"baseline": None}},
            ),
            "run_details": (
                report.run_details[0],
                report.run_details[0].model_copy(update={"run_id": "run-2"}),
            ),
            "failures": (
                FailureReport(
                    run_id="run-1",
                    case_id="case-1",
                    variant_id="baseline",
                    stage="task",
                    error_type="ValueError",
                    message="bad | value",
                    traceback="trace ``` <script>",
                ),
                FailureReport(
                    run_id="run-2",
                    case_id="case-2",
                    variant_id="baseline",
                    stage="score",
                    error_type="ScoreError",
                    message="missing",
                ),
            ),
            "source": SourceIdentityReport(
                benchmark_id="reporting-demo",
                experiment_id="exp-reporting",
                run_paths=("runs/run-1.yaml",),
                file_hashes=(FileHashReport(path="experiment.yaml", sha256="a" * 64),),
                correlation=ExecutionCorrelation(
                    group_id="group-1",
                    attempt=2,
                    phase="validation",
                ),
            ),
            "policies": (
                PolicyOutcomeReport(
                    name="quality-floor",
                    run_id="run-1",
                    case_id="case-1",
                    variant_id="baseline",
                    passed=False,
                    metric="quality.score",
                    actual=0.5,
                    reason="below_floor",
                ),
                PolicyOutcomeReport(
                    name="cost-ceiling",
                    run_id="run-2",
                    case_id="case-2",
                    variant_id="baseline",
                    passed=True,
                ),
            ),
            "traces": TraceSummaryReport(
                traced_run_count=1,
                untraced_run_count=0,
                complete_trace_count=0,
                partial_trace_count=1,
                span_count=2,
                root_span_count=1,
                orphan_span_count=1,
                error_span_count=1,
                cancelled_span_count=0,
                spans_by_kind={"llm": 2},
                spans_by_instrumentor={"autobench.pydantic_ai": 2},
                diagnostics_by_code={"accounting_mismatch": 1},
            ),
            "provenance": None,
        }
    )

    rendered = render_markdown_report(detailed)
    source_only = render_markdown_report(
        detailed.model_copy(update={"source": None, "provenance": _provenance()})
    )
    assert detailed.design is not None
    design_without_case_tags = render_markdown_report(
        detailed.model_copy(update={"design": detailed.design.model_copy(update={"case_tags": {}})})
    )
    no_variants = render_markdown_report(
        report.model_copy(update={"variant_configs": [], "run_details": (), "failures": ()})
    )
    empty_audit = render_markdown_report(
        report.model_copy(
            update={
                "markdown": report.markdown.model_copy(update={"profile": "audit"}),
                "run_details": (),
                "failures": (),
            }
        )
    )
    legacy_comparison = ComparisonReport(
        baseline="baseline",
        candidate="candidate",
        run_count=1,
        metric_deltas={"quality": {"baseline": 0.5, "candidate": 0.75, "delta": 0.25}},
    )
    legacy = render_markdown_report(report.model_copy(update={"comparisons": [legacy_comparison]}))
    bundle_edges = render_markdown_bundle(
        detailed.model_copy(
            update={
                "case_matrix": CaseMatrix(metric="quality.score"),
                "variant_configs": [
                    *detailed.variant_configs,
                    VariantConfigRow(variant_id="unobserved"),
                ],
                "markdown": detailed.markdown.model_copy(update={"layout": "bundle"}),
            }
        )
    )
    full_failure = render_markdown_report(
        detailed.model_copy(
            update={
                "markdown": detailed.markdown.model_copy(
                    update={
                        "profile": "full",
                        "limits": detailed.markdown.limits.model_copy(
                            update={"failure_details": 2}
                        ),
                    }
                )
            }
        )
    )
    full_failure_bundle = render_markdown_bundle(
        detailed.model_copy(
            update={
                "markdown": detailed.markdown.model_copy(
                    update={"profile": "full", "layout": "bundle"}
                )
            }
        )
    )
    audit_without_traceback = render_markdown_report(
        report.model_copy(
            update={
                "markdown": report.markdown.model_copy(update={"profile": "audit"}),
                "failures": (
                    FailureReport(
                        run_id="run-1",
                        case_id="case-1",
                        variant_id="baseline",
                        stage="task",
                        error_type="ValueError",
                        message="no traceback captured",
                    ),
                ),
            }
        )
    )
    empty_trace = render_markdown_report(
        report.model_copy(
            update={
                "markdown": report.markdown.model_copy(update={"profile": "audit"}),
                "traces": TraceSummaryReport(
                    traced_run_count=1,
                    untraced_run_count=0,
                    complete_trace_count=1,
                    partial_trace_count=0,
                    span_count=0,
                    root_span_count=0,
                    orphan_span_count=0,
                    error_span_count=0,
                    cancelled_span_count=0,
                ),
            }
        )
    )
    kind_only_trace = render_markdown_report(
        report.model_copy(
            update={
                "markdown": report.markdown.model_copy(update={"profile": "audit"}),
                "traces": TraceSummaryReport(
                    traced_run_count=1,
                    untraced_run_count=0,
                    complete_trace_count=1,
                    partial_trace_count=0,
                    span_count=1,
                    root_span_count=1,
                    orphan_span_count=0,
                    error_span_count=0,
                    cancelled_span_count=0,
                    spans_by_kind={"task": 1},
                ),
            }
        )
    )
    instrumentor_only_trace = render_markdown_report(
        report.model_copy(
            update={
                "markdown": report.markdown.model_copy(update={"profile": "audit"}),
                "traces": TraceSummaryReport(
                    traced_run_count=1,
                    untraced_run_count=0,
                    complete_trace_count=1,
                    partial_trace_count=0,
                    span_count=1,
                    root_span_count=1,
                    orphan_span_count=0,
                    error_span_count=0,
                    cancelled_span_count=0,
                    spans_by_instrumentor={"autobench.custom": 1},
                ),
            }
        )
    )
    summary_without_limitations = render_markdown_report(
        report.model_copy(
            update={
                "markdown": report.markdown.model_copy(update={"profile": "summary"}),
                "health": None,
                "summary": None,
                "notices": (),
            }
        )
    )
    assert detailed.health is not None
    health_without_variant_status = render_markdown_report(
        detailed.model_copy(
            update={
                "health": detailed.health.model_copy(update={"status_by_variant": {}}),
            }
        )
    )
    unrelated_artifact_bundle = render_markdown_bundle(
        report.model_copy(
            update={
                "markdown": report.markdown.model_copy(
                    update={"profile": "audit", "layout": "bundle"}
                ),
                "artifacts": ArtifactInventoryReport(
                    artifact_count=1,
                    complete_count=1,
                    partial_count=0,
                    truncated_count=0,
                    artifacts=(
                        ArtifactReport(
                            artifact_id="artifact-other",
                            name="other",
                            run_id="run-other",
                            media_type="text/plain",
                            source="value",
                            state="complete",
                        ),
                    ),
                ),
            }
        )
    )
    unrelated_artifact_run_page = next(
        content for path, content in unrelated_artifact_bundle.items() if path.startswith("runs/")
    )

    assert "Truncated to 1 case rows" in rendered
    assert "Truncated to 1 run rows" in rendered
    assert "trace ` ` ` &lt;script&gt;" in rendered
    assert "1 planned run is missing" in rendered
    assert "run-missing" in rendered
    assert "### Grouped Errors" in rendered
    assert "### Status By Variant" in rendered
    assert "### Case Tags" in rendered
    assert "### Design Warnings" in rendered
    assert "### Instrumentation Coverage" in rendered
    assert "### Trace Diagnostics" in rendered
    assert "Truncated to 1 policy outcome rows" in rendered
    assert "### Recorded File Hashes" in rendered
    assert "### Recorded Run Files" in rendered
    assert "### Execution Correlation" in rendered
    assert "The experiment is partial" in rendered
    assert "semantic registry" in source_only
    assert "### Case Tags" not in design_without_case_tags
    assert "### Design Warnings" in design_without_case_tags
    assert "## Variant Configurations" not in no_variants
    assert "## Run Results" not in empty_audit
    assert "| quality | 0.5 | 0.75 | 0.25 |" in legacy
    assert (
        "## Failures"
        in bundle_edges[next(path for path in bundle_edges if path.startswith("runs/"))]
    )
    unobserved_page = next(
        content
        for path, content in bundle_edges.items()
        if path.startswith("variants/") and "# Variant: unobserved" in content
    )
    assert "| run | case | status | partial |" not in unobserved_page
    assert "<details>" not in full_failure
    assert all("traceback</summary>" not in page for page in full_failure_bundle.values())
    assert "no traceback captured" in audit_without_traceback
    assert "<details>" not in audit_without_traceback
    assert "### Span Composition" not in empty_trace
    assert "### Slowest Spans" not in empty_trace
    assert "### Span Composition" in kind_only_trace
    assert "### Instrumentation Coverage" not in kind_only_trace
    assert "### Span Composition" not in instrumentor_only_trace
    assert "### Instrumentation Coverage" in instrumentor_only_trace
    assert "## Experiment Design" not in summary_without_limitations
    assert "## Limitations And Missing Evidence" not in summary_without_limitations
    assert "### Status By Variant" not in health_without_variant_status
    assert "### Grouped Errors" in health_without_variant_status
    assert "## Artifacts" not in unrelated_artifact_run_page
    assert markdown_module._format_value(False) == "false"
    assert markdown_module._format_value(float("inf")) == "unsupported"
    assert markdown_module._format_value(0.000000001, unit="USD") == "0.000000001"
    assert markdown_module._format_value(0.0, unit="USD") == "0"
    assert markdown_module._format_value(12.34567, unit="%") == "12.3457%"
    assert markdown_module._format_value({"value": 1}) == '{"value":1}'
    assert markdown_module._duration(999) == "999 ns"
    assert markdown_module._duration(1_000) == "1.000 us"
    assert markdown_module._duration(1_000_000) == "1.000 ms"
    assert markdown_module._duration(1_000_000_000) == "1.000 s"


def _report(*, layout: str = "auto") -> BenchmarkReport:
    return BenchmarkReport(
        benchmark_id="reporting-demo",
        experiment_id="exp-reporting",
        run_count=1,
        markdown=MarkdownReportConfig.model_validate({"layout": layout}),
        variant_configs=[
            VariantConfigRow(
                variant_id="baseline",
                label="Baseline",
                factors={"model": "model-a"},
                factor_details=(
                    FactorReport(
                        name="model",
                        value="model-a",
                        semantic_type="llm.model.name",
                        optimize=True,
                    ),
                ),
            )
        ],
        leaderboard=[
            LeaderboardRow(
                variant_id="baseline",
                run_count=1,
                metrics={"quality": 0.75},
                metric_details=(
                    LeaderboardMetricReport(
                        name="quality",
                        semantic_type="quality.score",
                        value=0.75,
                        sample_count=1,
                        missing_count=0,
                        direction="maximize",
                    ),
                ),
            )
        ],
        run_details=(
            RunDetailReport(
                run_id="run-1",
                case_id="case-1",
                variant_id="baseline",
                status="passed",
                evaluation_status="passed",
                partial=False,
                end_reason="completed",
                metrics={"quality (quality.score)": 0.75},
            ),
        ),
        case_matrix=CaseMatrix(
            metric="quality.score",
            rows={"case-1": {"baseline": 0.75}},
        ),
        assets=AssetLineageReport(
            asset_count=1,
            version_count=1,
            transition_count=0,
            versions=(
                AssetVersionReport(
                    asset_id="prompt:system",
                    version="v1",
                    content_hash="a" * 64,
                    representations=("effective",),
                    scopes=("agent:router",),
                    run_ids=("run-1",),
                    variant_ids=("baseline",),
                ),
            ),
        ),
    )


def _provenance() -> ProvenanceReport:
    return ProvenanceReport(
        python_version="3.11",
        platform="test",
        working_directory="workspace",
        working_directory_redacted=True,
        semantic_registry_version=1,
    )
