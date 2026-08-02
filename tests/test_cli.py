from __future__ import annotations as _annotations

import importlib
import runpy
import sys
from pathlib import Path
from textwrap import dedent

import pytest
from click.testing import CliRunner

from autobench import load_experiment_record
from autobench.cli import cli


def test_cli_help_succeeds() -> None:
    runner = CliRunner()

    result = runner.invoke(cli, ["--help"])

    assert result.exit_code == 0
    assert "validate" in result.output
    assert "report" in result.output
    assert "export" in result.output
    assert "compare" in result.output


def test_cli_validate_accepts_minimal_spec(tmp_path: Path) -> None:
    path = tmp_path / "autobench.yaml"
    path.write_text(
        "\n".join(
            (
                "benchmark:",
                "  id: minimal",
                "dataset:",
                "  cases:",
                "    - id: case_1",
                "task:",
                "  kind: python",
                "  target: app.tasks.run_demo",
                "variants:",
                "  - id: variant_1",
                "    factors: []",
            )
        ),
        encoding="utf-8",
    )
    runner = CliRunner()

    result = runner.invoke(cli, ["validate", str(path)])

    assert result.exit_code == 0
    assert "Autobench Spec Valid" in result.output
    assert "minimal" in result.output
    assert "Cases" in result.output
    assert "Variants" in result.output
    assert "Planned runs" in result.output


def test_cli_validate_rejects_missing_benchmark_id(tmp_path: Path) -> None:
    path = tmp_path / "autobench.yaml"
    path.write_text("benchmark:\n  description: missing id\n", encoding="utf-8")
    runner = CliRunner()

    result = runner.invoke(cli, ["validate", str(path)])

    assert result.exit_code == 1
    assert "Spec validation failed" in result.output


def test_cli_validate_reports_yaml_location_and_plan_warnings(tmp_path: Path) -> None:
    invalid_path = tmp_path / "invalid.yaml"
    invalid_path.write_text("benchmark:\n  id: [oops\n", encoding="utf-8")
    remote_ref_path = tmp_path / "remote-ref.yaml"
    remote_ref_path.write_text(
        "benchmark:\n  id: remote-ref\ndataset:\n  source: https://example.com/cases.yaml\n",
        encoding="utf-8",
    )
    warnings_path = tmp_path / "warnings.yaml"
    warnings_path.write_text(
        dedent(
            """
            benchmark:
              id: warning-demo
              description: validates but cannot run yet
            """
        ).strip()
        + "\n",
        encoding="utf-8",
    )
    runner = CliRunner()

    invalid = runner.invoke(cli, ["validate", str(invalid_path)])
    remote_ref = runner.invoke(cli, ["validate", str(remote_ref_path)])
    warnings = runner.invoke(cli, ["validate", str(warnings_path)])

    assert invalid.exit_code == 1
    assert "Spec load failed" in invalid.output
    assert "line " in invalid.output
    assert "column " in invalid.output
    assert remote_ref.exit_code == 1
    assert "Unsupported remote reference scheme" in remote_ref.output
    assert "line " not in remote_ref.output
    assert warnings.exit_code == 0
    assert "validates but cannot run yet" in warnings.output
    assert "Warnings" in warnings.output
    assert "No cases defined." in warnings.output
    assert "No variants defined." in warnings.output
    assert "No task defined." in warnings.output


def test_cli_run_reports_spec_load_validation_and_recording_failures(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    invalid_path = tmp_path / "invalid.yaml"
    invalid_path.write_text("benchmark:\n  id: [oops\n", encoding="utf-8")
    validation_path = tmp_path / "validation.yaml"
    validation_path.write_text("benchmark:\n  description: missing id\n", encoding="utf-8")
    _write_module(
        tmp_path,
        "cli_edge_tasks.py",
        """
        def run(ctx, case):
            return {"success": True}
        """,
    )
    monkeypatch.syspath_prepend(str(tmp_path))
    runnable_path = tmp_path / "runnable.yaml"
    runnable_path.write_text(
        dedent(
            """
            benchmark:
              id: record-failure
            dataset:
              cases:
                - id: case_1
            task:
              kind: python
              target: cli_edge_tasks:run
            variants:
              - id: variant_1
            """
        ).strip()
        + "\n",
        encoding="utf-8",
    )
    record_dir = tmp_path / "recorded"
    record_dir.mkdir()
    (record_dir / "experiment.yaml").write_text("already: recorded\n", encoding="utf-8")
    runner = CliRunner()

    invalid = runner.invoke(cli, ["run", str(invalid_path)])
    validation = runner.invoke(cli, ["run", str(validation_path)])
    recording = runner.invoke(cli, ["run", str(runnable_path), "--record", str(record_dir)])

    assert invalid.exit_code == 1
    assert "Spec load failed" in invalid.output
    assert validation.exit_code == 1
    assert "Spec validation failed" in validation.output
    assert recording.exit_code == 1
    assert "Recording failed" in recording.output


def test_cli_run_records_spec_and_dataset_source_hashes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _write_module(
        tmp_path,
        "cli_source_tasks.py",
        """
        def run(ctx, case):
            return {"success": True}
        """,
    )
    monkeypatch.syspath_prepend(str(tmp_path))
    dataset_path = tmp_path / "cases.yaml"
    dataset_path.write_text("cases:\n  - id: case_1\n", encoding="utf-8")
    spec_path = tmp_path / "source-hashes.yaml"
    spec_path.write_text(
        dedent(
            """
            benchmark:
              id: source-hashes
            dataset:
              source: cases.yaml
            task:
              kind: python
              target: cli_source_tasks:run
            variants:
              - id: variant_1
            """
        ).strip()
        + "\n",
        encoding="utf-8",
    )
    record_dir = tmp_path / "recorded"
    runner = CliRunner()

    result = runner.invoke(cli, ["run", str(spec_path), "--record", str(record_dir)])
    record = load_experiment_record(record_dir)

    assert result.exit_code == 0
    assert [Path(file_hash.path).name for file_hash in record.file_hashes] == [
        "source-hashes.yaml",
        "cases.yaml",
        "cli_source_tasks.py",
    ]
    assert all(file_hash.sha256 for file_hash in record.file_hashes)


def test_main_entrypoint_returns_zero_for_valid_command(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from autobench.cli import main

    path = tmp_path / "autobench.yaml"
    path.write_text("benchmark:\n  id: entrypoint\n", encoding="utf-8")
    monkeypatch.setattr(sys, "argv", ["autobench", "validate", str(path)])

    assert main() == 0


def test_cli_run_uses_default_record_path_when_not_overridden(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _write_module(
        tmp_path,
        "cli_default_record_task.py",
        """
        def run(ctx, case):
            return {"success": True}
        """,
    )
    monkeypatch.syspath_prepend(str(tmp_path))
    spec_path = tmp_path / "default-record.yaml"
    spec_path.write_text(
        dedent(
            """
            benchmark:
              id: default-record
            dataset:
              cases:
                - id: case_1
            task:
              kind: python
              target: cli_default_record_task:run
            variants:
              - id: variant_1
            scoring:
              - kind: pass_fail
                name: success
                path: output.success
                semantic_type: result.success
            """
        ).strip()
        + "\n",
        encoding="utf-8",
    )
    runner = CliRunner()

    result = runner.invoke(cli, ["run", str(spec_path)])
    recorded_runs = list((tmp_path / ".autobench" / "default-record").glob("*/experiment.yaml"))

    assert result.exit_code == 0
    assert recorded_runs


def test_cli_report_export_and_compare_recorded_evidence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _write_module(
        tmp_path,
        "cli_reporting_tasks.py",
        """
        def run(ctx, case):
            is_candidate = ctx.variant.id == "candidate"
            is_hard = case.id == "case_hard"
            coverage = (0.75 if is_candidate else 0.5) + (0.1 if is_hard else 0.0)
            cost = 0.2 if is_candidate else 0.1
            tokens = 20 if is_candidate else 10
            ctx.metric("cost", cost, semantic_type="money.cost")
            ctx.metric("input_tokens", tokens, semantic_type="llm.tokens.input")
            return {
                "success": is_candidate or is_hard,
                "coverage": coverage,
            }
        """,
    )
    monkeypatch.syspath_prepend(str(tmp_path))
    spec_path = tmp_path / "reporting.yaml"
    spec_path.write_text(
        dedent(
            """
            benchmark:
              id: cli-reporting
            dataset:
              cases:
                - id: case_easy
                - id: case_hard
            task:
              kind: python
              target: cli_reporting_tasks:run
            variants:
              - id: baseline
                factors:
                  model.name: model-a
                  temperature: 0.1
              - id: candidate
                factors:
                  model.name: model-b
                  temperature: 0.2
            scoring:
              - kind: pass_fail
                name: success
                path: output.success
                semantic_type: result.success
              - kind: output
                name: coverage
                path: output.coverage
                semantic_type: coverage.ratio
            """
        ).strip()
        + "\n",
        encoding="utf-8",
    )
    runner = CliRunner()
    record_dir = tmp_path / "recorded"

    run = runner.invoke(cli, ["run", str(spec_path), "--record", str(record_dir)])
    report_stdout = runner.invoke(cli, ["report", str(record_dir)])
    summary_path = tmp_path / "report.yaml"
    summary_stdout = runner.invoke(
        cli,
        ["export", str(record_dir), "--format", "yaml", "--path", str(summary_path)],
    )
    csv_path = tmp_path / "runs.csv"
    csv_file = runner.invoke(
        cli,
        ["export", str(record_dir), "--format", "csv", "--path", str(csv_path)],
    )
    markdown_path = tmp_path / "report.md"
    markdown_stdout = runner.invoke(
        cli,
        ["export", str(record_dir), "--format", "markdown", "--path", str(markdown_path)],
    )
    comparison_stdout = runner.invoke(
        cli,
        ["compare", str(record_dir), "--baseline", "baseline", "--candidate", "candidate"],
    )

    assert run.exit_code == 0
    assert report_stdout.exit_code == 0
    assert "Benchmark Report" in report_stdout.output
    assert "Leaderboard" in report_stdout.output
    assert "candidate" in report_stdout.output
    assert summary_stdout.exit_code == 0
    assert "Exported YAML" in summary_stdout.output
    assert "Leaderboard" in summary_stdout.output
    summary_payload = summary_path.read_text(encoding="utf-8")
    assert "type: report" in summary_payload
    assert "benchmark: cli-reporting" in summary_payload
    assert csv_file.exit_code == 0
    assert "Exported CSV" in csv_file.output
    assert "Recorded Runs Preview" in csv_file.output
    assert csv_path.read_text(encoding="utf-8").startswith(
        "run_id,case_id,variant_id,status,success,coverage,cost,input_tokens"
    )
    assert markdown_stdout.exit_code == 0
    assert "Exported MARKDOWN" in markdown_stdout.output
    assert markdown_path.read_text(encoding="utf-8").startswith("# cli-reporting\n")
    assert comparison_stdout.exit_code == 0
    assert "Variant Comparison" in comparison_stdout.output
    assert "Factor Deltas" in comparison_stdout.output
    assert "temperature" in comparison_stdout.output


def test_dunder_main_import_and_script_execution(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = importlib.import_module("autobench.__main__")
    path = tmp_path / "autobench.yaml"
    path.write_text("benchmark:\n  id: module-entrypoint\n", encoding="utf-8")
    monkeypatch.setattr(sys, "argv", ["autobench", "validate", str(path)])
    sys.modules.pop("autobench.__main__", None)

    with pytest.raises(SystemExit) as exc_info:
        runpy.run_module("autobench.__main__", run_name="__main__")

    assert module.__all__ == ("main",)
    assert exc_info.value.code == 0


def _write_module(tmp_path: Path, filename: str, source: str) -> None:
    (tmp_path / filename).write_text(dedent(source).strip() + "\n", encoding="utf-8")
