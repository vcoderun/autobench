from __future__ import annotations

from pathlib import Path

import pytest

from autobench import (
    Semantic,
    build_report,
    collect_benchmark_source_files,
    export_runs_csv,
    export_summary_yaml,
    load_benchmark_spec,
    record_experiment,
    replay_experiment,
    resolve_python_callable,
    run_benchmark_path,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXAMPLES_ROOT = PROJECT_ROOT / "examples"


@pytest.mark.parametrize(
    ("name", "run_count"),
    [
        ("minimal", 4),
        ("basic", 6),
        ("mid", 6),
        ("advanced", 6),
    ],
)
def test_offline_examples_execute_record_replay_report_and_export(
    name: str,
    run_count: int,
    tmp_path: Path,
) -> None:
    spec_path = EXAMPLES_ROOT / name / "autobench.yaml"
    result = run_benchmark_path(spec_path, experiment_id=f"exp_{name}")

    assert result.total_count == run_count
    assert result.failed_count == 0
    assert result.errored_count == 0
    assert result.skipped_count == 0

    record_dir = tmp_path / name
    record = record_experiment(
        result,
        record_dir,
        source_files=list(collect_benchmark_source_files(spec_path)),
        path_root=PROJECT_ROOT,
    )
    replayed = replay_experiment(record_dir)
    report = build_report(replayed)
    yaml_path = tmp_path / f"{name}-report.yaml"
    csv_path = tmp_path / f"{name}-runs.csv"
    export_summary_yaml(replayed, yaml_path)
    export_runs_csv(replayed, csv_path)

    assert record.run_count == run_count
    assert record.environment.cwd == "."
    assert record.file_hashes
    assert all(not Path(file_hash.path).is_absolute() for file_hash in record.file_hashes)
    assert replayed.total_count == run_count
    assert report.leaderboard
    assert yaml_path.is_file()
    assert csv_path.is_file()


@pytest.mark.parametrize(
    ("name", "baseline", "candidate", "score_name"),
    [
        ("basic", "route_v1", "route_v2", "route_correctness"),
        ("mid", "baseline", "candidate", "acceptable_reply"),
    ],
)
def test_candidate_variants_improve_their_objective_score(
    name: str,
    baseline: str,
    candidate: str,
    score_name: str,
) -> None:
    result = run_benchmark_path(
        EXAMPLES_ROOT / name / "autobench.yaml",
        experiment_id=f"exp_{name}_comparison",
    )
    scores = {
        variant_id: sum(
            float(score.value)
            for run in result.runs
            if run.variant_id == variant_id
            for score in run.scores
            if score.name == score_name and isinstance(score.value, int | float)
        )
        for variant_id in (baseline, candidate)
    }

    assert scores[candidate] > scores[baseline]


def test_mid_example_derives_nonzero_cost() -> None:
    result = run_benchmark_path(
        EXAMPLES_ROOT / "mid" / "autobench.yaml",
        experiment_id="exp_mid_cost",
    )

    costs = [
        float(observation.value)
        for run in result.runs
        for observation in run.task_result.observations
        if observation.semantic_type == Semantic.MONEY_COST
    ]

    assert len(costs) == result.total_count
    assert all(cost > 0.0 for cost in costs)


def test_advanced_example_derives_paired_speedup() -> None:
    result = run_benchmark_path(
        EXAMPLES_ROOT / "advanced" / "autobench.yaml",
        experiment_id="exp_advanced_speedup",
    )

    speedups = [
        float(observation.value)
        for run in result.runs
        for observation in run.task_result.observations
        if observation.semantic_type == "performance.speedup"
    ]

    assert len(speedups) == result.total_count
    assert all(speedup > 0.0 for speedup in speedups)


def test_codemode_example_is_a_resolvable_live_integration() -> None:
    spec = load_benchmark_spec(EXAMPLES_ROOT / "codemode" / "autobench.yaml")

    assert spec.task is not None
    assert len(spec.dataset.cases) == 2
    assert len(spec.variants) == 2
    assert callable(
        resolve_python_callable(spec.task.target, search_paths=spec.task.module_search_paths)
    )


def test_example_yaml_files_declare_portable_schemas() -> None:
    yaml_paths = sorted(EXAMPLES_ROOT.rglob("*.yaml"))

    assert yaml_paths
    for path in yaml_paths:
        first_line = path.read_text(encoding="utf-8").splitlines()[0]
        assert first_line.startswith("# yaml-language-server: $schema=")
        assert "/Users/" not in first_line
