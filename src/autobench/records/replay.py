from __future__ import annotations as _annotations

from pathlib import Path

from autobench.io import load_yaml
from autobench.records.recording import (
    ExperimentRecord,
    RunRecord,
    experiment_record_payload_from_yaml_view,
    run_record_payload_from_yaml_view,
)
from autobench.runtime.pipeline import ExperimentResult, RunResult
from autobench.runtime.tasks import TaskResult


def load_experiment_record(run_dir: Path) -> ExperimentRecord:
    raw = load_yaml(run_dir / "experiment.yaml")
    if isinstance(raw, dict):
        raw = experiment_record_payload_from_yaml_view(raw)
    return ExperimentRecord.model_validate(raw)


def load_run_record(path: Path) -> RunRecord:
    raw = load_yaml(path)
    if isinstance(raw, dict):
        raw = run_record_payload_from_yaml_view(raw)
    return RunRecord.model_validate(raw)


def replay_experiment(run_dir: Path) -> ExperimentResult:
    record = load_experiment_record(run_dir)
    runs = [
        _run_result_from_record(load_run_record(run_dir / run_path))
        for run_path in record.run_paths
    ]
    return ExperimentResult(
        experiment_id=record.experiment_id,
        benchmark_id=record.benchmark_id,
        plan=record.plan,
        runs=runs,
        environment=record.environment,
        report_spec_data=record.report_spec_data,
        semantic_registry=record.semantic_registry,
        spec_snapshot=record.spec_snapshot,
        spec_hash=record.spec_hash,
    )


def _run_result_from_record(record: RunRecord) -> RunResult:
    task_result = TaskResult(
        output=record.task_output,
        status=record.task_status,
        error=record.error,
        errors=list(record.errors),
        observations=list(record.observations),
        spans=list(record.spans),
        artifacts=list(record.artifacts),
    )
    return RunResult(
        run_id=record.run_id,
        benchmark_id=record.benchmark_id,
        experiment_id=record.experiment_id,
        case_id=record.case_id,
        variant_id=record.variant_id,
        status=record.status,
        evaluation_status=record.evaluation_status,
        case=record.case,
        task_result=task_result,
        scores=list(record.scores),
        factors=list(record.factors),
        asset_versions=list(record.asset_versions),
        parent_run_id=record.parent_run_id,
        error=record.error,
    )


__all__ = (
    "load_experiment_record",
    "load_run_record",
    "replay_experiment",
)
