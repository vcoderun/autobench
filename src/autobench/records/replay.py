from __future__ import annotations as _annotations

import hashlib
from collections.abc import Iterable
from pathlib import Path

from autobench.errors import AutobenchError
from autobench.evaluation.extraction import (
    ExtractionContext,
    ExtractionEvidence,
    TraceExtractor,
)
from autobench.io import load_yaml
from autobench.metrics.mappings import SourceMap, recanonicalize
from autobench.metrics.observations import Observation, ObservationKind, ObservationSource
from autobench.metrics.semantics import DEFAULT_SEMANTIC_REGISTRY, SemanticRegistry
from autobench.protocol.traces import Trace
from autobench.records.files import RecordManifest, hash_and_size, validate_manifest
from autobench.records.models import (
    RECORD_VERSION,
    ExperimentRecord,
    RecordLineage,
    ReplayKind,
    RunRecord,
)
from autobench.records.views import (
    experiment_record_payload_from_yaml_view,
    manifest_payload_from_yaml_view,
    run_record_payload_from_yaml_view,
)
from autobench.runtime.models import ExperimentResult, RunResult
from autobench.runtime.tasks import TaskResult
from autobench.runtime.traces import trace_payload_from_yaml_view


class ReplayError(AutobenchError):
    """Raised when immutable evidence cannot be replayed."""


def load_experiment_record(run_dir: Path) -> ExperimentRecord:
    raw = load_yaml(run_dir / "experiment.yaml")
    if isinstance(raw, dict):
        raw = experiment_record_payload_from_yaml_view(raw)
    record = ExperimentRecord.model_validate(raw)
    if record.manifest_path is not None:
        resolved_root = run_dir.resolve()
        manifest_path = (resolved_root / record.manifest_path).resolve()
        if not manifest_path.is_relative_to(resolved_root):
            raise ReplayError("Manifest path must stay inside the experiment directory.")
        try:
            manifest_raw = load_yaml(manifest_path)
            if isinstance(manifest_raw, dict):
                manifest_raw = manifest_payload_from_yaml_view(manifest_raw)
            manifest = RecordManifest.model_validate(manifest_raw)
            if manifest.experiment_id != record.experiment_id:
                raise ReplayError("Manifest experiment identity does not match experiment record.")
            validate_manifest(run_dir, manifest)
        except ReplayError:
            raise
        except (AutobenchError, OSError, ValueError) as exc:
            raise ReplayError(f"Invalid experiment manifest: {manifest_path}") from exc
    return record


def load_run_record(path: Path, *, root_dir: Path | None = None) -> RunRecord:
    raw = load_yaml(path)
    if isinstance(raw, dict):
        raw = run_record_payload_from_yaml_view(raw)
    record = RunRecord.model_validate(raw)
    active_root = _record_root(path) if root_dir is None else root_dir
    resolved_root = active_root.resolve()
    for artifact in record.artifacts:
        if artifact.sha256 is None or artifact.byte_count is None:
            continue
        if not isinstance(artifact.value, str):
            raise ReplayError(f"Artifact payload path must be a string: {artifact.id}")
        payload_path = (resolved_root / artifact.value).resolve()
        if not payload_path.is_relative_to(resolved_root):
            raise ReplayError(f"Artifact payload path escapes the experiment: {artifact.id}")
        if not payload_path.is_file():
            raise ReplayError(f"Artifact payload does not exist: {artifact.id}")
        digest, byte_count = hash_and_size(payload_path)
        if digest != artifact.sha256:
            raise ReplayError(f"Artifact payload hash mismatch: {artifact.id}")
        if byte_count != artifact.byte_count:
            raise ReplayError(f"Artifact payload byte count mismatch: {artifact.id}")
    if record.trace is not None or record.trace_artifact is None:
        return record
    if not isinstance(record.trace_artifact.value, str):
        raise ReplayError("Trace artifact path must be a string.")
    artifact_path = (resolved_root / record.trace_artifact.value).resolve()
    if not artifact_path.is_relative_to(resolved_root):
        raise ReplayError("Trace artifact path must stay inside the experiment directory.")
    if not artifact_path.is_file():
        raise ReplayError(f"Trace artifact does not exist: {artifact_path}")
    trace_raw = load_yaml(artifact_path)
    if not isinstance(trace_raw, dict):
        raise ReplayError(f"Trace artifact must contain a mapping: {artifact_path}")
    try:
        trace_payload, extensions = trace_payload_from_yaml_view(trace_raw)
        trace = Trace.model_validate(trace_payload)
    except (TypeError, ValueError) as exc:
        raise ReplayError(f"Invalid trace artifact: {artifact_path}") from exc
    return record.model_copy(
        update={
            "trace": trace,
            "trace_extensions": {**record.trace_extensions, **extensions},
        }
    )


def replay_experiment(run_dir: Path) -> ExperimentResult:
    record = load_experiment_record(run_dir)
    runs = [
        _run_result_from_record(load_run_record(run_dir / run_path, root_dir=run_dir))
        for run_path in record.run_paths
    ]
    return ExperimentResult(
        experiment_id=record.experiment_id,
        benchmark_id=record.benchmark_id,
        plan=record.plan,
        runs=runs,
        environment=record.environment,
        termination=record.termination,
        report_spec_data=record.report_spec_data,
        semantic_registry=record.semantic_registry,
        spec_snapshot=record.spec_snapshot,
        spec_hash=record.spec_hash,
        correlation=record.correlation,
    )


def _run_result_from_record(record: RunRecord) -> RunResult:
    task_result = TaskResult(
        output=record.task_output,
        status=record.task_status,
        partial=record.partial,
        end_reason=record.end_reason,
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
        partial=record.partial,
        end_reason=record.end_reason,
        case=record.case,
        task_result=task_result,
        scores=list(record.scores),
        factors=list(record.factors),
        asset_versions=list(record.asset_versions),
        asset_uses=list(record.asset_uses),
        parent_run_id=record.parent_run_id,
        error=record.error,
        trace=record.trace,
        source_snapshots=record.source_snapshots,
        extensions=record.extensions,
        correlation=record.correlation,
    )


def replay_extraction(
    record: RunRecord,
    extractor: TraceExtractor,
    *,
    registry: SemanticRegistry | None = None,
    run_id: str | None = None,
) -> RunRecord:
    if record.trace is None:
        raise ReplayError("Extraction replay requires a recorded ABP trace.")
    active_registry = DEFAULT_SEMANTIC_REGISTRY if registry is None else registry
    result = extractor.extract(
        record.trace,
        registry=active_registry,
        context=ExtractionContext(
            run_id=record.run_id,
            benchmark_id=record.benchmark_id,
            experiment_id=record.experiment_id,
            case_id=record.case_id,
            variant_id=record.variant_id,
        ),
    )
    extracted_ids = {observation.id for observation in result.observations}
    extracted_ids.update(
        observation_id
        for evidence in record.extractions
        if evidence.extractor == extractor.name
        for observation_id in evidence.observation_ids
    )
    observations = (
        tuple(
            observation
            for observation in record.observations
            if observation.id not in extracted_ids
        )
        + result.observations
    )
    evidence = ExtractionEvidence(
        extractor=extractor.name,
        version=extractor.version,
        observation_ids=tuple(observation.id for observation in result.observations),
        diagnostics=result.diagnostics,
        references=result.references,
    )
    previous = tuple(item for item in record.extractions if item.extractor != extractor.name)
    return record.model_copy(
        update={
            "record_version": RECORD_VERSION,
            "run_id": run_id
            or _derived_run_id(
                record.run_id, ReplayKind.EXTRACTION, extractor.name, extractor.version
            ),
            "parent_run_id": record.run_id,
            "observations": observations,
            "extractions": (*previous, evidence),
            "semantic_registry_version": active_registry.version,
            "lineage": RecordLineage(
                kind=ReplayKind.EXTRACTION,
                parent_run_id=record.run_id,
                processor=extractor.name,
                processor_version=extractor.version,
                source_record_version=record.record_version,
                source_protocol_version=record.protocol_version,
                source_semantic_registry_version=record.semantic_registry_version,
            ),
        }
    )


def replay_canonicalization(
    record: RunRecord,
    source_maps: Iterable[SourceMap],
    *,
    registry: SemanticRegistry | None = None,
    run_id: str | None = None,
) -> RunRecord:
    if not record.source_snapshots:
        raise ReplayError("Canonicalization replay requires retained source snapshots.")
    maps: dict[str, SourceMap] = {}
    for source_map in sorted(source_maps, key=lambda item: (item.id, item.version)):
        maps[source_map.id] = source_map
    missing = tuple(
        snapshot.source_map_id
        for snapshot in record.source_snapshots
        if snapshot.source_map_id not in maps
    )
    if missing:
        raise ReplayError(f"Missing source maps: {', '.join(sorted(set(missing)))}")

    active_registry = DEFAULT_SEMANTIC_REGISTRY if registry is None else registry
    results = tuple(
        recanonicalize(
            snapshot,
            maps[snapshot.source_map_id],
            registry=active_registry,
        )
        for snapshot in record.source_snapshots
    )
    observations = tuple(
        observation
        for observation in record.observations
        if observation.tags.get("replay") != ReplayKind.CANONICALIZATION
    )
    derived: list[Observation] = []
    for result_index, result in enumerate(results, start=1):
        for fact_index, fact in enumerate(result.facts, start=1):
            semantic_type = active_registry.normalize(fact.semantic_type) or fact.semantic_type
            type_info = active_registry.types.get(semantic_type)
            kind = (
                ObservationKind.METRIC
                if type_info is not None
                and type_info.value_shape in {"boolean", "integer", "number"}
                else ObservationKind.FACTOR
            )
            value = fact.value if fact.reference is None else fact.reference.model_dump(mode="json")
            derived.append(
                Observation(
                    id=f"canonical_{result_index}_{fact_index}",
                    name=fact.semantic_type,
                    kind=kind,
                    semantic_type=semantic_type,
                    value=value,
                    unit=fact.unit,
                    source=ObservationSource.IMPORTED,
                    tags={
                        "replay": ReplayKind.CANONICALIZATION,
                        "source_map_id": result.source_map_id,
                        "source_map_version": result.source_map_version,
                        "authority": fact.authority,
                    },
                    case_id=record.case_id,
                    variant_id=record.variant_id,
                )
            )

    map_versions = tuple(
        f"{source_map.id}@{source_map.version}"
        for source_map in sorted(maps.values(), key=lambda item: item.id)
    )
    return record.model_copy(
        update={
            "record_version": RECORD_VERSION,
            "run_id": run_id
            or _derived_run_id(
                record.run_id,
                ReplayKind.CANONICALIZATION,
                "source-map",
                ",".join(map_versions),
            ),
            "parent_run_id": record.run_id,
            "observations": (*observations, *derived),
            "canonicalizations": results,
            "semantic_registry_version": active_registry.version,
            "lineage": RecordLineage(
                kind=ReplayKind.CANONICALIZATION,
                parent_run_id=record.run_id,
                processor="autobench.source-map",
                processor_version="1",
                source_record_version=record.record_version,
                source_protocol_version=record.protocol_version,
                source_semantic_registry_version=record.semantic_registry_version,
                source_maps=map_versions,
            ),
        }
    )


def _record_root(path: Path) -> Path:
    for parent in path.parents:
        if (parent / "experiment.yaml").is_file():
            return parent
    return path.parent


def _derived_run_id(
    parent_run_id: str,
    kind: ReplayKind,
    processor: str,
    version: str,
) -> str:
    digest = hashlib.sha256(
        f"{parent_run_id}\0{kind}\0{processor}\0{version}".encode()
    ).hexdigest()[:12]
    return f"{parent_run_id}__{kind}_{digest}"


__all__ = (
    "ReplayError",
    "load_experiment_record",
    "load_run_record",
    "replay_canonicalization",
    "replay_extraction",
    "replay_experiment",
)
