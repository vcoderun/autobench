from __future__ import annotations

from hashlib import sha256
from pathlib import Path

import pytest
from pydantic import ValidationError

from autobench import (
    DEFAULT_SEMANTIC_REGISTRY,
    RECORD_VERSION,
    ArtifactRef,
    BenchmarkPlan,
    CanonicalizationResult,
    Case,
    EnvironmentMetadata,
    EvaluationStatus,
    ExperimentResult,
    ExtractionContext,
    ExtractionResult,
    Observation,
    ObservationKind,
    RecordingError,
    RecordLineage,
    ReplayError,
    ReplayKind,
    RunContext,
    RunRecord,
    RunResult,
    RunStatus,
    Semantic,
    SemanticRegistry,
    SignalExtractor,
    SourceData,
    SourceMap,
    SourceSelector,
    SourceSnapshot,
    TaskResult,
    TaskStatus,
    Variant,
    __version__,
    canonicalize,
    load_run_record,
    record_experiment,
    replay_canonicalization,
    replay_extraction,
    run_record_from_result,
)
from autobench.io import dump_yaml
from autobench.metrics.mappings import RenameRule
from autobench.protocol import (
    CaptureLevel,
    CapturePolicy,
    CaptureSession,
    Event,
    Reference,
    ReferenceKind,
)
from autobench.protocol.traces import Diagnostic, Trace
from autobench.protocol.values import EvidenceRef
from autobench.records.recording import (
    TRACE_ARTIFACT_MEDIA_TYPE,
    run_record_payload_from_yaml_view,
    run_record_to_yaml_view,
)
from autobench.runtime.traces import (
    trace_payload_from_yaml_view,
    trace_to_yaml_view,
    trace_yaml_schema,
)


def test_inline_trace_round_trips_with_protocol_scope_and_extensions(tmp_path: Path) -> None:
    result = _experiment_result()
    output_dir = tmp_path / "record"

    record_experiment(result, output_dir)
    run_path = output_dir / "cases" / "case_1" / "variant_1" / "run.yaml"
    original_bytes = run_path.read_bytes()
    record = load_run_record(run_path)
    replayed = load_run_record(run_path)

    assert record.trace == result.runs[0].trace
    assert record.protocol_version == 1
    assert record.semantic_registry_version == DEFAULT_SEMANTIC_REGISTRY.version
    assert record.trace_artifact is None
    assert record.trace is not None
    assert record.trace.spans[0].scope.instrumentor_name == "autobench.manual"
    assert record.trace.signals
    assert replayed == record
    assert run_path.read_bytes() == original_bytes

    view = run_record_to_yaml_view(
        record.model_copy(
            update={
                "extensions": {"future_run_field": {"enabled": True}},
                "trace_extensions": {"future_trace_field": [1, 2]},
            }
        )
    )
    restored = RunRecord.model_validate(run_record_payload_from_yaml_view(view))
    assert restored.extensions == {"future_run_field": {"enabled": True}}
    assert restored.trace_extensions == {"future_trace_field": [1, 2]}


def test_large_trace_uses_schema_header_artifact_and_loads_offline(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("AUTOBENCH_HOME", str(tmp_path / "home"))
    result = _experiment_result(output={"content": "x" * 2_000})
    output_dir = tmp_path / "record"

    record_experiment(result, output_dir, trace_inline_limit_bytes=1)
    run_path = output_dir / "cases" / "case_1" / "variant_1" / "run.yaml"
    raw_run = run_path.read_text(encoding="utf-8")
    trace_path = output_dir / "artifacts" / result.runs[0].run_id / "trace.yaml"
    record = load_run_record(run_path)

    assert "artifact:" in raw_run
    assert trace_path.read_text(encoding="utf-8").startswith("# yaml-language-server: $schema=")
    assert (tmp_path / "home" / __version__ / "schemas" / "trace_schema.json").is_file()
    assert record.trace == result.runs[0].trace
    assert record.trace_artifact is not None
    assert record.trace_artifact.media_type == TRACE_ARTIFACT_MEDIA_TYPE
    assert record.trace_artifact.value == f"artifacts/{result.runs[0].run_id}/trace.yaml"


def test_signal_extraction_replay_replaces_matching_observations_and_tracks_lineage() -> None:
    original = _run_record()
    original_payload = original.model_dump(mode="json")

    derived = replay_extraction(original, SignalExtractor())
    repeated = replay_extraction(derived, SignalExtractor(), run_id="explicit_derived")

    assert original.model_dump(mode="json") == original_payload
    assert derived.run_id.startswith(f"{original.run_id}__extraction_")
    assert derived.parent_run_id == original.run_id
    assert derived.lineage == RecordLineage(
        kind=ReplayKind.EXTRACTION,
        parent_run_id=original.run_id,
        processor="abp.signals",
        processor_version="2",
        source_record_version=original.record_version,
        source_protocol_version=1,
        source_semantic_registry_version=1,
    )
    assert {observation.name for observation in derived.observations} >= {
        "quality",
        "model",
        "diagnostic",
    }
    assert len({observation.id for observation in repeated.observations}) == len(
        repeated.observations
    )
    assert repeated.run_id == "explicit_derived"
    assert repeated.extractions[0].extractor == "abp.signals"
    assert repeated.extractions[0].version == "2"
    assert (
        RunRecord.model_validate(
            run_record_payload_from_yaml_view(run_record_to_yaml_view(repeated))
        )
        == repeated
    )


def test_custom_extractor_persists_diagnostics_references_and_registry_version() -> None:
    class CustomExtractor:
        name = "custom"
        version = "2"

        def extract(
            self,
            trace: Trace,
            *,
            registry: SemanticRegistry,
            context: ExtractionContext,
        ) -> ExtractionResult:
            assert trace.trace_id
            assert registry.version == 7
            assert context.case_id == "case_1"
            return ExtractionResult(
                diagnostics=(Diagnostic(code="custom", message="retained"),),
                references=(EvidenceRef(kind=ReferenceKind.CUSTOM, id="ref_1"),),
            )

    registry = DEFAULT_SEMANTIC_REGISTRY.model_copy(update={"version": 7})
    derived = replay_extraction(_run_record(), CustomExtractor(), registry=registry)

    assert derived.semantic_registry_version == 7
    assert derived.extractions[0].diagnostics[0].code == "custom"
    assert derived.extractions[0].references[0].id == "ref_1"


def test_new_extractor_version_replaces_prior_version_evidence() -> None:
    class VersionedExtractor:
        name = "versioned"

        def __init__(self, version: str, observation_id: str) -> None:
            self.version = version
            self.observation_id = observation_id

        def extract(
            self,
            trace: Trace,
            *,
            registry: SemanticRegistry,
            context: ExtractionContext,
        ) -> ExtractionResult:
            assert trace.trace_id
            assert registry.version
            return ExtractionResult(
                observations=(
                    Observation(
                        id=self.observation_id,
                        name="versioned",
                        kind=ObservationKind.METRIC,
                        semantic_type=Semantic.OPERATION_COUNT,
                        value=int(self.version),
                        case_id=context.case_id,
                        variant_id=context.variant_id,
                    ),
                )
            )

    first = replay_extraction(_run_record(), VersionedExtractor("1", "versioned_v1"))
    second = replay_extraction(first, VersionedExtractor("2", "versioned_v2"))

    assert "versioned_v1" not in {item.id for item in second.observations}
    assert "versioned_v2" in {item.id for item in second.observations}
    assert [(item.extractor, item.version) for item in second.extractions] == [("versioned", "2")]
    assert second.lineage is not None
    assert second.lineage.processor_version == "2"


def test_signal_extractor_collects_event_root_and_error_references() -> None:
    record = _run_record()
    trace = record.trace
    assert trace is not None
    event = next(signal for signal in trace.signals if isinstance(signal, Event))
    reference_signal = next(signal for signal in trace.signals if isinstance(signal, Reference))
    event_ref = EvidenceRef(kind=ReferenceKind.CUSTOM, id="event_ref")
    error_ref = EvidenceRef(kind=ReferenceKind.ERROR, id="error_ref")
    referenced_event = event.model_copy(update={"body": None, "reference": event_ref})
    root_reference = reference_signal.model_copy(update={"span_id": None})
    spans = tuple(
        span.model_copy(update={"errors": (error_ref,)}) if span.operation == "tool" else span
        for span in trace.spans
    )
    expanded = trace.model_copy(
        update={
            "signals": (*trace.signals, referenced_event),
            "references": (root_reference,),
            "spans": spans,
        }
    )

    result = SignalExtractor().extract(
        expanded,
        registry=DEFAULT_SEMANTIC_REGISTRY,
        context=ExtractionContext(
            run_id=record.run_id,
            benchmark_id=record.benchmark_id,
            experiment_id=record.experiment_id,
            case_id=record.case_id,
            variant_id=record.variant_id,
        ),
    )

    assert {reference.id for reference in result.references} >= {
        "event_ref",
        "error_ref",
        root_reference.reference.id,
    }


def test_extraction_replay_requires_inline_or_loaded_trace() -> None:
    with pytest.raises(ReplayError, match="requires a recorded ABP trace"):
        replay_extraction(_run_record().model_copy(update={"trace": None}), SignalExtractor())


def test_canonicalization_replay_uses_newest_map_and_preserves_source_lineage() -> None:
    old_map = _source_map(version=1, semantic_type="llm.model.name")
    snapshot = _source_snapshot(old_map, "model-a")
    record = _run_record().model_copy(update={"source_snapshots": (snapshot,)})
    newer_map = _source_map(version=2, semantic_type="llm.model.response")
    duplicate_old_map = _source_map(version=1, semantic_type="llm.model.requested")

    derived = replay_canonicalization(record, (newer_map, duplicate_old_map))
    repeated = replay_canonicalization(derived, (newer_map,), run_id="canonical_2")

    canonical = next(
        observation
        for observation in derived.observations
        if observation.tags.get("replay") == ReplayKind.CANONICALIZATION
    )
    assert canonical.semantic_type == "llm.model.response"
    assert canonical.value == "model-a"
    assert canonical.kind.value == "factor"
    assert derived.canonicalizations[0] == CanonicalizationResult.model_validate(
        derived.canonicalizations[0]
    )
    assert derived.lineage is not None
    assert derived.lineage.kind is ReplayKind.CANONICALIZATION
    assert derived.lineage.source_maps == ("sdk.map@2",)
    assert repeated.run_id == "canonical_2"
    assert (
        len(
            [
                observation
                for observation in repeated.observations
                if observation.tags.get("replay") == ReplayKind.CANONICALIZATION
            ]
        )
        == 1
    )
    assert (
        RunRecord.model_validate(
            run_record_payload_from_yaml_view(run_record_to_yaml_view(repeated))
        )
        == repeated
    )


def test_canonicalization_replay_handles_numeric_facts_and_missing_inputs() -> None:
    numeric_map = _source_map(version=1, semantic_type=Semantic.LLM_TOKENS_INPUT)
    snapshot = _source_snapshot(numeric_map, 12)
    record = _run_record().model_copy(update={"source_snapshots": (snapshot,)})

    derived = replay_canonicalization(record, (numeric_map,))

    canonical = next(
        observation
        for observation in derived.observations
        if observation.tags.get("replay") == ReplayKind.CANONICALIZATION
    )
    assert canonical.kind.value == "metric"
    assert canonical.value == 12
    with pytest.raises(ReplayError, match="requires retained source snapshots"):
        replay_canonicalization(_run_record(), (numeric_map,))
    with pytest.raises(ReplayError, match="Missing source maps: sdk.map"):
        replay_canonicalization(record, ())


def test_trace_yaml_preserves_additive_fields_and_rejects_invalid_shapes() -> None:
    trace = _run_record().trace
    assert trace is not None
    view = trace_to_yaml_view(trace, extensions={"new_field": {"value": 1}})
    payload, extensions = trace_payload_from_yaml_view(view)

    assert Trace.model_validate(payload) == trace
    assert extensions == {"new_field": {"value": 1}}
    supplied_extensions = {"retained": True}
    payload, extensions = trace_payload_from_yaml_view(
        trace.model_dump(mode="json") | {"extensions": supplied_extensions, "future": True}
    )
    assert Trace.model_validate(payload) == trace
    assert extensions == {"retained": True, "future": True}
    assert supplied_extensions == {"retained": True}
    assert trace_yaml_schema()["properties"]["record"]["properties"]["type"] == {"const": "trace"}
    with pytest.raises(ValueError, match="trace must be a mapping"):
        trace_payload_from_yaml_view({"record": {"type": "trace"}, "trace": []})
    with pytest.raises(ValueError, match="trace.extensions must be a mapping"):
        trace_payload_from_yaml_view(trace.model_dump(mode="json") | {"extensions": []})


def test_run_record_protocol_and_extension_validation(tmp_path: Path) -> None:
    record = _run_record()
    view = run_record_to_yaml_view(record)
    assert view["metrics"]["measurements"]["quality"]["sequence"] == 1
    assert view["metrics"]["factors"]["model"]["sequence"] == 2
    view["future_section"] = {"value": 1}
    restored = RunRecord.model_validate(run_record_payload_from_yaml_view(view))

    assert restored.extensions == {"future_section": {"value": 1}}
    inferred_payload = record.model_dump(mode="json")
    inferred_payload.pop("protocol_version")
    inferred_payload.pop("semantic_registry_version")
    inferred_payload.pop("parent_run_id")
    inferred_payload["lineage"] = {
        "kind": "extraction",
        "parent_run_id": "parent_run",
        "processor": "extractor",
        "processor_version": "1",
        "source_record_version": RECORD_VERSION,
    }
    inferred = RunRecord.model_validate(inferred_payload)
    assert inferred.protocol_version == 1
    assert inferred.semantic_registry_version == 1
    assert inferred.parent_run_id == "parent_run"
    empty_sections = RunRecord.model_validate(
        run_record_payload_from_yaml_view(
            view
            | {
                "protocol": None,
                "canonicalization": None,
                "extraction": None,
            }
        )
    )
    assert empty_sections.trace == record.trace
    with pytest.raises(ReplayError, match="path must be a string"):
        load_run_record(_write_artifact_record(record, trace_value={"bad": True}, root=tmp_path))
    with pytest.raises(ValidationError):
        RunRecord.model_validate(record.model_dump(mode="json") | {"protocol_version": 2})
    with pytest.raises(ValidationError):
        RunRecord.model_validate(record.model_dump(mode="json") | {"record_version": 99})
    with pytest.raises(RecordingError, match="protocol.name must be 'abp'"):
        run_record_payload_from_yaml_view(view | {"protocol": {"name": "otel", "version": 1}})
    with pytest.raises(RecordingError, match="protocol must be a mapping"):
        run_record_payload_from_yaml_view(view | {"protocol": []})
    with pytest.raises(RecordingError, match="canonicalization must be a mapping"):
        run_record_payload_from_yaml_view(view | {"canonicalization": []})
    with pytest.raises(RecordingError, match="extraction must be a mapping"):
        run_record_payload_from_yaml_view(view | {"extraction": []})
    with pytest.raises(RecordingError, match="extensions must be a mapping"):
        run_record_payload_from_yaml_view(view | {"extensions": []})
    invalid_sequence = run_record_to_yaml_view(record)
    invalid_sequence["metrics"]["measurements"]["quality"]["sequence"] = "first"
    with pytest.raises(RecordingError, match="sequence must be an integer"):
        run_record_payload_from_yaml_view(invalid_sequence)
    legacy_grouped = run_record_to_yaml_view(record)
    for group in legacy_grouped["metrics"].values():
        for observation in group.values():
            observation.pop("sequence")
    legacy_payload = run_record_payload_from_yaml_view(legacy_grouped)
    assert [observation["id"] for observation in legacy_payload["observations"]] == [
        "obs_2",
        "obs_1",
        "obs_3",
        "obs_4",
    ]
    artifact_view = run_record_to_yaml_view(
        record.model_copy(
            update={
                "trace": None,
                "trace_artifact": ArtifactRef(
                    id="abp_trace",
                    name="ABP trace",
                    value="trace.yaml",
                ),
            }
        )
    )
    artifact_view["trace"]["extensions"] = []
    with pytest.raises(RecordingError, match="trace.extensions must be a mapping"):
        run_record_payload_from_yaml_view(artifact_view)


def test_recording_validates_trace_limit_and_preflights_trace_collision(tmp_path: Path) -> None:
    result = _experiment_result(output={"content": "x" * 2_000})
    with pytest.raises(ValueError, match="trace_inline_limit_bytes must be at least 1"):
        record_experiment(result, tmp_path / "invalid", trace_inline_limit_bytes=0)
    with pytest.raises(ValueError, match="trace_inline_limit_bytes must be at least 1"):
        run_record_from_result(
            result.runs[0],
            artifacts_dir=tmp_path / "direct-artifacts",
            root_dir=tmp_path,
            trace_inline_limit_bytes=0,
        )

    output_dir = tmp_path / "record"
    trace_path = output_dir / "artifacts" / result.runs[0].run_id / "trace.yaml"
    trace_path.parent.mkdir(parents=True)
    trace_path.write_text("occupied", encoding="utf-8")
    with pytest.raises(RecordingError, match="Record target already exists"):
        record_experiment(result, output_dir, trace_inline_limit_bytes=1)
    assert not (output_dir / "experiment.yaml").exists()

    direct_artifacts = tmp_path / "direct" / "artifacts"
    direct_trace = direct_artifacts / result.runs[0].run_id / "trace.yaml"
    direct_trace.parent.mkdir(parents=True)
    direct_trace.write_text("occupied", encoding="utf-8")
    with pytest.raises(RecordingError, match="Trace artifact already exists"):
        run_record_from_result(
            result.runs[0],
            artifacts_dir=direct_artifacts,
            root_dir=tmp_path / "direct",
            trace_inline_limit_bytes=1,
        )


def test_trace_artifact_reports_invalid_content(tmp_path: Path) -> None:
    record = _run_record()
    missing_path = _write_artifact_record(
        record,
        trace_value="missing.yaml",
        root=tmp_path,
    )
    with pytest.raises(ReplayError, match="Trace artifact does not exist"):
        load_run_record(missing_path, root_dir=tmp_path)
    traversal_path = _write_artifact_record(
        record,
        trace_value="../outside.yaml",
        root=tmp_path,
    )
    with pytest.raises(ReplayError, match="must stay inside"):
        load_run_record(traversal_path, root_dir=tmp_path)

    run_path = _write_artifact_record(record, trace_value="trace.yaml", root=tmp_path)
    (tmp_path / "trace.yaml").write_text("- invalid\n", encoding="utf-8")
    with pytest.raises(ReplayError, match="must contain a mapping"):
        load_run_record(run_path, root_dir=tmp_path)

    (tmp_path / "trace.yaml").write_text("trace: []\n", encoding="utf-8")
    with pytest.raises(ReplayError, match="Invalid trace artifact"):
        load_run_record(run_path, root_dir=tmp_path)

    trace = record.trace
    assert trace is not None
    dump_yaml(trace_to_yaml_view(trace), tmp_path / "trace.yaml")
    loaded = load_run_record(run_path)
    assert loaded.trace == trace


def test_run_context_retains_source_snapshot() -> None:
    source_map = _source_map(version=1, semantic_type=Semantic.LLM_MODEL_NAME)
    snapshot = _source_snapshot(source_map, "model-a")
    ctx = RunContext(
        benchmark_id="trace-benchmark",
        case=Case(id="case_1"),
        variant=Variant(id="variant_1"),
    )

    assert ctx.retain_source_snapshot(snapshot) == snapshot
    assert ctx.source_snapshots == [snapshot]


def test_partial_trace_persists_without_closing_open_span_manually() -> None:
    ctx = RunContext(
        benchmark_id="trace-benchmark",
        case=Case(id="case_1"),
        variant=Variant(id="variant_1"),
    )
    unfinished = ctx.span("unfinished")
    unfinished.__enter__()

    trace = ctx.finalize(partial=True)
    record = _run_record().model_copy(update={"trace": trace, "spans": tuple(ctx.spans)})
    restored = RunRecord.model_validate(
        run_record_payload_from_yaml_view(run_record_to_yaml_view(record))
    )

    assert trace.partial is True
    assert trace.spans[-1].partial is True
    assert restored.trace == trace


def _experiment_result(*, output: dict[str, str] | None = None) -> ExperimentResult:
    record = _run_record(output=output)
    assert record.trace is not None
    run = RunResult(
        run_id=record.run_id,
        benchmark_id=record.benchmark_id,
        experiment_id=record.experiment_id,
        case_id=record.case_id,
        variant_id=record.variant_id,
        status=record.status,
        evaluation_status=record.evaluation_status,
        case=record.case,
        task_result=TaskResult(
            output=record.task_output,
            status=record.task_status,
            observations=list(record.observations),
            spans=list(record.spans),
            artifacts=list(record.artifacts),
        ),
        trace=record.trace,
        source_snapshots=record.source_snapshots,
    )
    return ExperimentResult(
        experiment_id=record.experiment_id,
        benchmark_id=record.benchmark_id,
        plan=BenchmarkPlan(
            benchmark_id=record.benchmark_id,
            case_ids=(record.case_id,),
            case_count=1,
            variant_count=1,
            planned_run_count=1,
        ),
        runs=[run],
        environment=EnvironmentMetadata(python_version="3.11", platform="test", cwd="."),
    )


def _run_record(*, output: dict[str, str] | None = None) -> RunRecord:
    ctx = RunContext(
        benchmark_id="trace-benchmark",
        experiment_id="experiment_1",
        run_id="run_1",
        case=Case(id="case_1"),
        variant=Variant(id="variant_1"),
    )
    ctx.metric("quality", 0.75, semantic_type=Semantic.QUALITY_SCORE)
    ctx.factor_observation("model", "model-a", semantic_type=Semantic.LLM_MODEL_NAME)
    ctx.diagnostic("diagnostic", "retained")
    with ctx.span("tool", kind="tool") as span:
        span.artifact("payload", {"retained": True})
    task_output = {"content": "ok"} if output is None else output
    trace = ctx.finalize(output=task_output)
    return RunRecord(
        protocol_version=trace.protocol_version,
        semantic_registry_version=DEFAULT_SEMANTIC_REGISTRY.version,
        run_id="run_1",
        experiment_id="experiment_1",
        benchmark_id="trace-benchmark",
        case_id="case_1",
        variant_id="variant_1",
        status=RunStatus.PASSED,
        evaluation_status=EvaluationStatus.PASSED,
        task_status=TaskStatus.PASSED,
        case=ctx.case,
        task_output=task_output,
        observations=tuple(ctx.observations),
        spans=tuple(ctx.spans),
        artifacts=tuple(ctx.artifacts),
        trace=trace,
    )


def _source_map(*, version: int, semantic_type: str) -> SourceMap:
    return SourceMap(
        id="sdk.map",
        version=version,
        source_system="sdk",
        convention_version="1",
        rules=(
            RenameRule(
                sources=(SourceSelector(key="model"),),
                semantic_type=semantic_type,
                capture=CaptureLevel.FULL,
            ),
        ),
    )


def _source_snapshot(source_map: SourceMap, value: str | int) -> SourceSnapshot:
    return canonicalize(
        SourceData(system="sdk", convention_version="1", values={"model": value}),
        source_map,
        capture=CaptureSession(CapturePolicy.full(retain_source_attributes=True)),
    ).source_snapshot


def _write_artifact_record(
    record: RunRecord,
    *,
    trace_value: str | dict[str, bool],
    root: Path | None = None,
) -> Path:
    active_root = Path.cwd() if root is None else root
    run_path = active_root / f"run-{sha256(repr(trace_value).encode()).hexdigest()[:8]}.yaml"
    artifact_record = record.model_copy(
        update={
            "trace": None,
            "trace_artifact": ArtifactRef(
                id="abp_trace",
                name="ABP trace",
                media_type=TRACE_ARTIFACT_MEDIA_TYPE,
                value=trace_value,
            ),
        }
    )
    dump_yaml(run_record_to_yaml_view(artifact_record), run_path, schema_name="run_record")
    return run_path
