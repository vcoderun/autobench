from __future__ import annotations

import pytest
from pydantic import ValidationError

from autobench import (
    OPENAI_RESPONSES_SOURCE_MAP,
    OPENINFERENCE_SOURCE_MAP,
    OTEL_GENAI_SOURCE_MAP,
    CanonicalizationResult,
    ClassificationRule,
    ReferenceRule,
    RenameRule,
    RetainedSourceFact,
    Semantic,
    SourceData,
    SourceMap,
    SourceSelector,
    SourceSnapshot,
    SpanClassification,
    SplitOutput,
    SplitRule,
    UnitConversionRule,
    canonicalize,
    recanonicalize,
    resolve_nested_value,
    resolve_source_value,
    source_map_payload_from_yaml_view,
    source_map_to_yaml_view,
    source_selector_label,
)
from autobench.protocol import (
    CapturePolicy,
    CaptureSession,
    ReferenceKind,
    SerializedValue,
    SourceProvenance,
)
from autobench.protocol.traces import DiagnosticSeverity


def fact_map(result: CanonicalizationResult) -> dict[str, SerializedValue]:
    return {fact.semantic_type: fact.value for fact in result.facts}


def test_otel_manifest_canonicalizes_usage_content_and_operation_with_provenance() -> None:
    result = canonicalize(
        SourceData(
            system="otel.genai",
            convention_version="1.43.0",
            values={
                "gen_ai.request.model": "requested-model",
                "gen_ai.response.model": "served-model",
                "gen_ai.provider.name": "openai",
                "gen_ai.usage.input_tokens": 12,
                "gen_ai.usage.output_tokens": 5,
                "gen_ai.usage.cache_read.input_tokens": 3,
                "gen_ai.usage.reasoning.output_tokens": 2,
                "gen_ai.tool.call.arguments": {"city": "Istanbul"},
                "gen_ai.input.messages": [{"role": "user", "content": "private"}],
                "gen_ai.operation.name": "chat",
            },
        ),
        OTEL_GENAI_SOURCE_MAP,
    )

    facts = fact_map(result)
    assert facts[Semantic.LLM_MODEL_REQUESTED] == "requested-model"
    assert facts[Semantic.LLM_MODEL_RESPONSE] == "served-model"
    assert facts[Semantic.LLM_PROVIDER_NAME] == "openai"
    assert facts[Semantic.LLM_TOKENS_INPUT] == 12
    assert facts[Semantic.LLM_TOKENS_OUTPUT] == 5
    assert facts[Semantic.LLM_TOKENS_CACHED_INPUT] == 3
    assert facts[Semantic.LLM_TOKENS_REASONING_OUTPUT] == 2
    tool_arguments = facts[Semantic.TOOL_CALL_ARGUMENTS]
    input_messages = facts[Semantic.MESSAGE_INPUT]
    assert isinstance(tool_arguments, dict)
    assert "sha256" in tool_arguments
    assert isinstance(input_messages, dict)
    assert "sha256" in input_messages
    assert result.classification is not None
    assert result.classification.kind == "llm"
    assert result.classification.operation == "chat"
    assert result.facts[0].sources[0].source_map_id == "otel.genai"
    assert result.facts[0].sources[0].source_map_version == 1
    assert result.source_snapshot.facts
    assert all(not fact.available for fact in result.source_snapshot.facts)
    assert {fact.reason for fact in result.source_snapshot.facts} == {"source_retention_disabled"}


@pytest.mark.parametrize(
    ("system", "version", "codes"),
    [
        ("wrong", "1.43.0", {"source_system_mismatch"}),
        ("otel.genai", "old", {"source_version_mismatch"}),
        ("wrong", "old", {"source_system_mismatch", "source_version_mismatch"}),
    ],
)
def test_source_map_rejects_incompatible_source_identity(
    system: str,
    version: str,
    codes: set[str],
) -> None:
    result = canonicalize(
        SourceData(system=system, convention_version=version),
        OTEL_GENAI_SOURCE_MAP,
    )

    assert not result.facts
    assert {diagnostic.code for diagnostic in result.diagnostics} == codes
    assert all(diagnostic.severity is DiagnosticSeverity.ERROR for diagnostic in result.diagnostics)


def test_alias_merge_reports_deprecation_and_rejects_conflicting_values() -> None:
    source_map = SourceMap(
        id="aliases",
        version=1,
        source_system="native",
        convention_version="1",
        rules=(
            RenameRule(
                sources=(
                    SourceSelector(key="provider"),
                    SourceSelector(key="legacy_provider", deprecated=True),
                ),
                semantic_type=Semantic.LLM_PROVIDER,
            ),
            RenameRule(
                sources=(SourceSelector(key="provider"),),
                semantic_type=Semantic.LLM_MODEL_RESPONSE,
            ),
        ),
    )
    equal = canonicalize(
        SourceData(
            system="native",
            convention_version="1",
            values={"provider": "openai", "legacy_provider": "openai"},
        ),
        source_map,
    )

    assert len(equal.facts) == 2
    provider_fact = next(
        fact for fact in equal.facts if fact.semantic_type == Semantic.LLM_PROVIDER_NAME
    )
    assert len(provider_fact.sources) == 2
    assert {diagnostic.code for diagnostic in equal.diagnostics} == {
        "deprecated_semantic",
        "deprecated_source_key",
    }

    conflicting = canonicalize(
        SourceData(
            system="native",
            convention_version="1",
            values={"provider": "openai", "legacy_provider": "azure"},
        ),
        source_map,
    )
    assert Semantic.LLM_PROVIDER_NAME not in fact_map(conflicting)
    assert fact_map(conflicting)[Semantic.LLM_MODEL_RESPONSE] == "openai"
    assert "ambiguous_source_aliases" in {diagnostic.code for diagnostic in conflicting.diagnostics}


def test_split_reference_unit_and_classification_rules_produce_typed_evidence() -> None:
    source_map = SourceMap(
        id="transforms",
        version=2,
        source_system="native",
        convention_version="2",
        instrumentor="tests",
        instrumented_library_version="9",
        rules=(
            SplitRule(
                source=SourceSelector(key="usage"),
                outputs=(
                    SplitOutput(path=("input",), semantic_type=Semantic.LLM_TOKENS_INPUT),
                    SplitOutput(path=("output",), semantic_type=Semantic.LLM_TOKENS_OUTPUT),
                    SplitOutput(path=("missing",), semantic_type=Semantic.LLM_TOKENS_TOTAL),
                ),
            ),
            UnitConversionRule(
                source=SourceSelector(key="latency_ms"),
                semantic_type=Semantic.TIME_LATENCY,
                source_unit="ms",
                target_unit="s",
                multiplier=0.001,
                offset=0.5,
            ),
            ReferenceRule(
                source=SourceSelector(key="asset"),
                semantic_type="artifact.output",
                reference_kind=ReferenceKind.ASSET,
                id_path=("id",),
                version_path=("version",),
                media_type="text/plain",
            ),
            ReferenceRule(
                source=SourceSelector(key="external_trace"),
                semantic_type="trace.external",
                reference_kind=ReferenceKind.EXTERNAL_TRACE,
            ),
            ClassificationRule(
                source=SourceSelector(key="operation"),
                cases={"lookup": SpanClassification(operation="lookup", kind="retriever")},
            ),
        ),
    )
    capture = CaptureSession(CapturePolicy.full(retain_source_attributes=True))
    result = canonicalize(
        SourceData(
            system="native",
            convention_version="2",
            values={
                "usage": {"input": 10, "output": 4},
                "latency_ms": 250,
                "asset": {"id": "asset-1", "version": "v3"},
                "external_trace": "trace-1",
                "operation": "lookup",
            },
        ),
        source_map,
        capture=capture,
    )

    facts = {fact.semantic_type: fact for fact in result.facts}
    assert facts[Semantic.LLM_TOKENS_INPUT].value == 10
    assert facts[Semantic.LLM_TOKENS_OUTPUT].value == 4
    assert facts[Semantic.TIME_LATENCY].value == 0.75
    assert facts[Semantic.TIME_LATENCY].unit == "s"
    reference = facts["artifact.output"].reference
    assert reference is not None
    assert reference.id == "asset-1"
    assert reference.version == "v3"
    assert reference.media_type == "text/plain"
    external_trace = facts["trace.external"].reference
    assert external_trace is not None
    assert external_trace.id == "trace-1"
    assert external_trace.version is None
    assert facts[Semantic.TIME_LATENCY].sources[0].instrumentor == "tests"
    assert result.classification is not None
    assert result.classification.kind == "retriever"
    assert "source_path_unavailable" in {diagnostic.code for diagnostic in result.diagnostics}
    assert all(fact.available for fact in result.source_snapshot.facts)


def test_mapping_failures_and_conflicts_remain_explicit() -> None:
    source_map = SourceMap(
        id="failures",
        version=1,
        source_system="native",
        convention_version="1",
        rules=(
            ClassificationRule(
                source=SourceSelector(key="operation_a"),
                cases={"run": SpanClassification(operation="run", kind="llm")},
            ),
            ClassificationRule(
                source=SourceSelector(key="operation_b"),
                cases={"run": SpanClassification(operation="run", kind="tool")},
            ),
            ClassificationRule(
                source=SourceSelector(key="unknown_operation"),
                cases={"known": SpanClassification(operation="known", kind="task")},
            ),
            ReferenceRule(
                source=SourceSelector(key="bad_reference"),
                semantic_type="artifact.bad",
                reference_kind=ReferenceKind.ARTIFACT,
            ),
            ReferenceRule(
                source=SourceSelector(key="bad_version"),
                semantic_type="artifact.versioned",
                reference_kind=ReferenceKind.ASSET,
                id_path=("id",),
                version_path=("version",),
            ),
            UnitConversionRule(
                source=SourceSelector(key="boolean_duration"),
                semantic_type=Semantic.TIME_LATENCY,
                source_unit="ms",
                target_unit="s",
            ),
            RenameRule(
                sources=(SourceSelector(key="score_a"),),
                semantic_type=Semantic.QUALITY_SCORE,
            ),
            RenameRule(
                sources=(SourceSelector(key="score_b"),),
                semantic_type=Semantic.QUALITY_SCORE,
            ),
            RenameRule(
                sources=(SourceSelector(key="success_a"),),
                semantic_type=Semantic.RESULT_SUCCESS,
                authority=0.5,
            ),
            RenameRule(
                sources=(SourceSelector(key="success_b"),),
                semantic_type=Semantic.RESULT_SUCCESS,
            ),
        ),
    )
    result = canonicalize(
        SourceData(
            system="native",
            convention_version="1",
            values={
                "operation_a": "run",
                "operation_b": "run",
                "unknown_operation": "unknown",
                "bad_reference": 7,
                "bad_version": {"id": "asset", "version": 2},
                "boolean_duration": True,
                "score_a": 0.2,
                "score_b": 0.9,
                "success_a": True,
                "success_b": True,
            },
        ),
        source_map,
    )

    codes = {diagnostic.code for diagnostic in result.diagnostics}
    assert {
        "ambiguous_canonical_value",
        "ambiguous_classification",
        "classification_unavailable",
        "reference_mapping_failed",
        "unit_conversion_failed",
    } <= codes
    assert result.classification is None
    scores = [fact for fact in result.facts if fact.semantic_type == Semantic.QUALITY_SCORE]
    assert [fact.value for fact in scores] == [0.2, 0.9]
    success = next(fact for fact in result.facts if fact.semantic_type == Semantic.RESULT_SUCCESS)
    assert success.authority == 1.0
    assert len(success.sources) == 2


def test_retained_source_facts_enable_recanonicalization_without_raw_content_guessing() -> None:
    original_map = SourceMap(
        id="native.models",
        version=1,
        source_system="native",
        convention_version="1",
        rules=(
            RenameRule(
                sources=(SourceSelector(key="request", path=("model",)),),
                semantic_type=Semantic.LLM_MODEL_REQUESTED,
            ),
        ),
    )
    original = canonicalize(
        SourceData(
            system="native",
            convention_version="1",
            values={"request": {"model": "gpt-x"}},
        ),
        original_map,
        capture=CaptureSession(CapturePolicy.full(retain_source_attributes=True)),
    )
    assert original.source_snapshot.facts[0].available is True

    revised_map = original_map.model_copy(
        update={
            "version": 2,
            "rules": (
                RenameRule(
                    sources=(SourceSelector(key="request", path=("model",)),),
                    semantic_type=Semantic.LLM_MODEL_RESPONSE,
                ),
            ),
        }
    )
    replayed = recanonicalize(original.source_snapshot, revised_map)
    assert replayed.facts[0].semantic_type == Semantic.LLM_MODEL_RESPONSE
    assert replayed.facts[0].value == "gpt-x"
    assert replayed.replayed_from == "native.models@1"

    prompt_map = SourceMap(
        id="native.prompt",
        version=1,
        source_system="native",
        convention_version="1",
        rules=(
            RenameRule(
                sources=(SourceSelector(key="prompt"),),
                semantic_type=Semantic.PROMPT_SYSTEM,
            ),
        ),
    )
    protected = canonicalize(
        SourceData(
            system="native",
            convention_version="1",
            values={"prompt": "private prompt"},
        ),
        prompt_map,
        capture=CaptureSession(CapturePolicy(retain_source_attributes=True)),
    )
    assert protected.source_snapshot.facts[0].available is False
    assert protected.source_snapshot.facts[0].reason == "capture_hash"
    unavailable = recanonicalize(protected.source_snapshot, prompt_map)
    assert not unavailable.facts
    assert unavailable.diagnostics[0].code == "source_fact_unavailable"
    assert unavailable.diagnostics[0].details == {"reason": "capture_hash"}

    missing_map = prompt_map.model_copy(
        update={
            "rules": (
                RenameRule(
                    sources=(SourceSelector(key="new_prompt"),),
                    semantic_type=Semantic.PROMPT_SYSTEM,
                ),
            )
        }
    )
    missing = recanonicalize(protected.source_snapshot, missing_map)
    assert missing.diagnostics[0].details == {"reason": "not_retained"}


def test_recanonicalization_resolves_descendants_and_rejects_identity_mismatch() -> None:
    snapshot = SourceSnapshot(
        system="native",
        convention_version="1",
        source_map_id="old",
        source_map_version=1,
        facts=(
            RetainedSourceFact(
                selector=SourceSelector(key="response", path=("usage",)),
                value={"input": 9},
                available=True,
            ),
        ),
    )
    source_map = SourceMap(
        id="new",
        version=2,
        source_system="native",
        convention_version="1",
        rules=(
            RenameRule(
                sources=(SourceSelector(key="response", path=("usage", "input")),),
                semantic_type=Semantic.LLM_TOKENS_INPUT,
            ),
        ),
    )
    replayed = recanonicalize(snapshot, source_map)
    assert replayed.facts[0].value == 9

    wrong = recanonicalize(
        snapshot,
        source_map.model_copy(update={"source_system": "other", "convention_version": "2"}),
    )
    assert not wrong.facts
    assert {diagnostic.code for diagnostic in wrong.diagnostics} == {
        "source_system_mismatch",
        "source_version_mismatch",
    }


def test_recanonicalization_reapplies_every_rule_without_rewriting_snapshot() -> None:
    snapshot = SourceSnapshot(
        system="native",
        convention_version="1",
        source_map_id="old",
        source_map_version=1,
        facts=(
            RetainedSourceFact(
                selector=SourceSelector(key="usage"),
                value={"input": 4},
                available=True,
            ),
            RetainedSourceFact(
                selector=SourceSelector(key="operation"),
                value="chat",
                available=True,
            ),
            RetainedSourceFact(
                selector=SourceSelector(key="trace_id"),
                value="trace-1",
                available=True,
            ),
            RetainedSourceFact(
                selector=SourceSelector(key="latency_ms"),
                value=500,
                available=True,
            ),
        ),
    )
    source_map = SourceMap(
        id="new",
        version=2,
        source_system="native",
        convention_version="1",
        rules=(
            SplitRule(
                source=SourceSelector(key="usage"),
                outputs=(
                    SplitOutput(
                        path=("input",),
                        semantic_type=Semantic.LLM_TOKENS_INPUT,
                    ),
                ),
            ),
            ClassificationRule(
                source=SourceSelector(key="operation"),
                cases={"chat": SpanClassification(operation="chat", kind="llm")},
            ),
            ReferenceRule(
                source=SourceSelector(key="trace_id"),
                semantic_type="trace.external",
                reference_kind=ReferenceKind.EXTERNAL_TRACE,
            ),
            UnitConversionRule(
                source=SourceSelector(key="latency_ms"),
                semantic_type=Semantic.TIME_LATENCY,
                source_unit="ms",
                target_unit="s",
                multiplier=0.001,
            ),
        ),
    )

    replayed = recanonicalize(snapshot, source_map)
    assert fact_map(replayed)[Semantic.LLM_TOKENS_INPUT] == 4
    assert fact_map(replayed)[Semantic.TIME_LATENCY] == 0.5
    assert replayed.classification is not None
    assert replayed.classification.kind == "llm"
    assert (
        next(fact for fact in replayed.facts if fact.semantic_type == "trace.external").reference
        is not None
    )
    assert replayed.source_snapshot == snapshot

    suppressed = recanonicalize(
        snapshot,
        source_map,
        capture=CaptureSession(CapturePolicy.none()),
    )
    assert {fact.semantic_type for fact in suppressed.facts} == {"trace.external"}
    assert [diagnostic.code for diagnostic in suppressed.diagnostics].count("capture_omitted") == 2


def test_recanonicalization_does_not_infer_missing_descendant_paths() -> None:
    snapshot = SourceSnapshot(
        system="native",
        convention_version="1",
        source_map_id="old",
        source_map_version=1,
        facts=(
            RetainedSourceFact(
                selector=SourceSelector(key="response", path=("usage",)),
                value={"output": 3},
                available=True,
            ),
        ),
    )
    source_map = SourceMap(
        id="new",
        version=2,
        source_system="native",
        convention_version="1",
        rules=(
            RenameRule(
                sources=(SourceSelector(key="response", path=("usage", "input")),),
                semantic_type=Semantic.LLM_TOKENS_INPUT,
            ),
        ),
    )

    result = recanonicalize(snapshot, source_map)
    assert not result.facts
    assert result.diagnostics[0].details == {"reason": "not_retained"}


@pytest.mark.parametrize(
    ("policy", "value", "semantic_type", "reason"),
    [
        (
            CapturePolicy.full(
                retain_source_attributes=True,
                max_inline_bytes=4,
                max_string_length=100,
            ),
            "long source value",
            "custom.payload",
            "artifact_reference",
        ),
        (
            CapturePolicy.full(
                retain_source_attributes=True,
                max_string_length=4,
                max_inline_bytes=100,
            ),
            "long source value",
            "custom.payload",
            "truncated",
        ),
        (
            CapturePolicy(retain_source_attributes=True),
            "secret",
            "environment.api_key",
            "capture_omitted",
        ),
    ],
)
def test_source_retention_records_why_replay_is_unavailable(
    policy: CapturePolicy,
    value: str,
    semantic_type: str,
    reason: str,
) -> None:
    source_map = SourceMap(
        id="retention",
        version=1,
        source_system="native",
        convention_version="1",
        rules=(
            RenameRule(
                sources=(SourceSelector(key="value"),),
                semantic_type=semantic_type,
            ),
        ),
    )
    result = canonicalize(
        SourceData(
            system="native",
            convention_version="1",
            values={"value": value},
        ),
        source_map,
        capture=CaptureSession(policy),
    )

    assert result.source_snapshot.facts[0].available is False
    assert result.source_snapshot.facts[0].reason == reason
    if reason == "artifact_reference":
        assert result.facts[0].reference is not None
    if reason == "capture_omitted":
        assert not result.facts


def test_source_map_yaml_resolvers_and_validation_are_stable() -> None:
    view = source_map_to_yaml_view(OPENAI_RESPONSES_SOURCE_MAP)
    payload = source_map_payload_from_yaml_view(view)
    assert SourceMap.model_validate(payload) == OPENAI_RESPONSES_SOURCE_MAP
    assert view["record"] == {"type": "source_map", "version": 1}
    assert view["source_map"]["source"] == {
        "system": "openai.responses",
        "convention": "v1",
        "instrumentor": "autobench.openai",
    }
    assert "source_system" not in view["source_map"]
    versioned = OPENAI_RESPONSES_SOURCE_MAP.model_copy(
        update={"instrumented_library_version": "2.4.0"}
    )
    versioned_view = source_map_to_yaml_view(versioned)
    assert versioned_view["source_map"]["source"]["library_version"] == "2.4.0"
    assert SourceMap.model_validate(source_map_payload_from_yaml_view(versioned_view)) == versioned
    plain_view = source_map_to_yaml_view(OPENINFERENCE_SOURCE_MAP)
    assert plain_view["source_map"]["source"] == {
        "system": "openinference",
        "convention": "1.0",
    }
    library_only = OPENINFERENCE_SOURCE_MAP.model_copy(
        update={"instrumented_library_version": "0.1.0"}
    )
    assert (
        SourceMap.model_validate(
            source_map_payload_from_yaml_view(source_map_to_yaml_view(library_only))
        )
        == library_only
    )
    assert (
        source_map_payload_from_yaml_view(OPENINFERENCE_SOURCE_MAP.model_dump(mode="json"))["id"]
        == "openinference"
    )
    with pytest.raises(TypeError, match="source_map must be a mapping"):
        source_map_payload_from_yaml_view([])
    with pytest.raises(TypeError, match="source_map.source must be a mapping"):
        source_map_payload_from_yaml_view({"source": "invalid"})
    assert "discriminator" in str(SourceMap.model_json_schema())

    selector = SourceSelector(key="response", path=("items", 1, "id"))
    values: dict[str, SerializedValue] = {"response": {"items": [{"id": "a"}, {"id": "b"}]}}
    assert resolve_source_value(values, selector) == (True, "b")
    assert resolve_nested_value(values["response"], ("items", 4)) == (False, None)
    assert resolve_nested_value(values["response"], ("missing",)) == (False, None)
    assert resolve_source_value(values, SourceSelector(key="missing")) == (False, None)
    assert source_selector_label(selector) == "response.items[1].id"

    with pytest.raises(ValidationError, match="unavailable source facts require a reason"):
        RetainedSourceFact(selector=selector, available=False)
    with pytest.raises(ValidationError, match="available source facts"):
        RetainedSourceFact(selector=selector, available=True, reason="wrong")
    with pytest.raises(ValidationError, match="provided together"):
        SourceProvenance(
            system="native",
            key="value",
            source_map_id="map",
        )


def test_external_manifests_cover_equivalent_paths_without_polluting_registry() -> None:
    openinference = canonicalize(
        SourceData(
            system="openinference",
            convention_version="1.0",
            values={
                "llm.model_name": "model",
                "llm.token_count.prompt": 3,
                "llm.token_count.completion": 2,
                "llm.token_count.total": 5,
            },
        ),
        OPENINFERENCE_SOURCE_MAP,
    )
    assert fact_map(openinference) == {
        Semantic.LLM_MODEL_RESPONSE: "model",
        Semantic.LLM_TOKENS_INPUT: 3,
        Semantic.LLM_TOKENS_OUTPUT: 2,
        Semantic.LLM_TOKENS_TOTAL: 5,
    }

    openai = canonicalize(
        SourceData(
            system="openai.responses",
            convention_version="v1",
            values={
                "request": {"model": "requested"},
                "response": {
                    "model": "served",
                    "usage": {
                        "input_tokens": 10,
                        "output_tokens": 4,
                        "input_tokens_details": {"cached_tokens": 3},
                        "output_tokens_details": {"reasoning_tokens": 2},
                    },
                },
            },
        ),
        OPENAI_RESPONSES_SOURCE_MAP,
    )
    assert fact_map(openai) == {
        Semantic.LLM_MODEL_REQUESTED: "requested",
        Semantic.LLM_MODEL_RESPONSE: "served",
        Semantic.LLM_TOKENS_INPUT: 10,
        Semantic.LLM_TOKENS_OUTPUT: 4,
        Semantic.LLM_TOKENS_CACHED_INPUT: 3,
        Semantic.LLM_TOKENS_REASONING_OUTPUT: 2,
    }

    deprecated = canonicalize(
        SourceData(
            system="otel.genai",
            convention_version="1.43.0",
            values={
                "gen_ai.usage.prompt_tokens": 7,
                "gen_ai.usage.completion_tokens": 4,
            },
        ),
        OTEL_GENAI_SOURCE_MAP,
    )
    assert fact_map(deprecated)[Semantic.LLM_TOKENS_INPUT] == 7
    assert [diagnostic.code for diagnostic in deprecated.diagnostics].count(
        "deprecated_source_key"
    ) == 2
