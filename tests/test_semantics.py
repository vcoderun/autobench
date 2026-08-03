from __future__ import annotations as _annotations

from typing import Any

import pytest
import yaml

from autobench import (
    DEFAULT_SEMANTIC_REGISTRY,
    Semantic,
    SemanticAggregation,
    SemanticCardinality,
    SemanticPrivacy,
    SemanticRegistry,
    SemanticStability,
    SemanticTypeInfo,
    semantic_registry_payload_from_yaml_view,
    semantic_registry_to_yaml_view,
)


def test_default_registry_normalizes_alias() -> None:
    assert DEFAULT_SEMANTIC_REGISTRY.normalize("quality.answer") == Semantic.QUALITY_SCORE
    assert DEFAULT_SEMANTIC_REGISTRY.normalize(Semantic.LLM_PROVIDER) == Semantic.LLM_PROVIDER_NAME
    assert DEFAULT_SEMANTIC_REGISTRY.normalize(Semantic.AGENT_TOOL_NAME) == Semantic.TOOL_NAME
    assert DEFAULT_SEMANTIC_REGISTRY.normalize("llm.requests") == Semantic.LLM_REQUEST_COUNT


def test_default_registry_describes_optimization_grade_semantics() -> None:
    input_tokens = DEFAULT_SEMANTIC_REGISTRY.info_for(Semantic.LLM_TOKENS_INPUT)
    tool_arguments = DEFAULT_SEMANTIC_REGISTRY.info_for(Semantic.TOOL_CALL_ARGUMENTS)

    assert input_tokens is not None
    assert input_tokens.description
    assert input_tokens.stability is SemanticStability.STABLE
    assert input_tokens.privacy is SemanticPrivacy.INTERNAL
    assert input_tokens.cardinality is SemanticCardinality.LOW
    assert input_tokens.aggregation is SemanticAggregation.SUM
    assert tool_arguments is not None
    assert tool_arguments.privacy is SemanticPrivacy.SENSITIVE
    assert tool_arguments.cardinality is SemanticCardinality.UNBOUNDED
    assert not any(
        semantic_id.startswith(("gen_ai.", "llm.token_count."))
        for semantic_id in DEFAULT_SEMANTIC_REGISTRY.types
    )


def test_default_registry_includes_generic_manual_runtime_semantics() -> None:
    assert {
        Semantic.OPERATION_INPUT,
        Semantic.OPERATION_OUTPUT,
        Semantic.ARTIFACT_CONTENT,
        Semantic.FACTOR_VALUE,
        Semantic.EVENT_OCCURRENCE,
        Semantic.DIAGNOSTIC_EVENT,
        Semantic.ERROR_EXCEPTION,
        Semantic.LLM_REQUEST_COUNT,
        Semantic.OPERATION_COUNT,
        Semantic.OPERATION_DEPTH_MAX,
        Semantic.OPERATION_FAN_OUT_MAX,
        Semantic.OPERATION_PARALLELISM,
        Semantic.OPERATION_RETRY_RECOVERED_COUNT,
        Semantic.TIME_CRITICAL_PATH,
        Semantic.VALIDATION_FAILURE_RATE,
        Semantic.STREAM_FIRST_CHUNK,
        Semantic.STREAM_COMPLETED,
        Semantic.STREAM_PARTIAL,
        Semantic.STREAM_FAILED,
        Semantic.OPERATION_RETRY,
        Semantic.OPERATION_REPAIR,
        Semantic.OPERATION_DEFERRED,
        Semantic.OPERATION_DEFERRED_RESOLVED,
        Semantic.VALIDATION_FAILURE,
        Semantic.APPROVAL_REQUESTED,
        Semantic.TOOL_CALL_REQUESTED,
    } <= DEFAULT_SEMANTIC_REGISTRY.types.keys()
    error = DEFAULT_SEMANTIC_REGISTRY.info_for(Semantic.ERROR_EXCEPTION)
    assert error is not None
    assert error.privacy is SemanticPrivacy.SENSITIVE


def test_default_registry_parent_lookup_works_for_codegen_model_semantics() -> None:
    assert DEFAULT_SEMANTIC_REGISTRY.parent_of("ai.codegen.spec_model") == Semantic.LLM_MODEL_NAME
    assert (
        DEFAULT_SEMANTIC_REGISTRY.parent_of("ai.codegen.exploration_model")
        == Semantic.LLM_MODEL_NAME
    )


def test_default_registry_is_a_understands_parent_semantics() -> None:
    assert DEFAULT_SEMANTIC_REGISTRY.is_a("ai.codegen.spec_model", Semantic.LLM_MODEL_NAME)
    assert DEFAULT_SEMANTIC_REGISTRY.is_a("ai.codegen.exploration_model", Semantic.LLM_MODEL_NAME)
    assert not DEFAULT_SEMANTIC_REGISTRY.is_a(Semantic.MONEY_COST, Semantic.LLM_MODEL_NAME)


def test_registry_handles_none_unknown_and_deprecated_semantics() -> None:
    registry = SemanticRegistry(
        types={
            "old.quality": SemanticTypeInfo(
                id="old.quality",
                parent=Semantic.QUALITY_SCORE,
                deprecated=True,
            ),
            "quality.child": SemanticTypeInfo(
                id="quality.child",
                parent=Semantic.QUALITY_SCORE,
            ),
            "quality.grandchild": SemanticTypeInfo(
                id="quality.grandchild",
                parent="quality.child",
            ),
        }
    )

    assert registry.info_for(None) is None
    assert registry.normalize("old.quality") == Semantic.QUALITY_SCORE
    assert registry.parent_of(None) is None
    assert registry.parent_of("missing.semantic") is None
    assert registry.is_a(None, Semantic.QUALITY_SCORE) is False
    assert registry.is_a("quality.child", None) is False
    assert registry.is_a(Semantic.QUALITY_SCORE, Semantic.QUALITY_SCORE) is True
    assert registry.is_a("quality.child", Semantic.QUALITY_SCORE) is True
    assert registry.is_a("quality.grandchild", Semantic.QUALITY_SCORE) is True


def test_semantic_registry_yaml_view_uses_dsl_shape() -> None:
    view = semantic_registry_to_yaml_view(DEFAULT_SEMANTIC_REGISTRY)

    assert view["record"] == {"type": "semantic_registry", "version": 1}
    assert view["semantic_registry"]["version"] == 1
    assert view["semantic_registry"]["types"][Semantic.MONEY_COST] == {
        "unit": "usd",
        "shape": "number",
    }
    assert view["semantic_registry"]["aliases"]["quality.answer"] == Semantic.QUALITY_SCORE


def test_semantic_registry_yaml_view_omits_empty_fields_and_keeps_optional_metadata() -> None:
    registry = SemanticRegistry(
        version=7,
        types={
            "custom.metric": SemanticTypeInfo(
                id="custom.metric",
                parent=Semantic.QUALITY_SCORE,
                description="A custom test metric.",
                aliases=["custom.alias"],
                deprecated=True,
                stability=SemanticStability.EXPERIMENTAL,
                privacy=SemanticPrivacy.SECRET,
                cardinality=SemanticCardinality.HIGH,
                aggregation=SemanticAggregation.ANY,
                tags={"owner": "tests"},
            ),
            "empty.metric": SemanticTypeInfo(id="empty.metric"),
        },
        aliases={"legacy.metric": "custom.metric"},
    )

    view = semantic_registry_to_yaml_view(registry)

    assert view["record"] == {"type": "semantic_registry", "version": 7}
    assert view["semantic_registry"]["types"]["custom.metric"] == {
        "parent": Semantic.QUALITY_SCORE,
        "description": "A custom test metric.",
        "aliases": ["custom.alias"],
        "deprecated": True,
        "stability": "experimental",
        "privacy": "secret",
        "cardinality": "high",
        "aggregation": "any",
        "tags": {"owner": "tests"},
    }
    assert view["semantic_registry"]["types"]["empty.metric"] == {}
    assert view["semantic_registry"]["aliases"] == {"legacy.metric": "custom.metric"}
    assert "stability: experimental" in yaml.safe_dump(view)


def test_semantic_registry_yaml_view_round_trips_to_registry_payload() -> None:
    registry = SemanticRegistry(
        version=3,
        types={
            "custom.metric": SemanticTypeInfo(
                id="custom.metric",
                parent=Semantic.QUALITY_SCORE,
                unit="ms",
                value_shape="number",
            ),
        },
        aliases={"custom.alias": "custom.metric"},
    )

    payload = semantic_registry_payload_from_yaml_view(semantic_registry_to_yaml_view(registry))
    round_trip = SemanticRegistry.model_validate(payload)

    assert round_trip == registry


def test_semantic_registry_payload_accepts_bare_registry_shape() -> None:
    payload = semantic_registry_payload_from_yaml_view(
        {
            "version": 5,
            "types": {
                "custom.metric": {
                    "id": "custom.metric",
                    "value_shape": "number",
                }
            },
            "aliases": {},
        }
    )

    registry = SemanticRegistry.model_validate(payload)

    assert registry.version == 5
    assert registry.types["custom.metric"].value_shape == "number"


@pytest.mark.parametrize(
    ("raw", "message"),
    [
        ("nope", "semantic_registry must be a mapping"),
        (
            {"record": {"type": "semantic_registry"}, "semantic_registry": []},
            "semantic_registry must be a mapping",
        ),
        ({"types": []}, "semantic_registry.types must be a mapping"),
        ({"types": {}, "aliases": []}, "semantic_registry.aliases must be a mapping"),
        (
            {"types": {"bad.metric": []}, "aliases": {}},
            "semantic_registry.types.bad.metric must be a mapping",
        ),
    ],
)
def test_semantic_registry_payload_rejects_invalid_shapes(raw: Any, message: str) -> None:
    with pytest.raises(TypeError, match=message):
        semantic_registry_payload_from_yaml_view(raw)
