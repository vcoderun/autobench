from __future__ import annotations as _annotations

from typing import Any, cast

import pytest

from autobench import (
    DEFAULT_SEMANTIC_REGISTRY,
    Semantic,
    SemanticRegistry,
    SemanticTypeInfo,
    semantic_registry_payload_from_yaml_view,
    semantic_registry_to_yaml_view,
)


def test_default_registry_normalizes_alias() -> None:
    assert DEFAULT_SEMANTIC_REGISTRY.normalize("quality.answer") == Semantic.QUALITY_SCORE


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
        }
    )

    assert registry.info_for(cast(Any, None)) is None
    assert registry.normalize("old.quality") == Semantic.QUALITY_SCORE
    assert registry.parent_of(None) is None
    assert registry.parent_of("missing.semantic") is None
    assert registry.is_a(None, Semantic.QUALITY_SCORE) is False
    assert registry.is_a("quality.child", None) is False
    assert registry.is_a("quality.child", Semantic.QUALITY_SCORE) is True


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
                aliases=["custom.alias"],
                deprecated=True,
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
        "aliases": ["custom.alias"],
        "deprecated": True,
        "tags": {"owner": "tests"},
    }
    assert view["semantic_registry"]["types"]["empty.metric"] == {}
    assert view["semantic_registry"]["aliases"] == {"legacy.metric": "custom.metric"}


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
