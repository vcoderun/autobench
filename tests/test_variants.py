from __future__ import annotations as _annotations

from autobench import FactorValue, Semantic, normalize_variant_factors


def test_normalize_variant_factors_accepts_none_lists_and_dicts() -> None:
    existing = FactorValue(
        name="model",
        value="gpt-x",
        semantic_type=Semantic.LLM_MODEL_NAME,
        optimize=True,
    )

    assert normalize_variant_factors(None) == []
    assert normalize_variant_factors([existing]) == [existing]
    assert normalize_variant_factors([{"name": "temperature", "value": 0.2}]) == [
        FactorValue(name="temperature", value=0.2)
    ]
    assert normalize_variant_factors(
        {
            "provider": {
                "value": "openai",
                "semantic_type": Semantic.LLM_PROVIDER,
            },
            "retries": 2,
        }
    ) == [
        FactorValue(
            name="provider",
            value="openai",
            semantic_type=Semantic.LLM_PROVIDER,
        ),
        FactorValue(name="retries", value=2),
    ]
