from __future__ import annotations as _annotations

import json
from pathlib import Path
from types import TracebackType
from typing import Any

import pytest

from autobench import (
    GenAIPricesSource,
    LLMPricesSource,
    ModelPricing,
    PricingTable,
    StaticPriceSource,
    TokenPriceTier,
    dump_pricing_table,
    load_pricing_table,
    pricing_table_to_yaml_view,
)
from autobench.evaluation.pricing import (
    TokenPrice,
    _assign_model_pricing,
    _canonical_model_id,
    _default_rate,
    _iter_mappings,
    _iter_pricing_entries,
    _load_pricing_mapping,
    _load_token_price,
    _model_lookup_candidates,
    _model_pricing_yaml_view,
    _optional_float,
    _optional_text,
    _parse_model_pricing,
    _price_block_from_genai,
    _pricing_matches_provider,
    _required_float,
    _required_mapping_list,
    _required_text,
    _resolve_token_rate,
    _split_model_key,
    _token_price_yaml_view,
)
from autobench.io import load_yaml


def test_static_pricing_table_supports_legacy_models_provider_maps_and_aliases() -> None:
    table = PricingTable(
        models={
            "gpt-primary": ModelPricing(
                model_id="openai/gpt-primary",
                input_cost_per_million_tokens=1.0,
                output_cost_per_million_tokens=2.0,
                aliases=("gpt-alias",),
            )
        },
        providers={
            "openai": {
                "gpt-provider": ModelPricing(
                    input_cost_per_million_tokens=3.0,
                    output_cost_per_million_tokens=4.0,
                )
            }
        },
    )

    assert StaticPriceSource(table).pricing_table() == table
    assert table.model_pricing(provider="openai", model="gpt-primary") is not None
    assert table.model_pricing(provider="openai", model="gpt-alias") is not None
    assert table.model_pricing(provider="openai", model="openai:gpt-primary") is not None
    assert table.model_pricing(provider="openai", model="openai/gpt-primary") is not None
    assert table.model_pricing(provider="openai", model="gpt-provider") is not None
    assert table.model_pricing(provider="anthropic", model="gpt-primary") is None
    assert table.model_pricing(provider="openai", model="missing") is None
    resolved = table.resolve_model_pricing(provider="openai", model="openai:gpt-primary")
    assert resolved == (
        "openai/gpt-primary",
        table.models["gpt-primary"],
    )


def test_llm_prices_source_maps_current_v1_shape_and_rejects_bad_inputs(tmp_path: Path) -> None:
    payload = {
        "updated_at": "2026-04-24",
        "prices": [
            {
                "id": "gpt-demo",
                "vendor": "openai",
                "name": "GPT Demo",
                "input": 1,
                "output": 2.5,
                "input_cached": 0.25,
            },
            "ignored",
        ],
    }
    path = tmp_path / "llm-prices.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    table = LLMPricesSource.from_json_file(path).pricing_table()
    pricing = table.model_pricing(provider="openai", model="gpt-demo")

    assert table.source == "llm-prices"
    assert table.updated_at == "2026-04-24"
    assert pricing is not None
    assert pricing.model_id == "openai/gpt-demo"
    assert pricing.input_cost_per_million_tokens == 1.0
    assert pricing.output_cost_per_million_tokens == 2.5
    assert pricing.cache_read_cost_per_million_tokens == 0.25
    with pytest.raises(ValueError, match="llm-prices JSON must contain an object"):
        LLMPricesSource.from_json_file(_write_json(tmp_path, "bad-llm.json", []))
    with pytest.raises(ValueError, match="vendor"):
        LLMPricesSource({"prices": [{"id": "bad", "input": 1, "output": 2}]}).pricing_table()
    with pytest.raises(ValueError, match="input"):
        LLMPricesSource(
            {"prices": [{"id": "bad", "vendor": "openai", "input": None, "output": 2}]}
        ).pricing_table()


def test_genai_prices_source_maps_provider_data_and_price_variants(tmp_path: Path) -> None:
    payload = [
        {
            "id": "anthropic",
            "models": [
                {
                    "id": "claude-demo",
                    "name": "Claude Demo",
                    "context_window": 1000,
                    "prices": {
                        "input_mtok": 3,
                        "cache_read_mtok": {"base": 0.3},
                        "cache_write_mtok": {"base": 3.75},
                        "output_mtok": 15,
                    },
                },
                {
                    "id": "claude-tiered",
                    "prices": [
                        {
                            "match": {"equals": "default"},
                            "prices": {"input_mtok": {"base": 1.5}, "output_mtok": {"base": 6}},
                        }
                    ],
                },
                {"id": "empty-list", "prices": [{"prices": []}, "ignored"]},
                {"id": "no-prices"},
                {"id": "missing-output", "prices": {"input_mtok": 1}},
            ],
        }
    ]
    path = _write_json(tmp_path, "genai.json", payload)

    table = GenAIPricesSource.from_json_file(path).pricing_table()
    direct = table.model_pricing(provider="anthropic", model="claude-demo")
    tiered = table.model_pricing(provider="anthropic", model="claude-tiered")

    assert table.source == "genai-prices"
    assert direct is not None
    assert direct.model_id == "anthropic/claude-demo"
    assert direct.name == "Claude Demo"
    assert direct.cache_read_cost_per_million_tokens == 0.3
    assert direct.cache_write_cost_per_million_tokens == 3.75
    assert direct.metadata == {"context_window": 1000}
    assert tiered is not None
    assert tiered.model_id == "anthropic/claude-tiered"
    assert tiered.input_cost_per_million_tokens == 1.5
    assert table.model_pricing(provider="anthropic", model="empty-list") is None
    assert table.model_pricing(provider="anthropic", model="no-prices") is None
    assert table.model_pricing(provider="anthropic", model="missing-output") is None
    with pytest.raises(ValueError, match="list of objects"):
        GenAIPricesSource.from_json_file(_write_json(tmp_path, "bad-genai.json", {}))


def test_load_and_dump_pricing_table_support_all_yaml_shapes(tmp_path: Path) -> None:
    legacy_path = tmp_path / "legacy.yaml"
    legacy_path.write_text(
        "\n".join(
            (
                "provider: openai",
                "models:",
                "  gpt-demo:",
                "    input_cost_per_million_tokens: 1.0",
                "    output_cost_per_million_tokens: 2.0",
            )
        ),
        encoding="utf-8",
    )
    pricing_path = tmp_path / "pricing.yaml"
    rendered = dump_pricing_table(load_pricing_table(legacy_path), pricing_path)
    llm_shape_path = tmp_path / "llm-shape.yaml"
    llm_shape_path.write_text(
        "\n".join(
            (
                "prices:",
                "  - id: gpt-yaml",
                "    vendor: openai",
                "    input: 1",
                "    output: 2",
            )
        ),
        encoding="utf-8",
    )
    genai_shape_path = tmp_path / "genai-shape.yaml"
    genai_shape_path.write_text(
        "\n".join(
            (
                "- id: openai",
                "  models:",
                "    - id: gpt-genai",
                "      prices:",
                "        input_mtok: 1",
                "        output_mtok: 2",
            )
        ),
        encoding="utf-8",
    )
    invalid_path = tmp_path / "invalid.yaml"
    invalid_path.write_text("42\n", encoding="utf-8")
    empty_path = tmp_path / "empty.yaml"
    empty_path.write_text("", encoding="utf-8")

    assert rendered.startswith("# yaml-language-server: $schema=")
    assert (
        pricing_table_to_yaml_view(load_pricing_table(pricing_path))["record"]["type"] == "pricing"
    )
    assert (
        load_pricing_table(llm_shape_path).model_pricing(provider="openai", model="gpt-yaml")
        is not None
    )
    assert (
        load_pricing_table(genai_shape_path).model_pricing(provider="openai", model="gpt-genai")
        is not None
    )
    assert load_pricing_table(empty_path).models == {}
    with pytest.raises(ValueError, match="Expected pricing YAML mapping"):
        load_pricing_table(invalid_path)


def test_load_and_dump_pricing_table_supports_dsl_models_tiers_and_normalized_ids(
    tmp_path: Path,
) -> None:
    pricing_path = tmp_path / "pricing-dsl.yaml"
    pricing_path.write_text(
        "\n".join(
            (
                "pricing:",
                "  source: manual",
                "  updated_at: 2026-05-07",
                "  models:",
                "    openai/gpt-dsl:",
                "      aliases:",
                "        - gpt-dsl",
                "        - openai:gpt-dsl",
                "      input:",
                "        unit: mtok",
                "        price: 1.0",
                "      output:",
                "        unit: mtok",
                "        tiers:",
                "          - up_to: 1000",
                "            price: 4.0",
                "          - price: 2.0",
                "      cache_read:",
                "        unit: mtok",
                "        price: 0.25",
            )
        ),
        encoding="utf-8",
    )

    table = load_pricing_table(pricing_path)
    pricing = table.model_pricing(provider="openai", model="gpt-dsl")

    assert pricing is not None
    assert pricing.model_id == "openai/gpt-dsl"
    assert pricing.output_pricing is not None
    assert pricing.output_pricing.tiers == (
        TokenPriceTier(up_to_tokens=1000, price_per_million_tokens=4.0),
        TokenPriceTier(up_to_tokens=None, price_per_million_tokens=2.0),
    )
    assert pricing.output_rate_for_tokens(500) == 4.0
    assert pricing.output_rate_for_tokens(10_000) == 2.0

    rendered = dump_pricing_table(table, tmp_path / "dumped-pricing.yaml")
    dumped = load_yaml(tmp_path / "dumped-pricing.yaml")
    view = pricing_table_to_yaml_view(table)

    assert rendered.startswith("# yaml-language-server: $schema=")
    assert view["pricing"]["updated_at"] == "2026-05-07"
    assert dumped["pricing"]["models"]["openai/gpt-dsl"]["output"]["tiers"] == [
        {"up_to": 1000, "price": 4.0},
        {"price": 2.0},
    ]
    assert dumped["pricing"]["models"]["openai/gpt-dsl"]["input"] == {
        "unit": "mtok",
        "price": 1.0,
    }


def test_price_sources_can_load_from_url(monkeypatch: pytest.MonkeyPatch) -> None:
    llm_payload = {"prices": [{"id": "gpt-url", "vendor": "openai", "input": 1, "output": 2}]}
    genai_payload = [
        {
            "id": "openai",
            "models": [{"id": "gpt-url", "prices": {"input_mtok": 1, "output_mtok": 2}}],
        }
    ]

    monkeypatch.setattr(
        "autobench.evaluation.pricing.urlopen", lambda url, timeout: _FakeResponse(llm_payload)
    )
    assert (
        LLMPricesSource.from_url("https://example.test/llm.json").pricing_table().source
        == "llm-prices"
    )
    monkeypatch.setattr(
        "autobench.evaluation.pricing.urlopen", lambda url, timeout: _FakeResponse(genai_payload)
    )
    assert (
        GenAIPricesSource.from_url("https://example.test/genai.json").pricing_table().source
        == "genai-prices"
    )
    monkeypatch.setattr(
        "autobench.evaluation.pricing.urlopen", lambda url, timeout: _FakeResponse([])
    )
    with pytest.raises(ValueError, match="llm-prices JSON must contain an object"):
        LLMPricesSource.from_url("https://example.test/bad.json")


def test_pricing_private_helpers_cover_branchy_paths() -> None:
    tier_only = TokenPrice(
        tiers=(
            TokenPriceTier(up_to_tokens=10, price_per_million_tokens=5.0),
            TokenPriceTier(up_to_tokens=None, price_per_million_tokens=2.0),
        )
    )
    tier_no_fallback = TokenPrice(
        tiers=(TokenPriceTier(up_to_tokens=10, price_per_million_tokens=5.0),)
    )
    canonical_pricing = ModelPricing(
        model_id="openai/gpt-helper",
        input_pricing=TokenPrice(price_per_million_tokens=1.0),
        output_pricing=TokenPrice(price_per_million_tokens=2.0),
        cache_read_pricing=TokenPrice(price_per_million_tokens=0.1),
        cache_write_pricing=TokenPrice(price_per_million_tokens=0.2),
        aliases=("gpt-helper",),
        source="manual",
        metadata={"owner": "tests"},
    )

    assert tier_only.rate_for_tokens(5) == 5.0
    assert tier_only.rate_for_tokens(50) == 2.0
    assert tier_no_fallback.rate_for_tokens(50) is None
    assert canonical_pricing.cache_read_rate_for_tokens(10) == 0.1
    assert canonical_pricing.cache_write_rate_for_tokens(10) == 0.2
    assert _resolve_token_rate(None, 3.0, 10) == 3.0
    assert _default_rate(tier_only) == 2.0
    assert _default_rate(TokenPrice()) is None
    assert _price_block_from_genai("bad") is None
    genai_price_block = _price_block_from_genai({"base": 2})
    assert genai_price_block is not None
    assert genai_price_block.price_per_million_tokens == 2.0
    assert _iter_mappings("bad") == []
    assert _optional_text(1) is None
    assert _optional_float("bad") is None
    assert _canonical_model_id("", "gpt") == "gpt"
    assert _pricing_matches_provider(ModelPricing(), "openai") is True
    assert _pricing_matches_provider(ModelPricing(model_id="openai:gpt"), "openai") is True
    assert _model_lookup_candidates(provider="openai", model="openai/gpt") == (
        "openai/gpt",
        "gpt",
        "openai:gpt",
    )
    assert _model_lookup_candidates(provider="openai", model="openai:gpt") == (
        "openai:gpt",
        "gpt",
        "openai/gpt",
    )
    assert _model_lookup_candidates(provider="openai", model="other/gpt") == (
        "other/gpt",
        "openai/other/gpt",
        "openai:other/gpt",
    )
    assert _model_lookup_candidates(provider="openai", model="other:gpt") == (
        "other:gpt",
        "openai/other:gpt",
        "openai:other:gpt",
    )
    assert _split_model_key("anthropic:claude", default_provider=None) == (
        "anthropic",
        "claude",
        "anthropic/claude",
    )
    assert _split_model_key("gpt", default_provider=None) == (None, "gpt", "gpt")

    legacy_pricing = _parse_model_pricing(
        {
            "input_cost_per_million_tokens": 1.0,
            "output_cost_per_million_tokens": 2.0,
            "cache_read_cost_per_million_tokens": 0.3,
            "cache_write_cost_per_million_tokens": 0.4,
            "aliases": ["a", 1],
            "metadata": {"owner": "tests"},
        },
        model_id="openai/gpt-legacy",
    )
    assert legacy_pricing.cache_read_pricing is not None
    assert legacy_pricing.cache_write_pricing is not None
    assert legacy_pricing.aliases == ("a",)
    assert legacy_pricing.metadata == {"owner": "tests"}

    assert _load_token_price(1.5) == TokenPrice(price_per_million_tokens=1.5)
    assert _load_token_price("bad") is None
    assert _load_token_price({"unit": "tokens"}) == TokenPrice(unit="mtok")
    parsed_tiers = _load_token_price(
        {
            "tiers": [
                "bad",
                {"up_to": 25, "price": 5},
                {"up_to": 50},
            ]
        }
    )
    assert parsed_tiers == TokenPrice(
        tiers=(TokenPriceTier(up_to_tokens=25, price_per_million_tokens=5.0),)
    )
    assert _token_price_yaml_view(None, None) is None
    assert _token_price_yaml_view(TokenPrice(price_per_million_tokens=1.0), None) == {
        "unit": "mtok",
        "price": 1.0,
    }
    named_view = _model_pricing_yaml_view(canonical_pricing.model_copy(update={"name": "Helper"}))
    assert named_view == {
        "name": "Helper",
        "aliases": ["gpt-helper"],
        "source": "manual",
        "input": {"unit": "mtok", "price": 1.0},
        "output": {"unit": "mtok", "price": 2.0},
        "cache_read": {"unit": "mtok", "price": 0.1},
        "cache_write": {"unit": "mtok", "price": 0.2},
        "metadata": {"owner": "tests"},
    }
    assert _model_pricing_yaml_view(ModelPricing()) == {}

    table = _load_pricing_mapping(
        {
            "source": "manual",
            "updated_at": "2026-05-07",
            "models": {
                "gpt-inline": {"input": {"price": 1.0}, "output": {"price": 2.0}},
                1: {"ignored": True},
                "skip": [],
            },
            "providers": {
                "anthropic": {
                    "claude": {"input": {"price": 3.0}, "output": {"price": 6.0}},
                    2: {"ignored": True},
                    "skip": [],
                },
                3: {},
            },
        }
    )
    assert table.provider is None
    assert table.source == "manual"
    assert table.updated_at == "2026-05-07"
    assert table.model_pricing(provider="openai", model="gpt-inline") is not None
    assert table.model_pricing(provider="anthropic", model="claude") is not None
    entries = _iter_pricing_entries(table)
    assert {model_id for model_id, _ in entries} == {"anthropic/claude", "gpt-inline"}

    direct_table = PricingTable()
    _assign_model_pricing(
        direct_table,
        model_key="standalone",
        payload={"input": {"price": 1.0}, "output": {"price": 2.0}, "name": "Standalone"},
        default_provider=None,
    )
    assert direct_table.models["standalone"].name == "Standalone"
    resolved_table = PricingTable(
        models={
            "different-key": ModelPricing(
                model_id="openai/gpt-resolved",
                input_cost_per_million_tokens=1.0,
                output_cost_per_million_tokens=2.0,
            )
        }
    )
    assert resolved_table.resolve_model_pricing(provider="openai", model="openai/gpt-resolved") == (
        "openai/gpt-resolved",
        resolved_table.models["different-key"],
    )

    with pytest.raises(ValueError, match="list of objects"):
        _required_mapping_list({}, "bad")
    with pytest.raises(ValueError, match="text field 'id'"):
        _required_text({}, "id")
    with pytest.raises(ValueError, match="numeric field 'input'"):
        _required_float({"input": None}, "input")


def _write_json(tmp_path: Path, name: str, payload: Any) -> Path:
    path = tmp_path / name
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


class _FakeResponse:
    def __init__(self, payload: Any) -> None:
        self._payload = payload

    def __enter__(self) -> _FakeResponse:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        return None

    def read(self) -> bytes:
        return json.dumps(self._payload).encode("utf-8")
