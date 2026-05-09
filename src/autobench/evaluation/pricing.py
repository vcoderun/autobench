from __future__ import annotations as _annotations

import json
from datetime import date, datetime
from pathlib import Path
from typing import Any, Literal, Protocol
from urllib.request import urlopen

from pydantic import BaseModel, Field

from autobench.io import dump_yaml, load_yaml


class TokenPriceTier(BaseModel):
    up_to_tokens: int | None = Field(default=None, ge=1)
    price_per_million_tokens: float


class TokenPrice(BaseModel):
    unit: Literal["mtok"] = "mtok"
    price_per_million_tokens: float | None = None
    tiers: tuple[TokenPriceTier, ...] = ()

    def rate_for_tokens(self, tokens: float) -> float | None:
        for tier in self.tiers:
            if tier.up_to_tokens is None or tokens <= float(tier.up_to_tokens):
                return tier.price_per_million_tokens
        return self.price_per_million_tokens


class ModelPricing(BaseModel):
    model_id: str | None = None
    input_cost_per_million_tokens: float | None = None
    output_cost_per_million_tokens: float | None = None
    cache_read_cost_per_million_tokens: float | None = None
    cache_write_cost_per_million_tokens: float | None = None
    input_pricing: TokenPrice | None = None
    output_pricing: TokenPrice | None = None
    cache_read_pricing: TokenPrice | None = None
    cache_write_pricing: TokenPrice | None = None
    name: str | None = None
    aliases: tuple[str, ...] = ()
    source: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    def input_rate_for_tokens(self, tokens: float) -> float | None:
        return _resolve_token_rate(self.input_pricing, self.input_cost_per_million_tokens, tokens)

    def output_rate_for_tokens(self, tokens: float) -> float | None:
        return _resolve_token_rate(self.output_pricing, self.output_cost_per_million_tokens, tokens)

    def cache_read_rate_for_tokens(self, tokens: float) -> float | None:
        return _resolve_token_rate(
            self.cache_read_pricing,
            self.cache_read_cost_per_million_tokens,
            tokens,
        )

    def cache_write_rate_for_tokens(self, tokens: float) -> float | None:
        return _resolve_token_rate(
            self.cache_write_pricing,
            self.cache_write_cost_per_million_tokens,
            tokens,
        )


class PricingTable(BaseModel):
    provider: str | None = None
    source: str | None = None
    updated_at: str | None = None
    models: dict[str, ModelPricing] = Field(default_factory=dict)
    providers: dict[str, dict[str, ModelPricing]] = Field(default_factory=dict)

    def model_pricing(self, *, provider: str, model: str) -> ModelPricing | None:
        resolved = self.resolve_model_pricing(provider=provider, model=model)
        return resolved[1] if resolved is not None else None

    def resolve_model_pricing(
        self, *, provider: str, model: str
    ) -> tuple[str, ModelPricing] | None:
        if self.provider is not None and self.provider != provider:
            return None
        candidates = _model_lookup_candidates(provider=provider, model=model)
        provider_models = self.providers.get(provider, {})
        for candidate in candidates:
            direct = self.models.get(candidate)
            if direct is not None and _pricing_matches_provider(direct, provider):
                return direct.model_id or candidate, direct
            direct = provider_models.get(candidate)
            if direct is not None:
                return direct.model_id or _canonical_model_id(provider, candidate), direct
        for resolved_id, pricing in _iter_pricing_entries(self):
            if not _pricing_matches_provider(pricing, provider):
                continue
            if resolved_id in candidates:
                return resolved_id, pricing
            aliases = set(pricing.aliases)
            if pricing.model_id is not None:
                aliases.add(pricing.model_id)
            if aliases.intersection(candidates):
                return resolved_id, pricing
        return None


class PriceSource(Protocol):  # pragma: no cover
    def pricing_table(self) -> PricingTable: ...


class StaticPriceSource:
    def __init__(self, table: PricingTable) -> None:
        self._table = table

    def pricing_table(self) -> PricingTable:
        return self._table


class LLMPricesSource:
    def __init__(self, data: dict[str, Any]) -> None:
        self._data = data

    @classmethod
    def from_json_file(cls, path: Path) -> LLMPricesSource:
        raw = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            raise ValueError("llm-prices JSON must contain an object.")
        return cls(raw)

    @classmethod
    def from_url(cls, url: str) -> LLMPricesSource:
        with urlopen(url, timeout=30) as response:
            raw = json.loads(response.read().decode("utf-8"))
        if not isinstance(raw, dict):
            raise ValueError("llm-prices JSON must contain an object.")
        return cls(raw)

    def pricing_table(self) -> PricingTable:
        providers: dict[str, dict[str, ModelPricing]] = {}
        for entry in _iter_mappings(self._data.get("prices")):
            vendor = _required_text(entry, "vendor")
            model_id = _required_text(entry, "id")
            provider_models = providers.setdefault(vendor, {})
            provider_models[model_id] = ModelPricing(
                model_id=_canonical_model_id(vendor, model_id),
                input_cost_per_million_tokens=_required_float(entry, "input"),
                output_cost_per_million_tokens=_required_float(entry, "output"),
                cache_read_cost_per_million_tokens=_optional_float(entry.get("input_cached")),
                name=_optional_text(entry.get("name")),
                source="llm-prices",
            )
        return PricingTable(
            source="llm-prices",
            updated_at=_optional_text(self._data.get("updated_at")),
            providers=providers,
        )


class GenAIPricesSource:
    def __init__(self, data: list[dict[str, Any]]) -> None:
        self._data = data

    @classmethod
    def from_json_file(cls, path: Path) -> GenAIPricesSource:
        raw = json.loads(path.read_text(encoding="utf-8"))
        return cls(_required_mapping_list(raw, "genai-prices JSON"))

    @classmethod
    def from_url(cls, url: str) -> GenAIPricesSource:
        with urlopen(url, timeout=30) as response:
            raw = json.loads(response.read().decode("utf-8"))
        return cls(_required_mapping_list(raw, "genai-prices JSON"))

    def pricing_table(self) -> PricingTable:
        providers: dict[str, dict[str, ModelPricing]] = {}
        for provider_entry in self._data:
            provider_id = _required_text(provider_entry, "id")
            provider_models = providers.setdefault(provider_id, {})
            for model_entry in _iter_mappings(provider_entry.get("models")):
                pricing = _genai_model_pricing(model_entry)
                if pricing is not None:
                    model_id = _required_text(model_entry, "id")
                    provider_models[model_id] = pricing.model_copy(
                        update={"model_id": _canonical_model_id(provider_id, model_id)}
                    )
        return PricingTable(source="genai-prices", providers=providers)


def load_pricing_table(path: Path) -> PricingTable:
    raw = load_yaml(path)
    if raw is None:
        raw = {}
    if isinstance(raw, list):
        return GenAIPricesSource(_required_mapping_list(raw, str(path))).pricing_table()
    if not isinstance(raw, dict):
        raise ValueError(f"Expected pricing YAML mapping in {path}.")
    if isinstance(raw.get("prices"), list):
        return LLMPricesSource(raw).pricing_table()
    if isinstance(raw.get("pricing"), dict):
        raw = raw["pricing"]
    return _load_pricing_mapping(raw)


def pricing_table_to_yaml_view(table: PricingTable) -> dict[str, Any]:
    pricing_view: dict[str, Any] = {}
    if table.provider is not None:
        pricing_view["provider"] = table.provider
    if table.source is not None:
        pricing_view["source"] = table.source
    if table.updated_at is not None:
        pricing_view["updated_at"] = table.updated_at
    pricing_view["models"] = {
        model_id: _model_pricing_yaml_view(pricing)
        for model_id, pricing in _iter_pricing_entries(table)
    }
    return {
        "record": {
            "type": "pricing",
            "version": 1,
        },
        "pricing": pricing_view,
    }


def dump_pricing_table(table: PricingTable, path: Path) -> str:
    return dump_yaml(pricing_table_to_yaml_view(table), path, schema_name="pricing")


def _genai_model_pricing(model_entry: dict[str, Any]) -> ModelPricing | None:
    prices = _genai_prices_payload(model_entry.get("prices"))
    input_pricing = _price_block_from_genai(prices.get("input_mtok"))
    output_pricing = _price_block_from_genai(prices.get("output_mtok"))
    input_price = _default_rate(input_pricing)
    output_price = _default_rate(output_pricing)
    if input_price is None or output_price is None:
        return None
    return ModelPricing(
        input_cost_per_million_tokens=input_price,
        output_cost_per_million_tokens=output_price,
        cache_read_cost_per_million_tokens=_default_rate(
            _price_block_from_genai(prices.get("cache_read_mtok"))
        ),
        cache_write_cost_per_million_tokens=_default_rate(
            _price_block_from_genai(prices.get("cache_write_mtok"))
        ),
        input_pricing=input_pricing,
        output_pricing=output_pricing,
        cache_read_pricing=_price_block_from_genai(prices.get("cache_read_mtok")),
        cache_write_pricing=_price_block_from_genai(prices.get("cache_write_mtok")),
        name=_optional_text(model_entry.get("name")),
        source="genai-prices",
        metadata={
            "context_window": model_entry["context_window"],
        }
        if "context_window" in model_entry
        else {},
    )


def _genai_prices_payload(raw_prices: Any) -> dict[str, Any]:
    if isinstance(raw_prices, dict):
        return raw_prices
    if isinstance(raw_prices, list):
        for candidate in raw_prices:
            if isinstance(candidate, dict) and isinstance(candidate.get("prices"), dict):
                return candidate["prices"]
    return {}


def _price_block_from_genai(raw_price: Any) -> TokenPrice | None:
    if isinstance(raw_price, int | float):
        return TokenPrice(price_per_million_tokens=float(raw_price))
    if isinstance(raw_price, dict) and isinstance(raw_price.get("base"), int | float):
        return TokenPrice(price_per_million_tokens=float(raw_price["base"]))
    return None


def _iter_mappings(raw_items: Any) -> list[dict[str, Any]]:
    return (
        [item for item in raw_items if isinstance(item, dict)]
        if isinstance(raw_items, list)
        else []
    )


def _required_mapping_list(raw: Any, label: str) -> list[dict[str, Any]]:
    if not isinstance(raw, list) or not all(isinstance(item, dict) for item in raw):
        raise ValueError(f"{label} must contain a list of objects.")
    return raw


def _required_text(data: dict[str, Any], key: str) -> str:
    value = data.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"Pricing entry is missing text field '{key}'.")
    return value


def _optional_text(value: Any) -> str | None:
    if isinstance(value, date | datetime):
        return value.isoformat()
    return value if isinstance(value, str) else None


def _required_float(data: dict[str, Any], key: str) -> float:
    value = data.get(key)
    if not isinstance(value, int | float):
        raise ValueError(f"Pricing entry is missing numeric field '{key}'.")
    return float(value)


def _optional_float(value: Any) -> float | None:
    return float(value) if isinstance(value, int | float) else None


def _resolve_token_rate(
    pricing: TokenPrice | None,
    flat_price: float | None,
    tokens: float,
) -> float | None:
    if pricing is not None:
        return pricing.rate_for_tokens(tokens)
    return flat_price


def _default_rate(pricing: TokenPrice | None) -> float | None:
    if pricing is None:
        return None
    if pricing.price_per_million_tokens is not None:
        return pricing.price_per_million_tokens
    if pricing.tiers:
        return pricing.tiers[-1].price_per_million_tokens
    return None


def _model_lookup_candidates(*, provider: str, model: str) -> tuple[str, ...]:
    candidates: list[str] = [model]
    if model.startswith(f"{provider}/"):
        candidates.append(model.removeprefix(f"{provider}/"))
        candidates.append(model.replace("/", ":", 1))
    elif model.startswith(f"{provider}:"):
        candidates.append(model.removeprefix(f"{provider}:"))
        candidates.append(model.replace(":", "/", 1))
    else:
        candidates.append(_canonical_model_id(provider, model))
        candidates.append(f"{provider}:{model}")
    if ":" in model:
        prefix, suffix = model.split(":", 1)
        if prefix == provider:
            candidates.append(suffix)
            candidates.append(_canonical_model_id(provider, suffix))
    if "/" in model:
        prefix, suffix = model.split("/", 1)
        if prefix == provider:
            candidates.append(suffix)
            candidates.append(f"{provider}:{suffix}")
    return tuple(dict.fromkeys(candidate for candidate in candidates if candidate))


def _canonical_model_id(provider: str, model: str) -> str:
    return f"{provider}/{model}" if provider else model


def _pricing_matches_provider(pricing: ModelPricing, provider: str) -> bool:
    if pricing.model_id is None:
        return True
    if "/" in pricing.model_id:
        pricing_provider, _ = pricing.model_id.split("/", 1)
        return pricing_provider == provider
    return True


def _iter_pricing_entries(table: PricingTable) -> list[tuple[str, ModelPricing]]:
    entries: list[tuple[str, ModelPricing]] = []
    for model_id, pricing in sorted(table.models.items()):
        entries.append(
            (pricing.model_id or _canonical_model_id(table.provider or "", model_id), pricing)
        )
    for provider, models in sorted(table.providers.items()):
        for model_id, pricing in sorted(models.items()):
            entries.append((pricing.model_id or _canonical_model_id(provider, model_id), pricing))
    return entries


def _model_pricing_yaml_view(pricing: ModelPricing) -> dict[str, Any]:
    view: dict[str, Any] = {}
    if pricing.name is not None:
        view["name"] = pricing.name
    if pricing.aliases:
        view["aliases"] = list(pricing.aliases)
    if pricing.source is not None:
        view["source"] = pricing.source
    input_view = _token_price_yaml_view(
        pricing.input_pricing,
        pricing.input_cost_per_million_tokens,
    )
    output_view = _token_price_yaml_view(
        pricing.output_pricing,
        pricing.output_cost_per_million_tokens,
    )
    if input_view is not None:
        view["input"] = input_view
    if output_view is not None:
        view["output"] = output_view
    cache_read_view = _token_price_yaml_view(
        pricing.cache_read_pricing,
        pricing.cache_read_cost_per_million_tokens,
    )
    cache_write_view = _token_price_yaml_view(
        pricing.cache_write_pricing,
        pricing.cache_write_cost_per_million_tokens,
    )
    if cache_read_view is not None:
        view["cache_read"] = cache_read_view
    if cache_write_view is not None:
        view["cache_write"] = cache_write_view
    if pricing.metadata:
        view["metadata"] = pricing.metadata
    return view


def _token_price_yaml_view(
    pricing: TokenPrice | None,
    flat_price: float | None,
) -> dict[str, Any] | None:
    if pricing is None and flat_price is None:
        return None
    effective = pricing or TokenPrice(price_per_million_tokens=flat_price)
    view: dict[str, Any] = {"unit": effective.unit}
    if effective.price_per_million_tokens is not None:
        view["price"] = effective.price_per_million_tokens
    if effective.tiers:
        view["tiers"] = [
            {
                **({"up_to": tier.up_to_tokens} if tier.up_to_tokens is not None else {}),
                "price": tier.price_per_million_tokens,
            }
            for tier in effective.tiers
        ]
    return view


def _load_pricing_mapping(raw: dict[str, Any]) -> PricingTable:
    table = PricingTable(
        provider=_optional_text(raw.get("provider")),
        source=_optional_text(raw.get("source")),
        updated_at=_optional_text(raw.get("updated_at")),
    )
    raw_models = raw.get("models")
    if isinstance(raw_models, dict):
        for model_key, model_value in raw_models.items():
            if not isinstance(model_key, str) or not isinstance(model_value, dict):
                continue
            _assign_model_pricing(
                table, model_key=model_key, payload=model_value, default_provider=table.provider
            )
    raw_providers = raw.get("providers")
    if isinstance(raw_providers, dict):
        for provider_key, models_value in raw_providers.items():
            if not isinstance(provider_key, str) or not isinstance(models_value, dict):
                continue
            for model_key, model_value in models_value.items():
                if not isinstance(model_key, str) or not isinstance(model_value, dict):
                    continue
                _assign_model_pricing(
                    table,
                    model_key=model_key,
                    payload=model_value,
                    default_provider=provider_key,
                )
    return table


def _assign_model_pricing(
    table: PricingTable,
    *,
    model_key: str,
    payload: dict[str, Any],
    default_provider: str | None,
) -> None:
    provider_id, local_model_id, canonical_model_id = _split_model_key(
        model_key,
        default_provider=default_provider,
    )
    pricing = _parse_model_pricing(payload, model_id=canonical_model_id)
    if provider_id is None:
        table.models[local_model_id] = pricing
        return
    table.providers.setdefault(provider_id, {})[local_model_id] = pricing


def _split_model_key(
    model_key: str, *, default_provider: str | None
) -> tuple[str | None, str, str]:
    if "/" in model_key:
        provider_id, local_model_id = model_key.split("/", 1)
        return provider_id, local_model_id, model_key
    if ":" in model_key:
        provider_id, local_model_id = model_key.split(":", 1)
        return provider_id, local_model_id, _canonical_model_id(provider_id, local_model_id)
    if default_provider is None:
        return None, model_key, model_key
    return default_provider, model_key, _canonical_model_id(default_provider, model_key)


def _parse_model_pricing(payload: dict[str, Any], *, model_id: str) -> ModelPricing:
    input_pricing = _load_token_price(payload.get("input"))
    output_pricing = _load_token_price(payload.get("output"))
    cache_read_pricing = _load_token_price(payload.get("cache_read"))
    cache_write_pricing = _load_token_price(payload.get("cache_write"))
    if input_pricing is None and "input_cost_per_million_tokens" in payload:
        input_pricing = TokenPrice(
            price_per_million_tokens=_optional_float(payload.get("input_cost_per_million_tokens"))
        )
    if output_pricing is None and "output_cost_per_million_tokens" in payload:
        output_pricing = TokenPrice(
            price_per_million_tokens=_optional_float(payload.get("output_cost_per_million_tokens"))
        )
    if cache_read_pricing is None and "cache_read_cost_per_million_tokens" in payload:
        cache_read_pricing = TokenPrice(
            price_per_million_tokens=_optional_float(
                payload.get("cache_read_cost_per_million_tokens")
            )
        )
    if cache_write_pricing is None and "cache_write_cost_per_million_tokens" in payload:
        cache_write_pricing = TokenPrice(
            price_per_million_tokens=_optional_float(
                payload.get("cache_write_cost_per_million_tokens")
            )
        )
    aliases = payload.get("aliases")
    metadata = payload.get("metadata")
    return ModelPricing(
        model_id=model_id,
        input_cost_per_million_tokens=_default_rate(input_pricing),
        output_cost_per_million_tokens=_default_rate(output_pricing),
        cache_read_cost_per_million_tokens=_default_rate(cache_read_pricing),
        cache_write_cost_per_million_tokens=_default_rate(cache_write_pricing),
        input_pricing=input_pricing,
        output_pricing=output_pricing,
        cache_read_pricing=cache_read_pricing,
        cache_write_pricing=cache_write_pricing,
        name=_optional_text(payload.get("name")),
        aliases=tuple(alias for alias in aliases if isinstance(alias, str))
        if isinstance(aliases, list)
        else (),
        source=_optional_text(payload.get("source")),
        metadata=dict(metadata) if isinstance(metadata, dict) else {},
    )


def _load_token_price(raw: Any) -> TokenPrice | None:
    if raw is None:
        return None
    if isinstance(raw, int | float):
        return TokenPrice(price_per_million_tokens=float(raw))
    if not isinstance(raw, dict):
        return None
    tiers_raw = raw.get("tiers")
    tiers: list[TokenPriceTier] = []
    if isinstance(tiers_raw, list):
        for entry in tiers_raw:
            if not isinstance(entry, dict):
                continue
            price = _optional_float(entry.get("price"))
            if price is None:
                continue
            up_to = entry.get("up_to")
            tiers.append(
                TokenPriceTier(
                    up_to_tokens=int(up_to) if isinstance(up_to, int | float) else None,
                    price_per_million_tokens=price,
                )
            )
    unit = raw.get("unit")
    return TokenPrice(
        unit=unit if unit == "mtok" else "mtok",
        price_per_million_tokens=_optional_float(raw.get("price")),
        tiers=tuple(tiers),
    )


__all__ = (
    "GenAIPricesSource",
    "LLMPricesSource",
    "ModelPricing",
    "PriceSource",
    "PricingTable",
    "StaticPriceSource",
    "TokenPrice",
    "TokenPriceTier",
    "dump_pricing_table",
    "load_pricing_table",
    "pricing_table_to_yaml_view",
)
