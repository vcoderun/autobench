from __future__ import annotations as _annotations

from pathlib import Path
from textwrap import dedent
from typing import Any, cast

import pytest

from autobench import (
    BenchmarkInfo,
    BenchmarkSpec,
    Case,
    DatasetSpec,
    Observation,
    ObservationKind,
    ObservationQuery,
    ObservationSource,
    RunContext,
    Semantic,
    TaskSpec,
    Variant,
    load_benchmark_spec,
    run_benchmark_spec,
)
from autobench.evaluation.derivation import (
    DerivedMetricOutput,
    ModelPricing,
    PricingTable,
    TokenCostDeriverSpec,
    TokenCostInputs,
    build_deriver,
    load_pricing_table,
)


def test_observation_query_supports_exact_and_parent_semantics() -> None:
    case = Case(id="case_1")
    variant = Variant(id="variant_1")
    ctx = RunContext(benchmark_id="demo", case=case, variant=variant)
    observations = [
        Observation(
            id=ctx._next_observation_id(),
            name="spec_model",
            kind=ObservationKind.FACTOR,
            semantic_type="ai.codegen.spec_model",
            value="gpt-x",
            source=ObservationSource.TASK_OBSERVATION,
            case_id=case.id,
            variant_id=variant.id,
        ),
        Observation(
            id=ctx._next_observation_id(),
            name="input_tokens",
            kind=ObservationKind.METRIC,
            semantic_type=Semantic.LLM_TOKENS_INPUT,
            value=120,
            source=ObservationSource.TASK_OBSERVATION,
            case_id=case.id,
            variant_id=variant.id,
        ),
    ]

    query = ObservationQuery(observations=observations)

    assert query.first_exact(Semantic.LLM_TOKENS_INPUT) is not None
    parent_match = query.first_related(
        Semantic.LLM_MODEL_NAME,
        kind=(ObservationKind.METRIC, ObservationKind.FACTOR),
    )
    assert parent_match is not None
    assert parent_match.semantic_type == "ai.codegen.spec_model"


async def test_token_cost_derivation_produces_money_cost(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pricing_path = _write_pricing(
        tmp_path,
        """
        provider: openai
        models:
          gpt-x:
            input_cost_per_million_tokens: 1.0
            output_cost_per_million_tokens: 2.0
        """,
    )
    _write_module(
        tmp_path,
        "derive_tasks.py",
        """
        def run(ctx, case):
            ctx.metric('input_tokens', 1000, semantic_type='llm.tokens.input')
            ctx.metric('output_tokens', 500, semantic_type='llm.tokens.output')
            ctx.factor_observation('provider', 'openai', semantic_type='llm.provider')
            ctx.factor_observation('model', 'gpt-x', semantic_type='ai.codegen.spec_model')
            return {'ok': True}
        """,
    )
    monkeypatch.syspath_prepend(str(tmp_path))
    spec = BenchmarkSpec(
        benchmark=BenchmarkInfo(id="derive-demo"),
        dataset=DatasetSpec(cases=[Case(id="case_1")]),
        task=TaskSpec(kind="python", target="derive_tasks:run"),
        variants=[Variant(id="variant_1")],
        derive=[
            TokenCostDeriverSpec(
                output=DerivedMetricOutput(
                    name="cost", semantic_type=Semantic.MONEY_COST, unit="usd"
                ),
                inputs=TokenCostInputs(),
                pricing=str(pricing_path),
            )
        ],
    )

    result = await run_benchmark_spec(spec, experiment_id="exp_fixed")

    cost = next(
        observation
        for observation in result.runs[0].task_result.observations
        if observation.semantic_type == Semantic.MONEY_COST
        and observation.source == ObservationSource.DERIVED
    )
    assert cost.value == pytest.approx(0.002)
    assert cost.tags["model_id"] == "openai/gpt-x"


async def test_token_cost_derivation_supports_normalized_model_ids_and_tiered_rates(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pricing_path = _write_pricing(
        tmp_path,
        """
        pricing:
          models:
            openai/gpt-tiered:
              input:
                unit: mtok
                price: 1.0
              output:
                unit: mtok
                tiers:
                  - up_to: 500
                    price: 4.0
                  - price: 2.0
        """,
    )
    _write_module(
        tmp_path,
        "tiered_tasks.py",
        """
        def run(ctx, case):
            ctx.metric('input_tokens', 1000, semantic_type='llm.tokens.input')
            ctx.metric('output_tokens', 400, semantic_type='llm.tokens.output')
            ctx.factor_observation('provider', 'openai', semantic_type='llm.provider')
            ctx.factor_observation('model', 'openai:gpt-tiered', semantic_type='llm.model.name')
            return {'ok': True}
        """,
    )
    monkeypatch.syspath_prepend(str(tmp_path))
    spec = _derive_spec(
        task_target="tiered_tasks:run",
        pricing_path=pricing_path,
    )

    result = await run_benchmark_spec(spec, experiment_id="exp_fixed")

    cost = next(
        observation
        for observation in result.runs[0].task_result.observations
        if observation.semantic_type == Semantic.MONEY_COST
        and observation.source == ObservationSource.DERIVED
    )
    assert cost.value == pytest.approx(0.0026)
    assert cost.tags["model_id"] == "openai/gpt-tiered"


async def test_unknown_model_does_not_invent_cost(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pricing_path = _write_pricing(
        tmp_path,
        """
        provider: openai
        models:
          gpt-known:
            input_cost_per_million_tokens: 1.0
            output_cost_per_million_tokens: 2.0
        """,
    )
    _write_module(
        tmp_path,
        "unknown_price_tasks.py",
        """
        def run(ctx, case):
            ctx.metric('input_tokens', 1000, semantic_type='llm.tokens.input')
            ctx.metric('output_tokens', 500, semantic_type='llm.tokens.output')
            ctx.factor_observation('provider', 'openai', semantic_type='llm.provider')
            ctx.factor_observation('model', 'gpt-unknown', semantic_type='llm.model.name')
            return {'ok': True}
        """,
    )
    monkeypatch.syspath_prepend(str(tmp_path))
    spec = _derive_spec(
        task_target="unknown_price_tasks:run",
        pricing_path=pricing_path,
    )

    result = await run_benchmark_spec(spec, experiment_id="exp_fixed")

    assert not any(
        observation.semantic_type == Semantic.MONEY_COST
        and observation.source == ObservationSource.DERIVED
        for observation in result.runs[0].task_result.observations
    )
    diagnostic = next(
        observation
        for observation in result.runs[0].task_result.observations
        if observation.name == "token_cost_unknown_pricing"
    )
    assert diagnostic.source == ObservationSource.DERIVED


async def test_missing_pricing_rates_emit_diagnostic(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pricing_path = _write_pricing(
        tmp_path,
        """
        pricing:
          models:
            openai/gpt-bad:
              input:
                unit: mtok
                price: 1.0
        """,
    )
    _write_module(
        tmp_path,
        "missing_rate_tasks.py",
        """
        def run(ctx, case):
            ctx.metric('input_tokens', 1000, semantic_type='llm.tokens.input')
            ctx.metric('output_tokens', 500, semantic_type='llm.tokens.output')
            ctx.factor_observation('provider', 'openai', semantic_type='llm.provider')
            ctx.factor_observation('model', 'gpt-bad', semantic_type='llm.model.name')
            return {'ok': True}
        """,
    )
    monkeypatch.syspath_prepend(str(tmp_path))
    spec = _derive_spec(
        task_target="missing_rate_tasks:run",
        pricing_path=pricing_path,
    )

    result = await run_benchmark_spec(spec, experiment_id="exp_fixed")

    diagnostic = next(
        observation
        for observation in result.runs[0].task_result.observations
        if observation.name == "token_cost_missing_rates"
    )
    assert diagnostic.source == ObservationSource.DERIVED
    assert diagnostic.tags["model_id"] == "openai/gpt-bad"


async def test_missing_inputs_emit_diagnostic(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pricing_path = _write_pricing(
        tmp_path,
        """
        provider: openai
        models:
          gpt-x:
            input_cost_per_million_tokens: 1.0
            output_cost_per_million_tokens: 2.0
        """,
    )
    _write_module(
        tmp_path,
        "missing_input_tasks.py",
        """
        def run(ctx, case):
            ctx.metric('input_tokens', 1000, semantic_type='llm.tokens.input')
            ctx.factor_observation('provider', 'openai', semantic_type='llm.provider')
            return {'ok': True}
        """,
    )
    monkeypatch.syspath_prepend(str(tmp_path))
    spec = _derive_spec(
        task_target="missing_input_tasks:run",
        pricing_path=pricing_path,
    )

    result = await run_benchmark_spec(spec, experiment_id="exp_fixed")

    diagnostic = next(
        observation
        for observation in result.runs[0].task_result.observations
        if observation.name == "token_cost_missing_inputs"
    )
    assert diagnostic.source == ObservationSource.DERIVED
    assert diagnostic.tags["missing"] == ["output_tokens", "model"]


async def test_user_and_derived_costs_both_remain_in_raw_evidence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pricing_path = _write_pricing(
        tmp_path,
        """
        provider: openai
        models:
          gpt-x:
            input_cost_per_million_tokens: 1.0
            output_cost_per_million_tokens: 2.0
        """,
    )
    _write_module(
        tmp_path,
        "user_cost_tasks.py",
        """
        def run(ctx, case):
            ctx.metric('input_tokens', 1000, semantic_type='llm.tokens.input')
            ctx.metric('output_tokens', 500, semantic_type='llm.tokens.output')
            ctx.factor_observation('provider', 'openai', semantic_type='llm.provider')
            ctx.factor_observation('model', 'gpt-x', semantic_type='llm.model.name')
            ctx.metric('cost', 9.99, semantic_type='money.cost')
            return {'ok': True}
        """,
    )
    monkeypatch.syspath_prepend(str(tmp_path))
    spec = _derive_spec(
        task_target="user_cost_tasks:run",
        pricing_path=pricing_path,
    )

    result = await run_benchmark_spec(spec, experiment_id="exp_fixed")

    costs = [
        observation
        for observation in result.runs[0].task_result.observations
        if observation.semantic_type == Semantic.MONEY_COST
    ]
    assert len(costs) == 2
    assert {observation.source for observation in costs} == {
        ObservationSource.TASK_OBSERVATION,
        ObservationSource.DERIVED,
    }


def test_load_benchmark_spec_resolves_relative_pricing_file(tmp_path: Path) -> None:
    pricing_dir = tmp_path / "pricing"
    pricing_dir.mkdir()
    pricing_path = pricing_dir / "models.yaml"
    pricing_path.write_text(
        dedent(
            """
            provider: openai
            models:
              gpt-x:
                input_cost_per_million_tokens: 1.0
                output_cost_per_million_tokens: 2.0
            """
        ).strip()
        + "\n",
        encoding="utf-8",
    )
    spec_path = tmp_path / "autobench.yaml"
    spec_path.write_text(
        dedent(
            """
            benchmark:
              id: derived-pricing
            dataset:
              cases:
                - id: case_1
            task:
              kind: python
              target: app.tasks.run_demo
            variants:
              - id: variant_1
                factors: []
            derive:
              - kind: token_cost
                pricing: file://pricing/models.yaml
            """
        ).strip()
        + "\n",
        encoding="utf-8",
    )

    spec = load_benchmark_spec(spec_path)

    assert spec.derive[0].pricing == str(pricing_path.resolve())


def test_pricing_table_rejects_wrong_provider_and_bad_yaml(tmp_path: Path) -> None:
    table = PricingTable(
        provider="openai",
        models={
            "gpt-x": ModelPricing(
                input_cost_per_million_tokens=1.0,
                output_cost_per_million_tokens=2.0,
            )
        },
    )
    empty_path = tmp_path / "empty.yaml"
    empty_path.write_text("", encoding="utf-8")
    invalid_path = tmp_path / "invalid.yaml"
    invalid_path.write_text("42\n", encoding="utf-8")

    assert table.model_pricing(provider="anthropic", model="gpt-x") is None
    assert load_pricing_table(empty_path).models == {}
    with pytest.raises(ValueError, match="Expected pricing YAML mapping"):
        load_pricing_table(invalid_path)


def test_build_deriver_rejects_unsupported_specs() -> None:
    with pytest.raises(TypeError, match="Unsupported deriver spec"):
        build_deriver(cast(Any, object()))


def _derive_spec(*, task_target: str, pricing_path: Path) -> BenchmarkSpec:
    return BenchmarkSpec(
        benchmark=BenchmarkInfo(id="derive-demo"),
        dataset=DatasetSpec(cases=[Case(id="case_1")]),
        task=TaskSpec(kind="python", target=task_target),
        variants=[Variant(id="variant_1")],
        derive=[
            TokenCostDeriverSpec(
                output=DerivedMetricOutput(
                    name="cost", semantic_type=Semantic.MONEY_COST, unit="usd"
                ),
                inputs=TokenCostInputs(),
                pricing=str(pricing_path),
            )
        ],
    )


def _write_module(tmp_path: Path, filename: str, source: str) -> None:
    (tmp_path / filename).write_text(dedent(source).strip() + "\n", encoding="utf-8")


def _write_pricing(tmp_path: Path, source: str) -> Path:
    path = tmp_path / "pricing.yaml"
    path.write_text(dedent(source).strip() + "\n", encoding="utf-8")
    return path
