from __future__ import annotations as _annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path
from textwrap import dedent
from typing import Any, Literal

import pytest
from pydantic import BaseModel, Field

from autobench import (
    AssetVersion,
    BenchContext,
    Benchmark,
    BenchmarkInfo,
    BenchmarkSpec,
    Case,
    DatasetSpec,
    ExactScorer,
    FactorValue,
    FieldAsset,
    ParamAsset,
    PassFailScorer,
    PydanticEvalsBridge,
    PydanticEvalsUnavailableError,
    RunContext,
    Semantic,
    TaskSpec,
    ToolAsset,
    TrackingRegistry,
    TypeAsset,
    Variant,
    dataset_content_hash,
    run_benchmark_spec,
    track,
)
from autobench.metrics.observations import ObservationSource

CarMake = Literal["audi", "bmw", "mercedes"]


def test_benchmark_builder_compiles_to_spec_and_runs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _write_module(
        tmp_path,
        "builder_tasks.py",
        """
        def run(ctx, case):
            return {
                "success": ctx.factor("enabled"),
                "answer": case.expected["answer"],
            }
        """,
    )
    monkeypatch.syspath_prepend(str(tmp_path))

    result = (
        Benchmark("builder-demo")
        .description("Programmatic builder smoke test.")
        .dataset(
            [Case(id="case_1", expected={"answer": "ok"})],
            dataset_id="builder-cases",
            version="v1",
            metadata={"owner": "tests"},
        )
        .variants(
            [
                Variant(
                    id="variant_1",
                    factors=[FactorValue(name="enabled", value=True)],
                ),
                {
                    "id": "variant_2",
                    "factors": {"enabled": True},
                },
            ]
        )
        .task(TaskSpec(kind="python", target="builder_tasks:run"))
        .scoring(
            [
                PassFailScorer(
                    name="success",
                    path="output.success",
                    semantic_type=Semantic.RESULT_SUCCESS,
                ),
                ExactScorer(
                    name="answer",
                    actual="output.answer",
                    expected="case.expected.answer",
                    semantic_type=Semantic.QUALITY_CORRECTNESS,
                ),
            ]
        )
        .derive([])
        .run(experiment_id="exp_builder")
    )

    assert result.total_count == 2
    assert result.plan.dataset_id == "builder-cases"
    assert result.plan.dataset_version == "v1"
    assert result.plan.case_ids == ("case_1",)
    assert result.plan.dataset_hash is not None
    assert result.passed_count == 2


def test_benchmark_builder_accepts_source_and_dict_defaults(tmp_path: Path) -> None:
    dataset_path = tmp_path / "cases.yaml"
    dataset_path.write_text(
        dedent(
            """
            cases:
              - id: case_1
                input:
                  value: 1
            """
        ).strip()
        + "\n",
        encoding="utf-8",
    )
    benchmark = (
        Benchmark("source-builder")
        .dataset(
            source=dataset_path,
            case_defaults={
                "expected": {"success": True},
                "tags": ["default"],
            },
        )
        .task("tests.test_components_tracking_and_runtime:_noop_task")
        .variants([{"id": "variant_1"}])
        .scoring([])
        .derive([])
    )

    spec = benchmark.to_spec()

    assert spec.dataset.source == str(dataset_path)
    assert spec.dataset.case_defaults.expected == {"success": True}


def test_bench_context_set_get_contract() -> None:
    ctx = BenchContext()

    ctx.set("result", {"ok": True})

    assert ctx.get("result") == {"ok": True}


def test_noop_task_reports_case_and_variant_ids() -> None:
    case = Case(id="case_1")
    variant = Variant(id="variant_1")
    ctx = RunContext(benchmark_id="benchmark_1", case=case, variant=variant)

    assert _noop_task(ctx, case) == {"case_id": "case_1", "variant_id": "variant_1"}


def test_tracking_registry_tracks_prompts_functions_and_manual_versions() -> None:
    registry = TrackingRegistry()

    @registry.asset(kind="tool", name="lookup_order", semantic_type=Semantic.AGENT_TOOL_VERSION)
    def lookup_order(order_id: str) -> dict[str, str]:
        return {"order_id": order_id}

    prompt = registry.prompt(name="refund_prompt", text="Refund policy text")
    manual_target = object()
    decorated_manual = registry.asset(
        kind="config",
        name="manual",
        version="v2",
        hash="abc123",
        source_path="settings.yaml",
        parent_version="v1",
        metadata={"owner": "tests"},
    )(manual_target)
    opaque_target = object()
    decorated_opaque = registry.asset(kind="config", name="opaque")(opaque_target)

    assert lookup_order("ord_1") == {"order_id": "ord_1"}
    assert registry.asset_of(lookup_order).id == "tool.lookup_order"
    assert len(registry.version_of(lookup_order)) == 12
    assert str(prompt) == "Refund policy text"
    assert registry.asset_of(prompt).semantic_type == Semantic.PROMPT_VERSION
    assert (
        registry.asset_version_of(prompt).content_hash
        != registry.asset_version_of(lookup_order).content_hash
    )
    assert decorated_manual is manual_target
    assert registry.asset_version_of(manual_target) == AssetVersion(
        asset_id="config.manual",
        version="v2",
        content_hash="abc123",
        source_path="settings.yaml",
        parent_version="v1",
        metadata={"owner": "tests"},
    )
    assert decorated_opaque is opaque_target
    assert registry.asset_version_of(opaque_target).source_path is None

    with pytest.raises(KeyError, match="not tracked"):
        registry.asset_of(object())
    with pytest.raises(KeyError, match="not tracked"):
        registry.version_of(object())


def test_tracking_registry_tool_and_type_assets_capture_structured_schema() -> None:
    registry = TrackingRegistry()

    class Car(BaseModel):
        make: CarMake
        model: str = Field(..., examples=["a3", "a4", "a5", "320i", "520i"])
        year: int = Field(..., gt=0)

    registry.type(Car)

    @registry.tool
    def create_car(make: CarMake, model: str, year: int) -> Car:
        """
        Create a new car instance.
        """
        return Car(make=make, model=model, year=year)

    car_asset = registry.assets["Car"]
    create_car_asset = registry.assets["create_car"]

    assert isinstance(car_asset, TypeAsset)
    assert car_asset.type_kind == "pydantic_model"
    assert car_asset.id == "type.Car"
    model_field = next(
        field_asset for field_asset in car_asset.field_assets if field_asset.name == "model"
    )
    year_field = next(
        field_asset for field_asset in car_asset.field_assets if field_asset.name == "year"
    )
    make_field = next(
        field_asset for field_asset in car_asset.field_assets if field_asset.name == "make"
    )
    assert model_field.examples == ("a3", "a4", "a5", "320i", "520i")
    assert year_field.constraints["exclusiveMinimum"] == 0
    assert make_field.literal_choices == ("audi", "bmw", "mercedes")

    assert isinstance(create_car_asset, ToolAsset)
    assert create_car_asset.doc == "Create a new car instance."
    assert create_car_asset.return_type_name == "Car"
    assert create_car_asset.return_type_asset_id == "type.Car"
    assert create_car_asset.param_schema.params[0] == ParamAsset(
        name="make",
        annotation="Literal['audi', 'bmw', 'mercedes']",
        required=True,
        kind="positional",
        literal_choices=("audi", "bmw", "mercedes"),
    )
    assert create_car_asset.param_schema.params[1:] == (
        ParamAsset(name="model", annotation="str", required=True, kind="positional"),
        ParamAsset(name="year", annotation="int", required=True, kind="positional"),
    )
    car_version = registry.asset_version_of(Car)
    tool_version = registry.asset_version_of(create_car)
    assert car_version.source_hash is not None
    assert tool_version.source_hash is not None
    assert car_version.content_hash != car_version.source_hash
    assert tool_version.content_hash != tool_version.source_hash


def test_tracking_registry_supports_dataclass_and_typed_class_assets() -> None:
    registry = TrackingRegistry()

    @registry.type
    @dataclass
    class Payload:
        order_id: str
        quantity: int = field(default=1, metadata={"unit": "count"})

    @registry.type
    class Result:
        ok: bool
        reason: str = "accepted"

    payload_asset = registry.asset_of(Payload)
    result_asset = registry.asset_of(Result)

    assert isinstance(payload_asset, TypeAsset)
    assert payload_asset.type_kind == "dataclass"
    assert payload_asset.field_assets == (
        FieldAsset(
            name="order_id",
            annotation="str",
            required=True,
            kw_only=False,
            compare=True,
            repr=True,
            init=True,
        ),
        FieldAsset(
            name="quantity",
            annotation="int",
            required=False,
            default=1,
            metadata={"unit": "count"},
            kw_only=False,
            compare=True,
            repr=True,
            init=True,
        ),
    )
    assert isinstance(result_asset, TypeAsset)
    assert result_asset.type_kind == "typed_class"
    assert result_asset.field_assets == (
        FieldAsset(name="ok", annotation="bool", required=True),
        FieldAsset(name="reason", annotation="str", required=False, default="accepted"),
    )
    assert (
        registry.asset_version_of(Payload).content_hash
        != registry.asset_version_of(Result).content_hash
    )


def test_global_track_namespace_tracks_assets() -> None:
    prompt = track.prompt(name="global_prompt", text="hello", version="v1")

    assert track.version_of(prompt) == "v1"


async def test_tracked_variant_assets_flow_into_run_evidence() -> None:
    prompt = track.prompt(name="global_prompt_evidence", text="hello", version="v7")
    spec = BenchmarkSpec(
        benchmark=BenchmarkInfo(id="tracked-evidence"),
        dataset=DatasetSpec(cases=[Case(id="case_1")]),
        task=TaskSpec(
            kind="python", target="tests.test_components_tracking_and_runtime:_noop_task"
        ),
        variants=[
            Variant(
                id="variant_1",
                factors=[
                    FactorValue(
                        name="prompt",
                        value=prompt,
                        semantic_type=Semantic.PROMPT_VERSION,
                    )
                ],
            )
        ],
    )

    result = await run_benchmark_spec(spec, experiment_id="exp_tracking")
    run = result.runs[0]

    assert run.asset_versions[0].asset_id == "prompt.global_prompt_evidence"
    observation = next(item for item in run.task_result.observations if item.name == "prompt")
    assert observation.source == ObservationSource.VARIANT
    assert observation.value == "v7"


def test_run_context_attach_tracked_asset_deduplicates_versions() -> None:
    prompt = track.prompt(name="dedupe_prompt", text="hello", version="v1")
    ctx = RunContext(
        benchmark_id="benchmark_1",
        case=Case(id="case_1"),
        variant=Variant(id="variant_1"),
    )

    first = ctx.attach_tracked_asset(prompt)
    second = ctx.attach_tracked_asset(prompt)

    assert first == second
    assert len(ctx.asset_versions) == 1


def test_pydantic_evals_runtime_payloads_and_optional_import() -> None:
    runtime = PydanticEvalsBridge(module_name="sys")
    spec = (
        Benchmark("runtime-demo")
        .dataset(
            [
                {
                    "id": "case_1",
                    "input": {"message": "hi"},
                    "expected": {"answer": "hello"},
                    "metadata": {"difficulty": "easy"},
                    "tags": ["greeting"],
                }
            ],
            dataset_id="runtime-cases",
        )
        .to_spec()
    )

    payload = runtime.dataset_payload(spec)

    assert runtime.is_available() is True
    assert runtime.require_module() is sys
    assert payload.name == "runtime-cases"
    assert payload.cases[0].metadata == {"tags": ["greeting"], "difficulty": "easy"}

    fallback_payload = runtime.dataset_payload(Benchmark("runtime-fallback").dataset([]).to_spec())
    unavailable = PydanticEvalsBridge(module_name="missing_autobench_runtime_module")
    assert fallback_payload.name == "runtime-fallback"
    assert unavailable.is_available() is False
    with pytest.raises(PydanticEvalsUnavailableError, match="not installed"):
        unavailable.require_module()


def test_dataset_content_hash_ignores_source_but_includes_content() -> None:
    first = DatasetSpec(
        source="cases/*.yaml",
        cases=[Case(id="case_1", input={"value": 1})],
    )
    second = first.model_copy(update={"source": "other/*.yaml"})
    changed = DatasetSpec(cases=[Case(id="case_1", input={"value": 2})])

    assert dataset_content_hash(first) == dataset_content_hash(second)
    assert dataset_content_hash(first) != dataset_content_hash(changed)


def _noop_task(ctx: Any, case: Case) -> dict[str, str]:
    return {"case_id": case.id, "variant_id": ctx.variant.id}


def _write_module(tmp_path: Path, filename: str, source: str) -> None:
    (tmp_path / filename).write_text(dedent(source).strip() + "\n", encoding="utf-8")
