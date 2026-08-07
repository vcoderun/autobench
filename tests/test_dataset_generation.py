from __future__ import annotations

import json
import math
import sys
from collections.abc import Iterator
from datetime import timedelta
from pathlib import Path

import pytest
from click.testing import CliRunner
from pydantic import ValidationError
from rich.console import Console

from autobench import (
    BenchmarkInfo,
    BenchmarkSpec,
    Case,
    CaseGeneratorInput,
    DatasetSpec,
    GeneratedCaseBatch,
    GeneratedCaseRecord,
    GeneratedCaseReview,
    GenerationCost,
    GenerationDeterminism,
    GenerationError,
    GenerationResult,
    GenerationUsage,
    ReviewStatus,
    TaskSpec,
    Variant,
    benchmark_spec_to_yaml_view,
    generate_dataset,
    generate_dataset_sync,
    generated_batch_from_cases,
    generated_case_content_hash,
    generation_request_from_yaml_view,
    generation_request_hash,
    generation_request_to_yaml_view,
    generation_result_to_yaml_view,
    load_benchmark_spec,
    load_generation_request,
    resolve_case_generator,
    write_generation_result,
)
from autobench.cli import cli
from autobench.io import (
    dataset_schema,
    dump_yaml,
    generation_request_schema,
    generation_schema,
    load_yaml,
    yaml_schema,
)
from autobench.reports.rich import render_generation_result


def test_generation_models_reject_inconsistent_review_and_completion_state() -> None:
    with pytest.raises(ValidationError, match="require a rejection reason"):
        GeneratedCaseReview(case_id="rejected", status=ReviewStatus.REJECTED)
    with pytest.raises(ValidationError, match="only rejected"):
        GeneratedCaseReview(
            case_id="accepted",
            status=ReviewStatus.ACCEPTED,
            rejection_reason="not applicable",
        )
    with pytest.raises(ValidationError, match="finite number"):
        GenerationCost(amount=math.inf)

    first = Case(id="first")
    for payload, message in (
        ({"cases": [first, first]}, "case ids must be unique"),
        (
            {
                "cases": [first],
                "reviews": [
                    GeneratedCaseReview(case_id="first"),
                    GeneratedCaseReview(case_id="first"),
                ],
            },
            "reviews must be unique",
        ),
        (
            {"cases": [first], "reviews": [GeneratedCaseReview(case_id="missing")]},
            "unknown cases: missing",
        ),
        ({"cases": [first], "incomplete_reason": "unexpected"}, "cannot have"),
        ({"cases": [first], "complete": False}, "require a reason"),
    ):
        with pytest.raises(ValidationError, match=message):
            GeneratedCaseBatch.model_validate(payload)

    generated_case = Case(id="generated", metadata={"source": "synthetic"})
    content_hash = generated_case_content_hash(generated_case)
    for payload, message in (
        (
            {
                "case": generated_case,
                "review_status": ReviewStatus.REJECTED,
                "content_hash": content_hash,
            },
            "require a rejection reason",
        ),
        (
            {
                "case": generated_case,
                "review_status": ReviewStatus.ACCEPTED,
                "rejection_reason": "not applicable",
                "content_hash": content_hash,
            },
            "only rejected",
        ),
        (
            {
                "case": generated_case,
                "review_status": ReviewStatus.CANDIDATE,
                "content_hash": "0" * 64,
            },
            "content hash does not match",
        ),
    ):
        with pytest.raises(ValidationError, match=message):
            GeneratedCaseRecord.model_validate(payload)


async def test_async_generation_records_review_provenance_usage_cost_and_stable_hashes() -> None:
    request = CaseGeneratorInput(
        seed_cases=(Case(id="seed", input={"topic": "billing"}),),
        prompt="Create routing cases without personal data.",
        prompt_asset_version="prompt.generator@v4",
        seed=17,
        settings={"temperature": 0, "count": 3},
        metadata={"owner": "evaluation"},
    )

    async def generator(active: CaseGeneratorInput) -> GeneratedCaseBatch:
        assert active == request
        return GeneratedCaseBatch(
            generator_asset_version="prompt.generator@v4",
            model_provider="openrouter",
            model_name="openai/gpt-5.6-luna",
            determinism=GenerationDeterminism.GUARANTEED,
            usage=GenerationUsage(input_tokens=40, output_tokens=80, requests=1),
            cost=GenerationCost(amount=0.012, currency="usd"),
            reviews=(
                GeneratedCaseReview(case_id="accepted", status=ReviewStatus.ACCEPTED),
                GeneratedCaseReview(
                    case_id="rejected",
                    status=ReviewStatus.REJECTED,
                    rejection_reason="duplicates seed case",
                ),
            ),
            cases=(
                Case(id="accepted", input={"message": "refund"}, expected={"route": "billing"}),
                Case(id="candidate", input={"message": "login"}, expected={"route": "account"}),
                Case(id="rejected", input={"message": "refund"}),
            ),
        )

    result = await generate_dataset(
        generator,
        request,
        generator_id="tests.generator:generate",
        dataset_id="generated-routing",
        version="v1",
        metadata={"suite": "synthetic"},
    )
    repeated = await generate_dataset(
        generator,
        request,
        generator_id="tests.generator:generate",
        dataset_id="generated-routing",
        version="v1",
        metadata={"suite": "synthetic"},
    )

    assert result.batch.complete is True
    assert result.batch.usage.input_tokens == 40
    assert result.batch.cost == GenerationCost(amount=0.012)
    assert result.request_hash == generation_request_hash(request)
    assert result.request_hash == repeated.request_hash
    assert result.dataset_hash == repeated.dataset_hash
    assert result.dataset is not None
    assert [case.id for case in result.dataset.cases] == ["accepted", "candidate"]
    assert result.dataset.metadata["generation"] == {
        "generator": "tests.generator:generate",
        "request_hash": result.request_hash,
        "determinism": "guaranteed",
    }
    assert result.generated_cases[1].review_status is ReviewStatus.CANDIDATE
    assert result.generated_cases[2].rejection_reason == "duplicates seed case"
    assert all(case.case.metadata["source"] == "synthetic" for case in result.generated_cases)
    assert all(
        case.case.metadata["content_hash"] == case.content_hash for case in result.generated_cases
    )
    assert (
        generated_case_content_hash(result.generated_cases[0].case)
        == result.generated_cases[0].content_hash
    )

    manifest = generation_result_to_yaml_view(result, dataset_path="routing.yaml")
    assert manifest["generation"]["request"]["prompt"] == {
        "sha256": "9fb134053c6eae31790e0c9a15efda1087a602928ee3e4bb96e717df10b00d3e",
        "asset_version": "prompt.generator@v4",
    }
    assert "Create routing" not in dump_yaml(manifest)
    assert manifest["generation"]["output"]["rejected"] == 1
    assert manifest["generation"]["generator"]["model"] == "openai/gpt-5.6-luna"


def test_generation_request_yaml_round_trip_and_validation(tmp_path: Path) -> None:
    request = CaseGeneratorInput(
        seed_cases=(Case(id="seed", input="hello"),),
        prompt="Generate a case.",
        prompt_asset_version="prompt@1",
        seed="stable-seed",
        settings={"temperature": 0.2},
    )
    view = generation_request_to_yaml_view(request)
    path = tmp_path / "request.yaml"
    dump_yaml(view, path, schema_name="generation_request")

    assert load_generation_request(path) == request
    assert generation_request_from_yaml_view(view) == request
    assert generation_request_from_yaml_view(view["generation"]) == request
    assert generation_request_from_yaml_view(view["generation"]["request"]) == request
    assert "generation_request_schema.json" in path.read_text(encoding="utf-8").splitlines()[0]

    for raw, message in (
        ([], "must contain a mapping"),
        ({"generation": "bad"}, "generation must be a mapping"),
        ({"generation": {"request": "bad"}}, "request must be a mapping"),
        ({"generation": {"request": {"prompt": "bad"}}}, "prompt must be a mapping"),
        (
            {"generation": {"request": {"settings": {"bad": math.inf}}}},
            "Invalid generation request",
        ),
    ):
        with pytest.raises(GenerationError, match=message):
            generation_request_from_yaml_view(raw)


def test_sync_generation_resolver_and_runtime_failures(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module_path = tmp_path / "generation_plugin.py"
    module_path.write_text(
        """from autobench import Case, GeneratedCaseBatch

def generate(request):
    return GeneratedCaseBatch(cases=(Case(id='generated', input={'seed': request.seed}),))

def wrong(request):
    return ['not-a-batch']

def fail(request):
    raise RuntimeError('provider unavailable')
""",
        encoding="utf-8",
    )
    monkeypatch.syspath_prepend(str(tmp_path))
    generator = resolve_case_generator("generation_plugin:generate")
    result = generate_dataset_sync(
        generator,
        CaseGeneratorInput(seed=9),
        generator_id="generation_plugin:generate",
        dataset_id="generated",
    )
    assert result.dataset is not None
    assert result.dataset.cases[0].input == {"seed": 9}

    wrong = resolve_case_generator("generation_plugin:wrong")
    with pytest.raises(GenerationError, match="must return GeneratedCaseBatch"):
        generate_dataset_sync(
            wrong,
            CaseGeneratorInput(),
            generator_id="generation_plugin:wrong",
            dataset_id="wrong",
        )
    failing = resolve_case_generator("generation_plugin:fail")
    with pytest.raises(GenerationError, match="provider unavailable"):
        generate_dataset_sync(
            failing,
            CaseGeneratorInput(),
            generator_id="generation_plugin:fail",
            dataset_id="failed",
        )
    for metadata, message in (
        ({"review_status": "unknown"}, "invalid review_status"),
        ({"rejection_reason": 42}, "rejection_reason must be a string"),
    ):
        with pytest.raises(GenerationError, match=message):
            generate_dataset_sync(
                lambda request, metadata=metadata: GeneratedCaseBatch(
                    cases=(Case(id="invalid", metadata=metadata),)
                ),
                CaseGeneratorInput(),
                generator_id="tests:invalid-review",
                dataset_id="invalid-review",
            )

    generator_calls = 0

    def should_not_run(request: CaseGeneratorInput) -> GeneratedCaseBatch:
        nonlocal generator_calls
        generator_calls += 1
        return GeneratedCaseBatch(cases=(Case(id=str(request.seed)),))

    for generator_id, dataset_id, message in (
        (" ", "valid", "Generator id must not be blank"),
        ("valid", " ", "dataset id must not be blank"),
    ):
        with pytest.raises(GenerationError, match=message):
            generate_dataset_sync(
                should_not_run,
                CaseGeneratorInput(),
                generator_id=generator_id,
                dataset_id=dataset_id,
            )
    assert generator_calls == 0


def test_complete_and_incomplete_generation_writes_are_separate_and_atomic(tmp_path: Path) -> None:
    complete = generate_dataset_sync(
        lambda request: generated_batch_from_cases(
            [Case(id="one", input={"seed": request.seed})],
            determinism=GenerationDeterminism.NOT_GUARANTEED,
        ),
        CaseGeneratorInput(seed=3),
        generator_id="tests:complete",
        dataset_id="generated",
        version="v2",
    )
    output_path = tmp_path / "cases.yaml"
    written = write_generation_result(complete, output_path)

    assert written.dataset_path == output_path
    assert written.manifest_path == tmp_path / "cases.generation.yaml"
    assert load_yaml(output_path)["dataset"]["id"] == "generated"
    assert (
        load_yaml(written.manifest_path)["generation"]["output"]["dataset"]["path"] == "cases.yaml"
    )
    assert "dataset_schema.json" in output_path.read_text(encoding="utf-8").splitlines()[0]
    assert (
        "generation_schema.json"
        in written.manifest_path.read_text(encoding="utf-8").splitlines()[0]
    )

    with pytest.raises(GenerationError, match="already exists"):
        write_generation_result(complete, output_path)
    assert write_generation_result(complete, output_path, force=True) == written

    incomplete = generate_dataset_sync(
        lambda request: GeneratedCaseBatch(
            complete=False,
            incomplete_reason=f"budget exhausted at seed {request.seed}",
            cases=(Case(id="partial"),),
        ),
        CaseGeneratorInput(seed=4),
        generator_id="tests:incomplete",
        dataset_id="unused",
    )
    incomplete_output = tmp_path / "incomplete-cases.yaml"
    incomplete_written = write_generation_result(incomplete, incomplete_output)
    assert incomplete_written.complete is False
    assert incomplete_written.dataset_path is None
    assert not incomplete_output.exists()
    assert incomplete_written.manifest_path == tmp_path / "incomplete-cases.incomplete.yaml"
    assert load_yaml(incomplete_written.manifest_path)["generation"]["status"] == "incomplete"

    incomplete_output.write_text("existing", encoding="utf-8")
    with pytest.raises(GenerationError, match="did not replace existing dataset"):
        write_generation_result(incomplete, incomplete_output, force=True)

    invalid_complete = complete.model_copy(update={"dataset": None})
    with pytest.raises(GenerationError, match="missing its dataset"):
        write_generation_result(invalid_complete, tmp_path / "invalid.yaml")


def test_generation_result_model_guards_complete_and_incomplete_dataset_contract() -> None:
    complete = generate_dataset_sync(
        lambda request: GeneratedCaseBatch(cases=(Case(id="one"),)),
        CaseGeneratorInput(),
        generator_id="tests:result",
        dataset_id="generated",
    )
    with pytest.raises(ValidationError, match="requires a frozen dataset"):
        GenerationResult.model_validate(
            complete.model_dump(mode="python") | {"dataset": None, "dataset_hash": None}
        )
    with pytest.raises(ValidationError, match="cannot publish"):
        GenerationResult.model_validate(
            complete.model_dump(mode="python")
            | {
                "batch": complete.batch.model_copy(
                    update={"complete": False, "incomplete_reason": "stopped"}
                )
            }
        )

    payload = complete.model_dump(mode="python")
    assert complete.dataset is not None
    invalid_results = (
        (
            payload | {"completed_at": complete.started_at - timedelta(microseconds=1)},
            "completion cannot precede",
        ),
        (
            payload | {"request": CaseGeneratorInput(seed="changed")},
            "request hash does not match",
        ),
        (payload | {"generated_cases": []}, "records must match batch cases"),
        (
            payload
            | {
                "batch": complete.batch.model_dump(mode="python")
                | {"cases": [Case(id="one", input="changed")]}
            },
            "contain the batch case payloads",
        ),
        (
            payload | {"dataset": complete.dataset.model_dump(mode="python") | {"cases": []}},
            "contain every non-rejected case",
        ),
        (payload | {"dataset_hash": "0" * 64}, "dataset hash does not match"),
    )
    for invalid, message in invalid_results:
        with pytest.raises(ValidationError, match=message):
            GenerationResult.model_validate(invalid)


def test_generation_cli_publishes_complete_or_incomplete_outputs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module_path = tmp_path / "cli_generator.py"
    module_path.write_text(
        """from autobench import Case, GeneratedCaseBatch, GenerationDeterminism

def complete(request):
    return GeneratedCaseBatch(
        determinism=GenerationDeterminism.GUARANTEED,
        cases=(Case(id='cli-case', input={'seed': request.seed}),),
    )

async def incomplete(request):
    return GeneratedCaseBatch(
        complete=False,
        incomplete_reason='manual review required',
        cases=(Case(id='candidate'),),
    )
""",
        encoding="utf-8",
    )
    monkeypatch.syspath_prepend(str(tmp_path))
    request_path = tmp_path / "request.yaml"
    dump_yaml(
        generation_request_to_yaml_view(CaseGeneratorInput(seed=21)),
        request_path,
        schema_name="generation_request",
    )
    runner = CliRunner()
    output_path = tmp_path / "generated.yaml"
    completed = runner.invoke(
        cli,
        [
            "dataset",
            "generate",
            "cli_generator:complete",
            "--request",
            str(request_path),
            "--output",
            str(output_path),
            "--id",
            "cli-generated",
            "--version",
            "v1",
        ],
    )
    assert completed.exit_code == 0, completed.output
    assert "Dataset Generation Complete" in completed.output
    assert "Generated Cases" in completed.output
    assert output_path.is_file()

    benchmark_path = tmp_path / "autobench.yaml"
    benchmark = BenchmarkSpec(
        benchmark=BenchmarkInfo(id="generated-benchmark"),
        dataset=DatasetSpec(source="generated.yaml"),
        variants=[Variant(id="baseline")],
        task=TaskSpec(kind="python", target="cli_generator:complete"),
    )
    dump_yaml(benchmark_spec_to_yaml_view(benchmark), benchmark_path, schema_name="benchmark")
    loaded = load_benchmark_spec(benchmark_path)
    assert [case.id for case in loaded.dataset.cases] == ["cli-case"]

    incomplete_path = tmp_path / "incomplete.yaml"
    partial = runner.invoke(
        cli,
        [
            "dataset",
            "generate",
            "cli_generator:incomplete",
            "--output",
            str(incomplete_path),
            "--id",
            "cli-incomplete",
        ],
    )
    assert partial.exit_code == 2
    assert "Dataset Generation Incomplete" in partial.output
    assert not incomplete_path.exists()
    assert (tmp_path / "incomplete.incomplete.yaml").is_file()

    missing = runner.invoke(
        cli,
        [
            "dataset",
            "generate",
            "missing_module:generate",
            "--output",
            str(tmp_path / "missing.yaml"),
            "--id",
            "missing",
        ],
    )
    assert missing.exit_code == 1
    assert "Dataset generation failed" in missing.output


def test_generation_schemas_and_rich_empty_case_table(tmp_path: Path) -> None:
    assert yaml_schema("dataset") == dataset_schema()
    assert yaml_schema("generation") == generation_schema()
    assert yaml_schema("generation_request") == generation_request_schema()
    request_properties = generation_request_schema()["properties"]["generation"]["properties"][
        "request"
    ]["properties"]
    assert set(request_properties) == {"seed", "prompt", "settings", "metadata", "seed_cases"}
    assert generation_schema()["properties"]["generation"]["properties"]["determinism"]["enum"] == [
        "guaranteed",
        "not_guaranteed",
        "unknown",
    ]
    manifest_properties = generation_schema()["properties"]["generation"]["properties"]
    assert set(manifest_properties["generator"]["properties"]) == {
        "id",
        "asset_version",
        "provider",
        "model",
    }
    assert set(manifest_properties["request"]["properties"]) == {
        "sha256",
        "seed",
        "prompt",
        "settings",
        "metadata",
        "seed_cases",
    }
    assert set(manifest_properties["usage"]["properties"]) == {
        "input_tokens",
        "output_tokens",
        "cached_input_tokens",
        "requests",
        "metadata",
    }
    assert set(manifest_properties["output"]["properties"]) == {
        "dataset",
        "generated",
        "included",
        "rejected",
    }
    schema_root = Path(__file__).resolve().parents[1] / "schemas" / "0.3.0"
    for name, generated in (
        ("dataset", dataset_schema()),
        ("generation", generation_schema()),
        ("generation_request", generation_request_schema()),
    ):
        assert (
            json.loads((schema_root / f"{name}_schema.json").read_text(encoding="utf-8"))
            == generated
        )

    empty = generate_dataset_sync(
        lambda request: GeneratedCaseBatch(cost=GenerationCost(amount=0)),
        CaseGeneratorInput(),
        generator_id="tests:empty",
        dataset_id="empty",
    )
    written = write_generation_result(empty, tmp_path / "empty.yaml")
    console = Console(record=True, width=120)
    render_generation_result(console, empty, written)
    assert "empty" in console.export_text()


@pytest.fixture(autouse=True)
def clear_generation_modules() -> Iterator[None]:
    yield
    for module_name in ("generation_plugin", "cli_generator"):
        sys.modules.pop(module_name, None)
