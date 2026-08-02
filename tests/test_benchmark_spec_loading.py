from __future__ import annotations as _annotations

from pathlib import Path

import pytest

from autobench import (
    ArtifactRef,
    BenchmarkInfo,
    BenchmarkSpec,
    Case,
    CaseDefaults,
    CaseMatrixReportSpec,
    ComparisonReportSpec,
    DatasetSpec,
    DerivedMetricOutput,
    DistributionReportSpec,
    FactorValue,
    LeaderboardReportSpec,
    MetricAggregation,
    ObservationRole,
    PairedBaselineDeriverSpec,
    PolicySpec,
    PythonScorer,
    ReportSpec,
    SchemaScorer,
    Semantic,
    SemanticRegistry,
    SemanticTypeInfo,
    TaskSpec,
    TokenCostDeriverSpec,
    Variant,
    benchmark_spec_payload_from_yaml_view,
    benchmark_spec_to_yaml_view,
    build_benchmark_plan,
    collect_benchmark_source_files,
    dataset_to_yaml_view,
    load_benchmark_spec,
    merge_case_defaults,
)
from autobench.errors import SpecValidationError
from autobench.spec.spec import _artifact_ref_to_yaml_view


def test_merge_case_defaults_merges_mappings_and_deduplicates_tags() -> None:
    defaults = CaseDefaults(
        input={"message": "hi", "config": {"retry": 1, "temperature": 0.1}},
        expected={"status": "ok", "scores": {"coverage": 1.0}},
        metadata={"difficulty": "easy"},
        tags=["base", "shared"],
        attachments=[
            ArtifactRef(id="default_attachment", name="default.txt", value="default"),
        ],
    )
    case = Case(
        id="case-1",
        input={"config": {"temperature": 0.2}, "payload": "x"},
        expected={"scores": {"quality": 0.9}},
        metadata={"suite": "smoke"},
        tags=["shared", "smoke"],
        attachments=[ArtifactRef(id="case_attachment", name="case.txt", value="case")],
    )

    merged = merge_case_defaults(case, defaults)

    assert merged.input == {
        "message": "hi",
        "config": {"retry": 1, "temperature": 0.2},
        "payload": "x",
    }
    assert merged.expected == {
        "status": "ok",
        "scores": {"coverage": 1.0, "quality": 0.9},
    }
    assert merged.metadata == {"difficulty": "easy", "suite": "smoke"}
    assert merged.tags == ["base", "shared", "smoke"]
    assert [attachment.id for attachment in merged.attachments] == [
        "default_attachment",
        "case_attachment",
    ]


def test_merge_case_defaults_overrides_scalar_payloads() -> None:
    merged = merge_case_defaults(
        Case(id="case_1", input="override", expected=2),
        CaseDefaults(input="default", expected=1),
    )

    assert merged.input == "override"
    assert merged.expected == 2


def test_load_benchmark_spec_supports_inline_dataset_and_variant_dict_factors(
    tmp_path: Path,
) -> None:
    path = tmp_path / "autobench.yaml"
    path.write_text(
        "\n".join(
            (
                "benchmark:",
                "  id: demo",
                "dataset:",
                "  case_defaults:",
                "    metadata:",
                "      suite: smoke",
                "    tags: [shared]",
                "  cases:",
                "    - id: case_1",
                "      input:",
                "        message: hi",
                "task:",
                "  kind: python",
                "  target: app.tasks.run_demo",
                "variants:",
                "  - id: model_pair_1",
                "    factors:",
                "      spec_model:",
                "        value: openrouter:google/gemini-3-flash-preview",
                "        semantic_type: ai.codegen.spec_model",
                "      exploration_model:",
                "        value: openrouter:google/gemini-3-flash-preview",
                "        semantic_type: ai.codegen.exploration_model",
            )
        ),
        encoding="utf-8",
    )

    spec = load_benchmark_spec(path)

    assert spec.benchmark.id == "demo"
    assert spec.task is not None
    assert spec.task.target == "app.tasks.run_demo"
    assert spec.dataset.cases[0].metadata == {"suite": "smoke"}
    assert spec.dataset.cases[0].tags == ["shared"]
    assert spec.variants[0].id == "model_pair_1"
    assert spec.variants[0].factors[0].semantic_type == "ai.codegen.spec_model"
    assert spec.semantic_registry.is_a("ai.codegen.spec_model", Semantic.LLM_MODEL_NAME)


def test_load_benchmark_spec_supports_authoring_dsl(tmp_path: Path) -> None:
    datasets_dir = tmp_path / "datasets"
    datasets_dir.mkdir()
    dataset_path = datasets_dir / "cases.yaml"
    dataset_path.write_text(
        "\n".join(
            (
                "cases:",
                "  - id: ticket_1",
                "    input:",
                "      subject: Refund",
                "    expected:",
                "      queue: billing",
            )
        ),
        encoding="utf-8",
    )
    path = tmp_path / "autobench.yaml"
    path.write_text(
        "\n".join(
            (
                "benchmark:",
                "  ticket-routing:",
                "    description: Route support tickets.",
                "    cases: datasets/cases.yaml",
                "    run:",
                "      python: app.benchmarks.support:run_ticket_case",
                "    variants:",
                "      route_v1:",
                "        label: baseline keyword router",
                "        factors:",
                "          retry_budget:",
                "            value: 2",
                "          prompt_version:",
                "            value: route-v1",
                "            semantic: prompt.version",
                "          routing_profile: baseline",
                "    score:",
                "      success:",
                "        pass: output.matched",
                "        semantic: result.success",
                "        goal: maximize",
                "        role: objective",
                "      routing_correctness:",
                "        exact:",
                "          actual: output.queue",
                "          expected: case.expected.queue",
                "        semantic: quality.correctness",
                "        goal: maximize",
                "        role: objective",
                "    report:",
                "      leaderboard:",
                "        show:",
                "          pass_rate:",
                "            metric: result.success",
                "            aggregate: ratio_true",
                "      matrix:",
                "        metric: quality.correctness",
                "      compare:",
                "        route_v1 -> route_v2:",
                "          show:",
                "            avg_correctness:",
                "              metric: quality.correctness",
                "              aggregate: mean",
            )
        ),
        encoding="utf-8",
    )

    spec = load_benchmark_spec(path)

    assert spec.benchmark.id == "ticket-routing"
    assert spec.benchmark.description == "Route support tickets."
    assert spec.dataset.source == "datasets/cases.yaml"
    assert spec.dataset.cases[0].id == "ticket_1"
    assert spec.task is not None
    assert spec.task.kind == "python"
    assert spec.task.target == "app.benchmarks.support:run_ticket_case"
    assert spec.variants[0].id == "route_v1"
    assert spec.variants[0].label == "baseline keyword router"
    assert spec.variants[0].factors[0].name == "retry_budget"
    assert spec.variants[0].factors[1].semantic_type == Semantic.PROMPT_VERSION
    assert spec.variants[0].factors[2].value == "baseline"
    assert spec.scoring[0].kind == "pass_fail"
    assert spec.scoring[0].semantic_type == Semantic.RESULT_SUCCESS
    assert spec.scoring[0].direction is not None
    assert spec.scoring[0].direction.value == "maximize"
    assert spec.scoring[1].kind == "exact"
    assert spec.reports.leaderboard.metrics[0].semantic_type == Semantic.RESULT_SUCCESS
    assert spec.reports.leaderboard.metrics[0].fn == "ratio_true"
    assert spec.reports.case_matrix.semantic_type == Semantic.QUALITY_CORRECTNESS
    assert spec.reports.comparisons[0].baseline == "route_v1"
    assert spec.reports.comparisons[0].candidate == "route_v2"
    assert spec.reports.comparisons[0].metrics[0].name == "avg_correctness"
    assert dataset_path.resolve() in collect_benchmark_source_files(path)


def test_benchmark_spec_yaml_view_round_trips_to_internal_payload(tmp_path: Path) -> None:
    dataset_path = tmp_path / "cases.yaml"
    dataset_path.write_text(
        "\n".join(
            (
                "cases:",
                "  - id: ticket_1",
                "    input:",
                "      subject: Refund",
                "    expected:",
                "      queue: billing",
            )
        ),
        encoding="utf-8",
    )
    path = tmp_path / "autobench.yaml"
    path.write_text(
        "\n".join(
            (
                "benchmark:",
                "  support-routing:",
                "    description: Route support tickets.",
                "    cases: cases.yaml",
                "    run:",
                "      python: app.benchmarks.support:run_ticket_case",
                "    variants:",
                "      route_v1:",
                "        factors:",
                "          prompt_version:",
                "            value: route-v1",
                "            semantic: prompt.version",
                "            optimize: true",
                "          routing_profile: baseline",
                "    score:",
                "      success:",
                "        pass: output.matched",
                "        semantic: result.success",
                "        goal: maximize",
                "      routing_correctness:",
                "        exact:",
                "          actual: output.queue",
                "          expected: case.expected.queue",
                "        semantic: quality.correctness",
                "        goal: maximize",
                "    report:",
                "      leaderboard:",
                "        show:",
                "          pass_rate:",
                "            metric: result.success",
                "            aggregate: ratio_true",
                "      compare:",
                "        route_v1 -> route_v2:",
                "          show:",
                "            correctness:",
                "              metric: quality.correctness",
                "              aggregate: mean",
                "    semantic_registry:",
                "      aliases:",
                "        quality.answer: quality.score",
                "      types:",
                "        support.priority:",
                "          parent: quality.score",
                "          shape: string",
            )
        ),
        encoding="utf-8",
    )

    spec = load_benchmark_spec(path)
    view = benchmark_spec_to_yaml_view(spec)
    payload = benchmark_spec_payload_from_yaml_view(view)

    assert view["benchmark"]["support-routing"]["run"] == {
        "python": "app.benchmarks.support:run_ticket_case"
    }
    assert view["benchmark"]["support-routing"]["dataset"]["cases"][0]["id"] == "ticket_1"
    assert view["benchmark"]["support-routing"]["variants"]["route_v1"]["factors"] == {
        "prompt_version": {
            "value": "route-v1",
            "semantic": "prompt.version",
            "optimize": True,
        },
        "routing_profile": "baseline",
    }
    assert view["benchmark"]["support-routing"]["score"]["success"] == {
        "pass": "output.matched",
        "semantic": "result.success",
        "goal": "maximize",
    }
    assert view["benchmark"]["support-routing"]["report"]["compare"] == {
        "route_v1 -> route_v2": {
            "show": {
                "correctness": {
                    "metric": "quality.correctness",
                    "aggregate": "mean",
                }
            }
        }
    }
    assert view["benchmark"]["support-routing"]["semantic_registry"] == {
        "types": {
            "support.priority": {
                "parent": "quality.score",
                "shape": "string",
            }
        },
    }
    assert BenchmarkSpec.model_validate(payload) == spec
    assert payload == spec.model_dump(mode="json")


def test_benchmark_spec_yaml_view_covers_exporter_optional_branches() -> None:
    registry = SemanticRegistry.with_defaults()
    custom_registry = registry.model_copy(
        update={
            "version": 2,
            "aliases": dict(registry.aliases) | {"custom.metric.alias": "custom.metric"},
            "types": dict(registry.types)
            | {
                "custom.metric": SemanticTypeInfo(
                    id="custom.metric",
                    parent="quality.score",
                    unit="ms",
                    value_shape="number",
                    aliases=["custom.metric.v1"],
                    deprecated=True,
                    tags={"owner": "bench"},
                ),
                "custom.unit_only": SemanticTypeInfo(
                    id="custom.unit_only",
                    unit="usd",
                ),
            },
        }
    )
    spec = BenchmarkSpec(
        benchmark=BenchmarkInfo(id="export-branches"),
        dataset=DatasetSpec(
            id="cases",
            source="cases.yaml",
            version="v2",
            metadata={"owner": "tests"},
            case_defaults=CaseDefaults(
                input={"channel": "chat"},
                expected={"mode": "strict"},
                metadata={"suite": "smoke"},
                tags=["golden"],
                attachments=[
                    ArtifactRef(
                        id="default",
                        name="default.txt",
                        media_type="text/plain",
                        value="seed",
                        span_id="span-default",
                        tags={"role": "seed"},
                    )
                ],
            ),
            cases=[Case(id="case_1")],
        ),
        task=TaskSpec(
            kind="python",
            target="pkg.tasks:run",
            module_search_paths=("src",),
        ),
        variants=[
            Variant(
                id="candidate",
                label="candidate label",
                factors=[
                    FactorValue(name="model", value="demo-model", semantic_type="llm.model.name"),
                    FactorValue(name="tunable", value=2, optimize=True),
                ],
            )
        ],
        scoring=[
            SchemaScorer(
                name="schema_score",
                semantic_type="quality.score",
                path="output.payload",
                schema={"type": "object"},
                unit="points",
                optional=True,
            ),
            SchemaScorer(
                name="schema_default_path",
                semantic_type="quality.score",
                schema={"type": "object"},
            ),
            PythonScorer(
                name="custom_score",
                semantic_type="quality.score",
                target="pkg.scoring:score",
                role=ObservationRole.DIAGNOSTIC,
            ),
        ],
        derive=[TokenCostDeriverSpec(pricing="pricing/models.yaml")],
        post_derive=[
            PairedBaselineDeriverSpec(
                baseline_variant="base",
                metric="time.latency",
                output=DerivedMetricOutput(
                    name="speedup",
                    semantic_type="performance.speedup",
                ),
            )
        ],
        policies=[
            PolicySpec(
                name="minimum_score",
                metric="quality.score",
                must_greater_equal=0.8,
            )
        ],
        reports=ReportSpec(
            leaderboard=LeaderboardReportSpec(
                metrics=(
                    MetricAggregation(
                        name="avg_score",
                        semantic_type="quality.score",
                        fn="mean",
                    ),
                )
            ),
            case_matrix=CaseMatrixReportSpec(semantic_type="quality.score"),
            comparisons=(ComparisonReportSpec(baseline="base", candidate="candidate"),),
            distributions=(
                DistributionReportSpec(name="score_distribution", semantic_type="quality.score"),
            ),
        ),
        semantic_registry=custom_registry,
    )

    view = benchmark_spec_to_yaml_view(spec)
    body = view["benchmark"]["export-branches"]

    assert body["dataset"] == {
        "id": "cases",
        "source": "cases.yaml",
        "version": "v2",
        "metadata": {"owner": "tests"},
        "case_defaults": {
            "input": {"channel": "chat"},
            "expected": {"mode": "strict"},
            "metadata": {"suite": "smoke"},
            "tags": ["golden"],
            "attachments": [
                {
                    "id": "default",
                    "name": "default.txt",
                    "media_type": "text/plain",
                    "value": "seed",
                    "span_id": "span-default",
                    "tags": {"role": "seed"},
                }
            ],
        },
        "cases": [{"id": "case_1"}],
    }
    assert body["run"] == {
        "kind": "python",
        "target": "pkg.tasks:run",
        "module_search_paths": ["src"],
    }
    assert body["variants"]["candidate"] == {
        "label": "candidate label",
        "factors": {
            "model": {"value": "demo-model", "semantic": "llm.model.name"},
            "tunable": {"value": 2, "optimize": True},
        },
    }
    assert body["score"]["schema_score"] == {
        "schema": {"type": "object"},
        "from": "output.payload",
        "semantic": "quality.score",
        "unit": "points",
        "optional": True,
    }
    assert body["score"]["schema_default_path"] == {
        "schema": {"type": "object"},
        "semantic": "quality.score",
    }
    assert body["score"]["custom_score"] == {
        "python": "pkg.scoring:score",
        "semantic": "quality.score",
        "role": "diagnostic",
    }
    assert body["report"] == {
        "leaderboard": {
            "show": {
                "avg_score": {"metric": "quality.score", "aggregate": "mean"},
            }
        },
        "matrix": "quality.score",
        "compare": {"base -> candidate": {}},
        "distributions": [
            {
                "name": "score_distribution",
                "semantic_type": "quality.score",
            }
        ],
    }
    assert body["derive"] == [{"pricing": "pricing/models.yaml"}]
    assert body["post_derive"] == [
        {
            "baseline_variant": "base",
            "metric": "time.latency",
            "output": {
                "name": "speedup",
                "semantic_type": "performance.speedup",
            },
        }
    ]
    assert body["policies"] == [
        {
            "name": "minimum_score",
            "metric": "quality.score",
            "must_greater_equal": 0.8,
        }
    ]
    assert body["semantic_registry"] == {
        "version": 2,
        "aliases": {"custom.metric.alias": "custom.metric"},
        "types": {
            "custom.metric": {
                "parent": "quality.score",
                "unit": "ms",
                "shape": "number",
                "aliases": ["custom.metric.v1"],
                "deprecated": True,
                "tags": {"owner": "bench"},
            },
            "custom.unit_only": {
                "unit": "usd",
            },
        },
    }

    shell_spec = spec.model_copy(update={"task": TaskSpec(kind="shell", target="bin/run")})
    assert benchmark_spec_to_yaml_view(shell_spec)["benchmark"]["export-branches"]["run"] == {
        "kind": "shell",
        "target": "bin/run",
    }


def test_benchmark_spec_yaml_view_omits_optional_sections_for_minimal_specs() -> None:
    spec = BenchmarkSpec(benchmark=BenchmarkInfo(id="minimal"))

    view = benchmark_spec_to_yaml_view(spec)

    assert view == {"benchmark": {"minimal": {"dataset": {"cases": []}}}}


def test_benchmark_spec_payload_rejects_non_mapping_input() -> None:
    with pytest.raises(TypeError, match="mapping"):
        benchmark_spec_payload_from_yaml_view("not-a-mapping")


def test_load_benchmark_spec_supports_dataset_dsl_source(tmp_path: Path) -> None:
    dataset_path = tmp_path / "cases.yaml"
    dataset_path.write_text(
        "\n".join(
            (
                "dataset:",
                "  tickets:",
                "    version: v1",
                "    metadata:",
                "      owner: support",
                "    case_defaults:",
                "      tags: [smoke]",
                "    cases:",
                "      - id: ticket_1",
                "        input:",
                "          subject: Refund",
            )
        ),
        encoding="utf-8",
    )
    path = tmp_path / "autobench.yaml"
    path.write_text(
        "\n".join(
            (
                "benchmark:",
                "  dataset-dsl:",
                "    cases: cases.yaml",
                "    run: app.tasks:run",
            )
        ),
        encoding="utf-8",
    )

    spec = load_benchmark_spec(path)
    view = dataset_to_yaml_view(spec.dataset)

    assert spec.dataset.id == "tickets"
    assert spec.dataset.version == "v1"
    assert spec.dataset.metadata == {"owner": "support"}
    assert spec.dataset.cases[0].tags == ["smoke"]
    assert view["dataset"]["id"] == "tickets"
    assert view["dataset"]["version"] == "v1"
    assert view["dataset"]["metadata"] == {"owner": "support"}
    assert view["dataset"]["defaults"] == {
        "tags": ["smoke"],
    }
    assert view["dataset"]["cases"][0]["id"] == "ticket_1"


def test_dataset_dsl_source_supports_empty_body_multi_entry_fallback_and_errors(
    tmp_path: Path,
) -> None:
    empty_dataset = tmp_path / "empty-dataset.yaml"
    empty_dataset.write_text("dataset:\n  empty:\n", encoding="utf-8")
    multi_dataset = tmp_path / "multi-dataset.yaml"
    multi_dataset.write_text("dataset:\n  first: {}\n  second: {}\n", encoding="utf-8")
    bad_dataset = tmp_path / "bad-dataset.yaml"
    bad_dataset.write_text("dataset:\n  bad: scalar\n", encoding="utf-8")
    spec_path = tmp_path / "autobench.yaml"

    spec_path.write_text("benchmark:\n  empty:\n    cases: empty-dataset.yaml\n", encoding="utf-8")
    empty_spec = load_benchmark_spec(spec_path)
    assert empty_spec.dataset.id == "empty"
    assert empty_spec.dataset.cases == []

    spec_path.write_text("benchmark:\n  multi:\n    cases: multi-dataset.yaml\n", encoding="utf-8")
    multi_spec = load_benchmark_spec(spec_path)
    assert multi_spec.dataset.id is None
    assert multi_spec.dataset.cases == []

    spec_path.write_text("benchmark:\n  bad:\n    cases: bad-dataset.yaml\n", encoding="utf-8")
    with pytest.raises(SpecValidationError, match="dataset.<id> must be a mapping"):
        load_benchmark_spec(spec_path)


def test_dataset_yaml_view_omits_empty_fields_and_renders_attachments() -> None:
    dataset = DatasetSpec(
        id="cases",
        source="file://datasets/cases.yaml",
        version="v2",
        metadata={"owner": "tests"},
        case_defaults=CaseDefaults(
            input={"channel": "chat"},
            expected={"mode": "strict"},
            metadata={"suite": "golden"},
            attachments=[
                ArtifactRef(
                    id="default_prompt",
                    name="prompt.md",
                    media_type="text/markdown",
                    value="hello",
                    span_id="span-default",
                    tags={"role": "system"},
                )
            ],
        ),
        cases=[
            Case(
                id="case_1",
                expected={"ok": True},
                metadata={"priority": "high"},
                attachments=[
                    ArtifactRef(
                        id="case_image",
                        name="image.png",
                        value="artifact://case-image",
                    )
                ],
            ),
            Case(id="case_2"),
        ],
    )

    view = dataset_to_yaml_view(dataset)

    assert view["record"] == {"type": "dataset", "version": 1}
    assert view["dataset"]["id"] == "cases"
    assert view["dataset"]["source"] == "file://datasets/cases.yaml"
    assert view["dataset"]["version"] == "v2"
    assert view["dataset"]["metadata"] == {"owner": "tests"}
    assert view["dataset"]["defaults"] == {
        "input": {"channel": "chat"},
        "expected": {"mode": "strict"},
        "metadata": {"suite": "golden"},
        "attachments": [
            {
                "id": "default_prompt",
                "name": "prompt.md",
                "media_type": "text/markdown",
                "value": "hello",
                "span_id": "span-default",
                "tags": {"role": "system"},
            }
        ],
    }
    assert view["dataset"]["cases"][0] == {
        "id": "case_1",
        "expected": {"ok": True},
        "metadata": {"priority": "high"},
        "attachments": [
            {
                "id": "case_image",
                "name": "image.png",
                "value": "artifact://case-image",
            }
        ],
    }
    assert view["dataset"]["cases"][1] == {"id": "case_2"}


def test_dataset_yaml_view_omits_optional_top_level_sections_when_empty() -> None:
    dataset = DatasetSpec(
        cases=[
            Case(
                id="case_1",
                input={"text": "hello"},
                tags=["smoke"],
                attachments=[
                    ArtifactRef(
                        id="inline_note",
                        name="note.txt",
                    )
                ],
            )
        ],
    )

    view = dataset_to_yaml_view(dataset)

    assert view == {
        "record": {"type": "dataset", "version": 1},
        "dataset": {
            "id": "inline",
            "cases": [
                {
                    "id": "case_1",
                    "input": {"text": "hello"},
                    "tags": ["smoke"],
                    "attachments": [
                        {
                            "id": "inline_note",
                            "name": "note.txt",
                        }
                    ],
                }
            ],
        },
    }


def test_artifact_ref_yaml_view_covers_present_and_absent_optional_fields() -> None:
    assert _artifact_ref_to_yaml_view(ArtifactRef(id="minimal", name="minimal.txt")) == {
        "id": "minimal",
        "name": "minimal.txt",
    }
    assert _artifact_ref_to_yaml_view(
        ArtifactRef(
            id="full",
            name="full.txt",
            media_type="text/plain",
            value="payload",
            span_id="span-1",
            tags={"role": "seed"},
        )
    ) == {
        "id": "full",
        "name": "full.txt",
        "media_type": "text/plain",
        "value": "payload",
        "span_id": "span-1",
        "tags": {"role": "seed"},
    }


def test_load_benchmark_spec_supports_authoring_dsl_alternate_shapes(
    tmp_path: Path,
) -> None:
    path = tmp_path / "autobench.yaml"
    path.write_text(
        "\n".join(
            (
                "benchmark:",
                "  branchy:",
                "    cases:",
                "      - id: case_1",
                "    run: app.tasks:run",
                "    variants:",
                "      empty_variant:",
                "      list_factor_variant:",
                "        factors:",
                "          - name: student_model",
                "            value: demo-model",
                "            semantic_type: llm.model.name",
                "    score:",
                "      confidence:",
                "        value: output.confidence",
                "        semantic_type: quality.score",
                "        direction: maximize",
                "      schema_score:",
                "        schema:",
                "          type: object",
                "        semantic: quality.score",
                "      custom_score:",
                "        python: app.scoring:score",
                "        semantic: quality.score",
                "    report:",
                "      leaderboard:",
                "        show:",
                "          - name: list_metric",
                "            metric: quality.score",
                "            aggregate: mean",
                "          - name: list_metric_legacy",
                "            semantic_type: result.success",
                "            fn: ratio_true",
                "      matrix: quality.score",
                "      compare:",
                "        empty_variant -> list_factor_variant:",
                "          show:",
                "            quick_metric: quality.score",
                "        list_factor_variant -> empty_variant:",
                "      distributions: []",
            )
        ),
        encoding="utf-8",
    )

    spec = load_benchmark_spec(path)

    assert spec.benchmark.id == "branchy"
    assert spec.dataset.cases[0].id == "case_1"
    assert spec.task is not None
    assert spec.task.target == "app.tasks:run"
    assert spec.variants[0].factors == []
    assert spec.variants[1].factors[0].name == "student_model"
    assert [scorer.kind for scorer in spec.scoring] == ["output", "schema", "python"]
    assert spec.scoring[0].direction is not None
    assert spec.scoring[0].direction.value == "maximize"
    assert spec.reports.leaderboard.metrics[0].name == "list_metric"
    assert spec.reports.leaderboard.metrics[1].name == "list_metric_legacy"
    assert spec.reports.case_matrix.semantic_type == Semantic.QUALITY_SCORE
    assert spec.reports.comparisons[0].metrics[0].name == "quick_metric"
    assert spec.reports.comparisons[1].metrics == ()


def test_load_benchmark_spec_supports_authoring_dsl_case_mapping_and_run_target(
    tmp_path: Path,
) -> None:
    path = tmp_path / "autobench.yaml"
    path.write_text(
        "\n".join(
            (
                "benchmark:",
                "  mapped:",
                "    cases:",
                "      cases: []",
                "    run:",
                "      target: app.tasks:run",
                "    report:",
                "      matrix: quality.score",
            )
        ),
        encoding="utf-8",
    )

    spec = load_benchmark_spec(path)

    assert spec.benchmark.id == "mapped"
    assert spec.dataset.cases == []
    assert spec.task is not None
    assert spec.task.kind == "python"
    assert spec.task.target == "app.tasks:run"
    assert spec.reports.case_matrix.semantic_type == Semantic.QUALITY_SCORE


def test_load_benchmark_spec_supports_authoring_dsl_body_fallback_sections(
    tmp_path: Path,
) -> None:
    path = tmp_path / "autobench.yaml"
    path.write_text(
        "\n".join(
            (
                "benchmark:",
                "  fallback:",
                "    dataset:",
                "      cases: []",
                "    task:",
                "      kind: python",
                "      target: app.tasks:run",
                "    reports:",
                "      leaderboard:",
                "        metrics: []",
            )
        ),
        encoding="utf-8",
    )

    spec = load_benchmark_spec(path)

    assert spec.benchmark.id == "fallback"
    assert spec.dataset.cases == []
    assert spec.task is not None
    assert spec.task.kind == "python"
    assert spec.task.target == "app.tasks:run"
    assert spec.reports.leaderboard.metrics == ()


@pytest.mark.parametrize(
    ("body", "message"),
    [
        ("    cases: 1", "cases must be a string, list, or mapping"),
        ("    run: []", "run must be a string or mapping"),
        ("    run: {}", "run must define python or target"),
        ("    variants: []", "variants must be a mapping"),
        ("    variants:\n      bad: scalar", "variants entries must be mappings"),
        (
            "    variants:\n      bad:\n        factors: 1",
            "factors must be a mapping",
        ),
        (
            "    variants:\n      bad:\n        factors:\n          - nope",
            "factors entries must be mappings",
        ),
        ("    score: []", "score must be a mapping"),
        ("    score:\n      bad: 1", "score.bad must be a mapping"),
        (
            "    score:\n      bad:\n        semantic: quality.score",
            "must define exactly one scoring action",
        ),
        (
            "    score:\n      bad:\n        pass: output.ok\n        value: output.ok\n"
            "        semantic: quality.score",
            "must define exactly one scoring action",
        ),
        (
            "    score:\n      bad:\n        exact: output.ok\n        semantic: quality.score",
            "score.bad.exact must be a mapping",
        ),
        ("    report: []", "report must be a mapping"),
        ("    report:\n      leaderboard: []", "report.leaderboard must be a mapping"),
        ("    report:\n      matrix: []", "report.matrix must be a string or mapping"),
        ("    report:\n      matrix: {}", "report.matrix must define metric"),
        ("    report:\n      compare: []", "report.compare must be a mapping"),
        (
            "    report:\n      compare:\n        bad-key: {}",
            "compare keys must use",
        ),
        (
            "    report:\n      compare:\n        a -> b: scalar",
            "compare entries must be mappings",
        ),
        (
            "    report:\n      leaderboard:\n        show:\n          - nope",
            "report metric entries must be mappings",
        ),
        (
            "    report:\n      leaderboard:\n        show: 1",
            "report show must be a mapping or list",
        ),
        (
            "    report:\n      leaderboard:\n        show:\n          bad: 1",
            "report show entries must be strings or mappings",
        ),
        (
            "    report:\n      leaderboard:\n        show:\n          bad:\n"
            "            metric: quality.score",
            "report show entries must define metric and aggregate",
        ),
    ],
)
def test_load_benchmark_spec_reports_authoring_dsl_errors(
    tmp_path: Path,
    body: str,
    message: str,
) -> None:
    path = tmp_path / "autobench.yaml"
    path.write_text(f"benchmark:\n  bad:\n{body}\n", encoding="utf-8")

    with pytest.raises(SpecValidationError, match=message):
        load_benchmark_spec(path)


def test_load_benchmark_spec_supports_file_backed_dataset(tmp_path: Path) -> None:
    datasets_dir = tmp_path / "datasets"
    datasets_dir.mkdir()
    dataset_path = datasets_dir / "cases.yaml"
    dataset_path.write_text(
        "\n".join(
            (
                "case_defaults:",
                "  tags: [shared]",
                "cases:",
                "  - id: case_1",
                "    input:",
                "      message: hello",
                "  - id: case_2",
                "    input:",
                "      message: bye",
            )
        ),
        encoding="utf-8",
    )
    spec_path = tmp_path / "autobench.yaml"
    spec_path.write_text(
        "\n".join(
            (
                "benchmark:",
                "  id: file-backed",
                "dataset:",
                "  source: file://datasets/cases.yaml",
                "task:",
                "  kind: python",
                "  target: app.tasks.run_demo",
                "variants:",
                "  - id: variant_1",
                "    factors:",
                "      - name: model",
                "        value: demo-model",
                "        semantic_type: llm.model.name",
            )
        ),
        encoding="utf-8",
    )

    spec = load_benchmark_spec(spec_path)

    assert spec.dataset.source == "file://datasets/cases.yaml"
    assert [case.id for case in spec.dataset.cases] == ["case_1", "case_2"]
    assert spec.dataset.cases[0].tags == ["shared"]
    assert collect_benchmark_source_files(spec_path) == (
        spec_path.resolve(),
        dataset_path.resolve(),
    )


def test_load_benchmark_spec_supports_glob_backed_case_files(tmp_path: Path) -> None:
    cases_dir = tmp_path / "cases"
    cases_dir.mkdir()
    (cases_dir / "01-case.yaml").write_text(
        "id: case_from_mapping\ninput:\n  value: 1\n",
        encoding="utf-8",
    )
    (cases_dir / "02-cases.yaml").write_text(
        "cases:\n  - id: case_from_dataset\n    input:\n      value: 2\n",
        encoding="utf-8",
    )
    (cases_dir / "03-list.yaml").write_text(
        "- id: case_from_list\n  input:\n    value: 3\n",
        encoding="utf-8",
    )
    (cases_dir / "04-empty.yaml").write_text("", encoding="utf-8")
    path = tmp_path / "autobench.yaml"
    path.write_text(
        "\n".join(
            (
                "benchmark:",
                "  id: glob-backed",
                "dataset:",
                "  source: cases/*.yaml",
            )
        ),
        encoding="utf-8",
    )

    spec = load_benchmark_spec(path)

    assert [case.id for case in spec.dataset.cases] == [
        "case_from_mapping",
        "case_from_dataset",
        "case_from_list",
    ]
    assert spec.dataset.source == "cases/*.yaml"
    assert collect_benchmark_source_files(path) == (
        path.resolve(),
        (cases_dir / "01-case.yaml").resolve(),
        (cases_dir / "02-cases.yaml").resolve(),
        (cases_dir / "03-list.yaml").resolve(),
        (cases_dir / "04-empty.yaml").resolve(),
    )
    assert build_benchmark_plan(spec).case_ids == (
        "case_from_mapping",
        "case_from_dataset",
        "case_from_list",
    )


def test_collect_benchmark_source_files_includes_pricing_once(tmp_path: Path) -> None:
    pricing_path = tmp_path / "pricing.yaml"
    pricing_path.write_text("models: {}\n", encoding="utf-8")
    path = tmp_path / "autobench.yaml"
    path.write_text(
        "\n".join(
            (
                "benchmark:",
                "  id: pricing-source-files",
                "derive:",
                "  - kind: token_cost",
                "    pricing: pricing.yaml",
                "  - kind: token_cost",
                "    pricing: pricing.yaml",
            )
        ),
        encoding="utf-8",
    )

    assert collect_benchmark_source_files(path) == (path.resolve(), pricing_path.resolve())


def test_collect_benchmark_source_files_handles_empty_and_invalid_specs(tmp_path: Path) -> None:
    empty_path = tmp_path / "empty.yaml"
    empty_path.write_text("", encoding="utf-8")
    invalid_path = tmp_path / "invalid.yaml"
    invalid_path.write_text("- not: a mapping\n", encoding="utf-8")

    assert collect_benchmark_source_files(empty_path) == (empty_path.resolve(),)
    with pytest.raises(SpecValidationError, match="Expected mapping"):
        collect_benchmark_source_files(invalid_path)


def test_collect_benchmark_source_files_ignores_non_file_ref_shapes(tmp_path: Path) -> None:
    path = tmp_path / "autobench.yaml"
    path.write_text(
        "\n".join(
            (
                "benchmark:",
                "  id: ignored-source-shapes",
                "dataset: []",
                "derive:",
                "  - not-a-mapping",
                "  - kind: token_cost",
                "    pricing: 42",
            )
        ),
        encoding="utf-8",
    )

    assert collect_benchmark_source_files(path) == (path.resolve(),)


def test_collect_benchmark_source_files_ignores_non_string_python_targets(tmp_path: Path) -> None:
    path = tmp_path / "autobench.yaml"
    path.write_text(
        "\n".join(
            (
                "benchmark:",
                "  id: ignored-python-targets",
                "task:",
                "  kind: python",
                "  target: 42",
                "scoring:",
                "  - kind: python",
                "    name: score",
                "    semantic_type: quality.score",
                "    target: 42",
                "  - not-a-mapping",
            )
        ),
        encoding="utf-8",
    )

    assert collect_benchmark_source_files(path) == (path.resolve(),)


def test_collect_benchmark_source_files_requires_glob_matches(tmp_path: Path) -> None:
    path = tmp_path / "autobench.yaml"
    path.write_text(
        "benchmark:\n  id: missing-source-files\ndataset:\n  source: missing/*.yaml\n",
        encoding="utf-8",
    )

    with pytest.raises(SpecValidationError, match="matched no files"):
        collect_benchmark_source_files(path)


def test_collect_benchmark_source_files_handles_unresolvable_python_modules(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import autobench.spec as spec_module

    path = tmp_path / "autobench.yaml"
    path.write_text(
        "\n".join(
            (
                "benchmark:",
                "  id: unresolved-python-target",
                "task:",
                "  kind: python",
                "  target: missing.module:run",
            )
        ),
        encoding="utf-8",
    )

    def raise_find_spec(module_name: str) -> None:
        raise ModuleNotFoundError(module_name)

    monkeypatch.setattr(spec_module.importlib.util, "find_spec", raise_find_spec)

    assert collect_benchmark_source_files(path) == (path.resolve(),)


def test_collect_benchmark_source_files_skips_targets_without_real_files(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import autobench.spec as spec_module

    path = tmp_path / "autobench.yaml"
    path.write_text(
        "\n".join(
            (
                "benchmark:",
                "  id: unresolved-python-target",
                "task:",
                "  kind: python",
                "  target: missing.module:run",
            )
        ),
        encoding="utf-8",
    )

    class MissingOrigin:
        origin = None

    monkeypatch.setattr(
        spec_module.importlib.util, "find_spec", lambda module_name: MissingOrigin()
    )
    assert collect_benchmark_source_files(path) == (path.resolve(),)

    class NonFileOrigin:
        origin = str(tmp_path / "not-a-file")

    (tmp_path / "not-a-file").mkdir()
    monkeypatch.setattr(
        spec_module.importlib.util, "find_spec", lambda module_name: NonFileOrigin()
    )
    assert collect_benchmark_source_files(path) == (path.resolve(),)


def test_collect_benchmark_source_files_includes_python_scorer_modules(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    scorer_module = tmp_path / "source_scorers.py"
    scorer_module.write_text(
        "def score(call):\n    return 1.0\n",
        encoding="utf-8",
    )
    monkeypatch.syspath_prepend(str(tmp_path))
    path = tmp_path / "autobench.yaml"
    path.write_text(
        "\n".join(
            (
                "benchmark:",
                "  id: scorer-source-files",
                "scoring:",
                "  - kind: python",
                "    name: score",
                "    semantic_type: quality.score",
                "    target: source_scorers:score",
            )
        ),
        encoding="utf-8",
    )

    assert collect_benchmark_source_files(path) == (path.resolve(), scorer_module.resolve())


def test_collect_benchmark_source_files_recurses_nested_targets_and_file_refs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    helper_module = tmp_path / "nested_helpers.py"
    helper_module.write_text(
        "def extract(call):\n    return 1\n",
        encoding="utf-8",
    )
    rubric_path = tmp_path / "rubrics" / "quality.md"
    rubric_path.parent.mkdir()
    rubric_path.write_text("judge rubric\n", encoding="utf-8")
    monkeypatch.syspath_prepend(str(tmp_path))
    path = tmp_path / "autobench.yaml"
    path.write_text(
        "\n".join(
            (
                "benchmark:",
                "  id: nested-source-files",
                "analysis:",
                "  extractor:",
                "    target: nested_helpers:extract",
                "  rubric_file: rubrics/quality.md",
            )
        ),
        encoding="utf-8",
    )

    assert collect_benchmark_source_files(path) == (
        path.resolve(),
        helper_module.resolve(),
        rubric_path.resolve(),
    )


def test_spec_source_helpers_cover_non_glob_target_and_invalid_file_ref(tmp_path: Path) -> None:
    import autobench.spec as spec_module

    dataset_path = tmp_path / "dataset.yaml"
    dataset_path.write_text("cases: []\n", encoding="utf-8")

    assert spec_module._resolve_source_paths(
        "dataset.yaml", base_path=tmp_path / "autobench.yaml"
    ) == [dataset_path.resolve()]
    assert (
        spec_module._resolve_python_target_source_paths(
            "not_a_python_target",
            base_path=tmp_path / "autobench.yaml",
        )
        == []
    )
    assert (
        spec_module._resolve_file_reference_paths(
            "https://example.com/spec.yaml",
            base_path=tmp_path / "autobench.yaml",
        )
        == []
    )


def test_spec_source_helpers_resolve_spec_relative_python_modules(tmp_path: Path) -> None:
    import autobench.spec as spec_module

    package_name = "spec_only_localpkg"
    package_dir = tmp_path / package_name
    package_dir.mkdir()
    (package_dir / "__init__.py").write_text("def run(ctx, case):\n    return {'ok': True}\n")
    (package_dir / "helpers.py").write_text("def score(call):\n    return 1.0\n")
    base_path = tmp_path / "autobench.yaml"

    assert spec_module._resolve_python_target_source_paths(
        f"{package_name}:run",
        base_path=base_path,
    ) == [(package_dir / "__init__.py").resolve()]
    assert spec_module._resolve_python_target_source_paths(
        f"{package_name}.helpers:score",
        base_path=base_path,
    ) == [(package_dir / "helpers.py").resolve()]


@pytest.mark.parametrize(
    ("filename", "content", "message"),
    [
        ("bad-cases.yaml", "cases:\n  id: nope\n", "Expected cases list"),
        ("bad-shape.yaml", "name: missing-id\n", "Expected case mapping or dataset mapping"),
        ("bad-scalar.yaml", "42\n", "Expected case mapping or list"),
    ],
)
def test_glob_backed_case_files_report_bad_shapes(
    tmp_path: Path,
    filename: str,
    content: str,
    message: str,
) -> None:
    cases_dir = tmp_path / "cases"
    cases_dir.mkdir()
    (cases_dir / filename).write_text(content, encoding="utf-8")
    path = tmp_path / "autobench.yaml"
    path.write_text(
        "benchmark:\n  id: bad-glob\ndataset:\n  source: cases/*.yaml\n",
        encoding="utf-8",
    )

    with pytest.raises(SpecValidationError, match=message):
        load_benchmark_spec(path)


def test_glob_backed_dataset_requires_matches(tmp_path: Path) -> None:
    path = tmp_path / "autobench.yaml"
    path.write_text(
        "benchmark:\n  id: no-glob-matches\ndataset:\n  source: missing/*.yaml\n",
        encoding="utf-8",
    )

    with pytest.raises(SpecValidationError, match="matched no files"):
        load_benchmark_spec(path)


def test_load_benchmark_spec_rejects_duplicate_case_ids(tmp_path: Path) -> None:
    path = tmp_path / "autobench.yaml"
    path.write_text(
        "\n".join(
            (
                "benchmark:",
                "  id: duplicate-cases",
                "dataset:",
                "  cases:",
                "    - id: case_1",
                "    - id: case_1",
                "variants: []",
            )
        ),
        encoding="utf-8",
    )

    with pytest.raises(SpecValidationError):
        load_benchmark_spec(path)


def test_load_benchmark_spec_rejects_duplicate_variant_ids(tmp_path: Path) -> None:
    path = tmp_path / "autobench.yaml"
    path.write_text(
        "\n".join(
            (
                "benchmark:",
                "  id: duplicate-variants",
                "dataset:",
                "  cases:",
                "    - id: case_1",
                "variants:",
                "  - id: variant_1",
                "    factors: []",
                "  - id: variant_1",
                "    factors: []",
            )
        ),
        encoding="utf-8",
    )

    with pytest.raises(SpecValidationError):
        load_benchmark_spec(path)


def test_load_benchmark_spec_rejects_runnable_matrix_without_task(tmp_path: Path) -> None:
    path = tmp_path / "autobench.yaml"
    path.write_text(
        "\n".join(
            (
                "benchmark:",
                "  id: missing-task",
                "dataset:",
                "  cases:",
                "    - id: case_1",
                "variants:",
                "  - id: variant_1",
                "    factors: []",
            )
        ),
        encoding="utf-8",
    )

    with pytest.raises(SpecValidationError):
        load_benchmark_spec(path)


def test_load_benchmark_spec_merges_custom_semantic_registry_with_defaults(
    tmp_path: Path,
) -> None:
    path = tmp_path / "autobench.yaml"
    path.write_text(
        "\n".join(
            (
                "benchmark:",
                "  id: custom-registry",
                "semantic_registry:",
                "  aliases:",
                "    answer.score: quality.score",
                "  types:",
                "    myapp.retrieval.chunk_count:",
                "      value_shape: integer",
                "      tags:",
                "        domain: retrieval",
            )
        ),
        encoding="utf-8",
    )

    spec = load_benchmark_spec(path)

    assert spec.semantic_registry.normalize("answer.score") == Semantic.QUALITY_SCORE
    assert spec.semantic_registry.info_for(Semantic.LLM_MODEL_NAME) is not None
    custom_info = spec.semantic_registry.info_for("myapp.retrieval.chunk_count")
    assert custom_info is not None
    assert custom_info.value_shape == "integer"
    assert custom_info.tags == {"domain": "retrieval"}


def test_load_benchmark_spec_preserves_custom_semantic_registry_version(
    tmp_path: Path,
) -> None:
    path = tmp_path / "autobench.yaml"
    path.write_text(
        "\n".join(
            (
                "benchmark:",
                "  id: custom-registry-version",
                "semantic_registry:",
                "  version: 2",
            )
        ),
        encoding="utf-8",
    )

    spec = load_benchmark_spec(path)

    assert spec.semantic_registry.version == 2


@pytest.mark.parametrize(
    ("content", "message"),
    [
        ("", "Field required"),
        ("- not: a mapping\n", "Expected mapping at top level"),
        ("benchmark:\n  id: bad-dataset\ndataset: []\n", "dataset must be a mapping"),
        (
            "benchmark:\n  id: bad-dataset-source\ndataset:\n  source: 42\n",
            "dataset.source must be a string",
        ),
        (
            "benchmark:\n  id: bad-variants\nvariants: {}\n",
            "variants must be a list",
        ),
        (
            "benchmark:\n  id: bad-variant-entry\nvariants:\n  - not-a-mapping\n",
            "variant entries must be mappings",
        ),
        (
            "benchmark:\n  id: bad-registry\nsemantic_registry: []\n",
            "semantic_registry must be a mapping",
        ),
        (
            "benchmark:\n  id: bad-aliases\nsemantic_registry:\n  aliases: []\n",
            "semantic_registry.aliases must be a mapping",
        ),
        (
            "benchmark:\n  id: bad-types\nsemantic_registry:\n  types: []\n",
            "semantic_registry.types must be a mapping",
        ),
        (
            "benchmark:\n  id: bad-type-payload\nsemantic_registry:\n  types:\n    custom.metric: []\n",
            "semantic_registry.types.custom.metric must be a mapping",
        ),
        (
            "benchmark:\n  id: bad-derive\nderive: {}\n",
            "derive must be a list",
        ),
        (
            "benchmark:\n  id: bad-derive-entry\nderive:\n  - not-a-mapping\n",
            "derive entries must be mappings",
        ),
        (
            "benchmark:\n  id: bad-derive-pricing\nderive:\n  - kind: token_cost\n    pricing: 42\n",
            "derive pricing must be a string",
        ),
        (
            "benchmark:\n  id: missing-derive-pricing\nderive:\n  - kind: token_cost\n",
            "Field required",
        ),
        (
            "benchmark:\n  id: bad-post-derive\npost_derive: {}\n",
            "post_derive must be a list",
        ),
        (
            "benchmark:\n  id: bad-post-derive-entry\npost_derive:\n  - not-a-mapping\n",
            "post_derive entries must be mappings",
        ),
        (
            "benchmark:\n  id: bad-policies\npolicies: {}\n",
            "policies must be a list",
        ),
        (
            "benchmark:\n  id: bad-policy-entry\npolicies:\n  - not-a-mapping\n",
            "policy entries must be mappings",
        ),
        (
            "benchmark:\n  single-entry-but-not-dsl: scalar\n",
            "Field required",
        ),
    ],
)
def test_load_benchmark_spec_reports_structured_spec_errors(
    tmp_path: Path,
    content: str,
    message: str,
) -> None:
    path = tmp_path / "autobench.yaml"
    path.write_text(content, encoding="utf-8")

    with pytest.raises(SpecValidationError, match=message):
        load_benchmark_spec(path)


def test_file_backed_dataset_must_be_a_mapping(tmp_path: Path) -> None:
    dataset_path = tmp_path / "cases.yaml"
    dataset_path.write_text("- id: case_1\n", encoding="utf-8")
    spec_path = tmp_path / "autobench.yaml"
    spec_path.write_text(
        "\n".join(
            (
                "benchmark:",
                "  id: bad-file-backed-dataset",
                "dataset:",
                "  source: cases.yaml",
            )
        ),
        encoding="utf-8",
    )

    with pytest.raises(SpecValidationError, match="Expected dataset mapping"):
        load_benchmark_spec(spec_path)


def test_empty_file_backed_dataset_defaults_to_empty_mapping(tmp_path: Path) -> None:
    dataset_path = tmp_path / "cases.yaml"
    dataset_path.write_text("", encoding="utf-8")
    spec_path = tmp_path / "autobench.yaml"
    spec_path.write_text(
        "\n".join(
            (
                "benchmark:",
                "  id: empty-file-backed-dataset",
                "dataset:",
                "  source: cases.yaml",
            )
        ),
        encoding="utf-8",
    )

    spec = load_benchmark_spec(spec_path)

    assert spec.dataset.cases == []


def test_build_benchmark_plan_counts_case_variant_matrix(tmp_path: Path) -> None:
    path = tmp_path / "autobench.yaml"
    path.write_text(
        "\n".join(
            (
                "benchmark:",
                "  id: matrix-demo",
                "dataset:",
                "  cases:",
                "    - id: case_1",
                "    - id: case_2",
                "task:",
                "  kind: python",
                "  target: app.tasks.run_demo",
                "variants:",
                "  - id: model_pair_1",
                "    factors: []",
                "  - id: model_pair_2",
                "    factors: []",
            )
        ),
        encoding="utf-8",
    )

    spec = load_benchmark_spec(path)
    plan = build_benchmark_plan(spec)

    assert plan.benchmark_id == "matrix-demo"
    assert plan.case_count == 2
    assert plan.variant_count == 2
    assert plan.planned_run_count == 4
    assert plan.warnings == []
