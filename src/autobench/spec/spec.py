from __future__ import annotations as _annotations

from typing import TYPE_CHECKING, Any

from pydantic import BaseModel

from autobench.data.datasets import CaseDefaults, DatasetSpec, case_to_yaml_view
from autobench.data.variants import FactorValue, Variant
from autobench.errors import SpecValidationError
from autobench.evaluation.scoring import (
    ExactScorer,
    ExpectedActionScorer,
    OutputMetricScorer,
    PassFailScorer,
    PythonScorer,
    SchemaScorer,
    ScoringSpec,
)
from autobench.instrumentation.config import InstrumentationConfig
from autobench.metrics.semantics import DEFAULT_SEMANTIC_REGISTRY, SemanticRegistry
from autobench.reports.reporting import MetricAggregation, ReportSpec

if TYPE_CHECKING:
    from autobench.spec import BenchmarkSpec, TaskSpec


def benchmark_spec_to_yaml_view(spec: BenchmarkSpec) -> dict[str, Any]:
    body: dict[str, Any] = {}
    if spec.benchmark.description is not None:
        body["description"] = spec.benchmark.description

    body["dataset"] = _benchmark_dataset_to_yaml_view(spec.dataset)

    if spec.task is not None:
        body["run"] = _task_to_yaml_view(spec.task)
    if spec.variants:
        body["variants"] = _variants_to_yaml_view(spec.variants)
    if spec.scoring:
        body["score"] = _scoring_to_yaml_view(spec.scoring)
    if spec.derive:
        body["derive"] = [_compact_model_dump(item) for item in spec.derive]
    if spec.post_derive:
        body["post_derive"] = [_compact_model_dump(item) for item in spec.post_derive]
    if spec.policies:
        body["policies"] = [_compact_model_dump(item) for item in spec.policies]
    if spec.instrumentation:
        body["instrumentation"] = _instrumentation_to_yaml_view(spec.instrumentation)

    report_view = _report_to_yaml_view(spec.reports)
    if report_view:
        body["report"] = report_view

    semantic_registry_view = _semantic_registry_delta_to_yaml_view(spec.semantic_registry)
    if semantic_registry_view:
        body["semantic_registry"] = semantic_registry_view

    return {"benchmark": {spec.benchmark.id: body}}


def _normalize_benchmark_dsl(raw: dict[str, Any]) -> dict[str, Any]:
    benchmark_entry = _benchmark_dsl_entry(raw.get("benchmark"))
    if benchmark_entry is None:
        return dict(raw)

    benchmark_id, body = benchmark_entry
    normalized = {key: value for key, value in raw.items() if key != "benchmark"}
    normalized["benchmark"] = _normalize_benchmark_info(str(benchmark_id), body)

    if "cases" in body:
        normalized["dataset"] = _normalize_cases_section(body["cases"])
    if "run" in body:
        normalized["task"] = _normalize_run_section(body["run"])
    if "variants" in body:
        normalized["variants"] = _normalize_dsl_variants(body["variants"])
    if "score" in body:
        normalized["scoring"] = _normalize_score_section(body["score"])
    if "report" in body:
        normalized["reports"] = _normalize_report_section(body["report"])
    if "instrumentation" in body:
        normalized["instrumentation"] = _normalize_instrumentation_section(body["instrumentation"])

    for section in (
        "dataset",
        "task",
        "derive",
        "post_derive",
        "policies",
        "reports",
        "semantic_registry",
        "instrumentation",
    ):
        if section in body and section not in normalized:
            normalized[section] = body[section]
    return normalized


def _normalize_instrumentation_section(raw: Any) -> list[dict[str, Any]]:
    if isinstance(raw, list):
        normalized_list: list[dict[str, Any]] = []
        for item in raw:
            if not isinstance(item, dict):
                raise SpecValidationError("instrumentation list entries must be mappings")
            normalized_list.append(dict(item))
        return normalized_list
    if not isinstance(raw, dict):
        raise SpecValidationError("benchmark.<id>.instrumentation must be a mapping")

    normalized: list[dict[str, Any]] = []
    for kind, settings in raw.items():
        if not isinstance(kind, str):
            raise SpecValidationError("instrumentation names must be strings")
        if isinstance(settings, bool):
            normalized.append({"kind": kind, "enabled": settings})
            continue
        if settings is None:
            settings = {}
        if not isinstance(settings, dict):
            raise SpecValidationError(f"instrumentation.{kind} must be a mapping or boolean")
        normalized.append({"kind": kind, **settings})
    return normalized


def _instrumentation_to_yaml_view(
    instrumentation: list[InstrumentationConfig],
) -> dict[str, Any]:
    view: dict[str, Any] = {}
    for config in instrumentation:
        payload = config.model_dump(
            mode="json",
            exclude={"kind"},
            exclude_defaults=True,
        )
        if not config.enabled and payload == {"enabled": False}:
            view[config.kind] = False
        elif payload:
            view[config.kind] = payload
        else:
            view[config.kind] = {}
    return view


def _benchmark_dsl_entry(raw_benchmark: Any) -> tuple[str, dict[str, Any]] | None:
    if not isinstance(raw_benchmark, dict):
        return None
    if "id" in raw_benchmark or len(raw_benchmark) != 1:
        return None
    benchmark_id, raw_body = next(iter(raw_benchmark.items()))
    if not isinstance(raw_body, dict):
        return None
    return str(benchmark_id), {str(key): value for key, value in raw_body.items()}


def _normalize_benchmark_info(benchmark_id: str, body: dict[str, Any]) -> dict[str, Any]:
    info: dict[str, Any] = {"id": benchmark_id}
    if "description" in body:
        info["description"] = body["description"]
    return info


def _normalize_cases_section(raw_cases: Any) -> dict[str, Any]:
    if isinstance(raw_cases, str):
        return {"source": raw_cases}
    if isinstance(raw_cases, list):
        return {"cases": raw_cases}
    if isinstance(raw_cases, dict):
        return dict(raw_cases)
    raise SpecValidationError("benchmark.<id>.cases must be a string, list, or mapping")


def _normalize_run_section(raw_run: Any) -> dict[str, Any]:
    if isinstance(raw_run, str):
        return {"kind": "python", "target": raw_run}
    if not isinstance(raw_run, dict):
        raise SpecValidationError("benchmark.<id>.run must be a string or mapping")
    if "python" in raw_run:
        return {"kind": "python", "target": raw_run["python"]}
    if "target" in raw_run:
        payload = dict(raw_run)
        payload.setdefault("kind", "python")
        return payload
    raise SpecValidationError("benchmark.<id>.run must define python or target")


def _normalize_dsl_variants(raw_variants: Any) -> list[dict[str, Any]]:
    if not isinstance(raw_variants, dict):
        raise SpecValidationError("benchmark.<id>.variants must be a mapping")

    variants: list[dict[str, Any]] = []
    for variant_id, raw_variant in raw_variants.items():
        if raw_variant is None:
            raw_variant = {}
        if not isinstance(raw_variant, dict):
            raise SpecValidationError("benchmark.<id>.variants entries must be mappings")
        variant = {
            "id": str(variant_id),
            "factors": _normalize_dsl_factors(raw_variant.get("factors", {})),
        }
        if "label" in raw_variant:
            variant["label"] = raw_variant["label"]
        variants.append(variant)
    return variants


def _normalize_dsl_factors(raw_factors: Any) -> list[dict[str, Any]]:
    if isinstance(raw_factors, list):
        list_factors: list[dict[str, Any]] = []
        for factor in raw_factors:
            if not isinstance(factor, dict):
                raise SpecValidationError(
                    "benchmark.<id>.variants.<variant>.factors entries must be mappings"
                )
            list_factors.append(dict(factor))
        return list_factors
    if not isinstance(raw_factors, dict):
        raise SpecValidationError("benchmark.<id>.variants.<variant>.factors must be a mapping")

    mapped_factors: list[dict[str, Any]] = []
    for name, raw_value in raw_factors.items():
        if isinstance(raw_value, dict):
            factor = dict(raw_value)
            factor.setdefault("value", raw_value.get("value"))
            factor["name"] = str(name)
            if "semantic" in factor:
                factor["semantic_type"] = factor.pop("semantic")
        else:
            factor = {"name": str(name), "value": raw_value}
        mapped_factors.append(factor)
    return mapped_factors


def _normalize_score_section(raw_scores: Any) -> list[dict[str, Any]]:
    if not isinstance(raw_scores, dict):
        raise SpecValidationError("benchmark.<id>.score must be a mapping")

    return [
        _normalize_score(name=str(name), raw_score=raw_score)
        for name, raw_score in raw_scores.items()
    ]


def _normalize_score(name: str, *, raw_score: Any) -> dict[str, Any]:
    if not isinstance(raw_score, dict):
        raise SpecValidationError(f"benchmark.<id>.score.{name} must be a mapping")

    score = _normalize_score_common(name, raw_score)
    actions = [
        key
        for key in ("pass", "value", "exact", "schema", "python", "expected_action")
        if key in raw_score
    ]
    if len(actions) != 1:
        raise SpecValidationError(
            f"benchmark.<id>.score.{name} must define exactly one scoring action"
        )

    action = actions[0]
    if action == "pass":
        return score | {"kind": "pass_fail", "path": raw_score["pass"]}
    if action == "value":
        return score | {"kind": "output", "path": raw_score["value"]}
    if action == "exact":
        exact = raw_score["exact"]
        if not isinstance(exact, dict):
            raise SpecValidationError(f"benchmark.<id>.score.{name}.exact must be a mapping")
        return score | {
            "kind": "exact",
            "actual": exact.get("actual"),
            "expected": exact.get("expected"),
        }
    if action == "schema":
        schema = raw_score["schema"]
        return score | {
            "kind": "schema",
            "path": raw_score.get("path", raw_score.get("from", "output")),
            "schema": schema,
        }
    if action == "expected_action":
        action_config = raw_score["expected_action"]
        if isinstance(action_config, str):
            return score | {"kind": "expected_action", "metric": action_config}
        if not isinstance(action_config, dict):
            raise SpecValidationError(
                f"benchmark.<id>.score.{name}.expected_action must be a string or mapping"
            )
        return score | {
            "kind": "expected_action",
            "metric": action_config.get("metric", "selection"),
            "observed_kind": action_config.get("observed_kind", "tool"),
        }
    return score | {"kind": "python", "target": raw_score["python"]}


def _normalize_score_common(name: str, raw_score: dict[str, Any]) -> dict[str, Any]:
    score: dict[str, Any] = {"name": name}
    for source, target in (
        ("semantic", "semantic_type"),
        ("semantic_type", "semantic_type"),
        ("unit", "unit"),
        ("role", "role"),
        ("optional", "optional"),
        ("span", "span"),
    ):
        if source in raw_score:
            score[target] = raw_score[source]
    if "direction" in raw_score:
        score["direction"] = raw_score["direction"]
    elif "goal" in raw_score:
        score["direction"] = raw_score["goal"]
    return score


def _normalize_report_section(raw_report: Any) -> dict[str, Any]:
    if not isinstance(raw_report, dict):
        raise SpecValidationError("benchmark.<id>.report must be a mapping")

    report: dict[str, Any] = {}
    if "leaderboard" in raw_report:
        report["leaderboard"] = _normalize_leaderboard_report(raw_report["leaderboard"])
    if "matrix" in raw_report:
        report["case_matrix"] = _normalize_matrix_report(raw_report["matrix"])
    if "compare" in raw_report:
        report["comparisons"] = _normalize_compare_report(raw_report["compare"])
    for key in ("distributions",):
        if key in raw_report:
            report[key] = raw_report[key]
    return report


def _normalize_leaderboard_report(raw_leaderboard: Any) -> dict[str, Any]:
    if not isinstance(raw_leaderboard, dict):
        raise SpecValidationError("benchmark.<id>.report.leaderboard must be a mapping")
    return {"metrics": _normalize_report_metrics(raw_leaderboard.get("show", {}))}


def _normalize_matrix_report(raw_matrix: Any) -> dict[str, Any]:
    if isinstance(raw_matrix, str):
        return {"semantic_type": raw_matrix}
    if not isinstance(raw_matrix, dict):
        raise SpecValidationError("benchmark.<id>.report.matrix must be a string or mapping")
    semantic_type = raw_matrix.get("metric", raw_matrix.get("semantic_type"))
    if semantic_type is None:
        raise SpecValidationError("benchmark.<id>.report.matrix must define metric")
    return {"semantic_type": semantic_type}


def _normalize_compare_report(raw_compare: Any) -> list[dict[str, Any]]:
    if not isinstance(raw_compare, dict):
        raise SpecValidationError("benchmark.<id>.report.compare must be a mapping")

    comparisons: list[dict[str, Any]] = []
    for comparison_key, raw_config in raw_compare.items():
        baseline, separator, candidate = str(comparison_key).partition("->")
        if not separator:
            raise SpecValidationError(
                "benchmark.<id>.report.compare keys must use '<baseline> -> <candidate>'"
            )
        if raw_config is None:
            raw_config = {}
        if not isinstance(raw_config, dict):
            raise SpecValidationError("benchmark.<id>.report.compare entries must be mappings")
        comparisons.append(
            {
                "baseline": baseline.strip(),
                "candidate": candidate.strip(),
                "metrics": _normalize_report_metrics(raw_config.get("show", {})),
            }
        )
    return comparisons


def _normalize_report_metrics(raw_metrics: Any) -> list[dict[str, Any]]:
    if isinstance(raw_metrics, list):
        metrics: list[dict[str, Any]] = []
        for raw_metric in raw_metrics:
            if not isinstance(raw_metric, dict):
                raise SpecValidationError("report metric entries must be mappings")
            metric = dict(raw_metric)
            if "metric" in metric:
                metric["semantic_type"] = metric.pop("metric")
            if "aggregate" in metric:
                metric["fn"] = metric.pop("aggregate")
            metrics.append(metric)
        return metrics
    if not isinstance(raw_metrics, dict):
        raise SpecValidationError("report show must be a mapping or list")

    metrics = []
    for name, raw_metric in raw_metrics.items():
        if isinstance(raw_metric, str):
            metrics.append({"name": str(name), "semantic_type": raw_metric, "fn": "mean"})
            continue
        if not isinstance(raw_metric, dict):
            raise SpecValidationError("report show entries must be strings or mappings")
        metric = {"name": str(name)}
        semantic_type = raw_metric.get("metric", raw_metric.get("semantic_type"))
        aggregate = raw_metric.get("aggregate", raw_metric.get("fn"))
        if semantic_type is None or aggregate is None:
            raise SpecValidationError("report show entries must define metric and aggregate")
        metric["semantic_type"] = semantic_type
        metric["fn"] = aggregate
        metrics.append(metric)
    return metrics


def _benchmark_dataset_to_yaml_view(dataset: DatasetSpec) -> dict[str, Any]:
    view: dict[str, Any] = {"cases": [case_to_yaml_view(case) for case in dataset.cases]}
    if dataset.id is not None:
        view["id"] = dataset.id
    if dataset.source is not None:
        view["source"] = dataset.source
    if dataset.version is not None:
        view["version"] = dataset.version
    if dataset.metadata:
        view["metadata"] = dataset.metadata
    defaults = _dataset_case_defaults_to_yaml_view(dataset.case_defaults)
    if defaults:
        view["case_defaults"] = defaults
    return view


def _task_to_yaml_view(task: TaskSpec) -> dict[str, Any] | str:
    if task.kind == "python" and not task.module_search_paths:
        return {"python": task.target}
    view: dict[str, Any] = {"kind": task.kind, "target": task.target}
    if task.module_search_paths:
        view["module_search_paths"] = list(task.module_search_paths)
    return view


def _variants_to_yaml_view(variants: list[Variant]) -> dict[str, Any]:
    rendered: dict[str, Any] = {}
    for variant in variants:
        variant_view: dict[str, Any] = {}
        if variant.label is not None:
            variant_view["label"] = variant.label
        if variant.factors:
            variant_view["factors"] = {
                factor.name: _factor_to_yaml_view(factor) for factor in variant.factors
            }
        rendered[variant.id] = variant_view
    return rendered


def _factor_to_yaml_view(factor: FactorValue) -> Any:
    if factor.semantic_type is None and not factor.optimize:
        return factor.value
    view: dict[str, Any] = {"value": factor.value}
    if factor.semantic_type is not None:
        view["semantic"] = factor.semantic_type
    if factor.optimize:
        view["optimize"] = True
    return view


def _scoring_to_yaml_view(scoring: list[ScoringSpec]) -> dict[str, Any]:
    return {score.name: _score_to_yaml_view(score) for score in scoring}


def _score_to_yaml_view(score: ScoringSpec) -> dict[str, Any]:
    view: dict[str, Any] = {}
    if isinstance(score, OutputMetricScorer):
        view["value"] = score.path
    elif isinstance(score, PassFailScorer):
        view["pass"] = score.path
    elif isinstance(score, ExactScorer):
        view["exact"] = {"actual": score.actual, "expected": score.expected}
    elif isinstance(score, SchemaScorer):
        view["schema"] = score.schema_definition
        if score.path != "output":
            view["from"] = score.path
    elif isinstance(score, ExpectedActionScorer):
        view["expected_action"] = {
            "metric": score.metric,
            "observed_kind": score.observed_kind,
        }
    else:
        assert isinstance(score, PythonScorer)
        view["python"] = score.target
    view["semantic"] = score.semantic_type
    if score.unit is not None:
        view["unit"] = score.unit
    if score.direction is not None:
        view["goal"] = score.direction.value
    if score.role is not None:
        view["role"] = score.role.value
    if score.optional:
        view["optional"] = True
    if score.span is not None:
        view["span"] = _compact_model_dump(score.span)
    return view


def _report_to_yaml_view(report: ReportSpec) -> dict[str, Any]:
    view: dict[str, Any] = {}
    if report.leaderboard.metrics:
        view["leaderboard"] = {
            "show": {
                metric.name: {
                    "metric": metric.semantic_type,
                    "aggregate": metric.fn,
                }
                for metric in report.leaderboard.metrics
            }
        }
    if report.case_matrix.semantic_type != "coverage.ratio":
        view["matrix"] = report.case_matrix.semantic_type
    if report.comparisons:
        view["compare"] = {
            f"{comparison.baseline} -> {comparison.candidate}": (
                {"show": _report_metric_map(comparison.metrics)} if comparison.metrics else {}
            )
            for comparison in report.comparisons
        }
    if report.distributions:
        view["distributions"] = [
            _compact_model_dump(distribution) for distribution in report.distributions
        ]
    return view


def _report_metric_map(metrics: tuple[MetricAggregation, ...]) -> dict[str, Any]:
    return {
        metric.name: {
            "metric": metric.semantic_type,
            "aggregate": metric.fn,
        }
        for metric in metrics
    }


def _semantic_registry_delta_to_yaml_view(registry: SemanticRegistry) -> dict[str, Any]:
    default_registry = DEFAULT_SEMANTIC_REGISTRY
    aliases = {
        alias: target
        for alias, target in registry.aliases.items()
        if default_registry.aliases.get(alias) != target
    }
    types: dict[str, Any] = {}
    for semantic_type, info in registry.types.items():
        default_info = default_registry.types.get(semantic_type)
        if default_info == info:
            continue
        payload: dict[str, Any] = {}
        if info.parent is not None and (default_info is None or default_info.parent != info.parent):
            payload["parent"] = info.parent
        if info.description is not None and (
            default_info is None or default_info.description != info.description
        ):
            payload["description"] = info.description
        if info.unit is not None and (default_info is None or default_info.unit != info.unit):
            payload["unit"] = info.unit
        if info.value_shape is not None and (
            default_info is None or default_info.value_shape != info.value_shape
        ):
            payload["shape"] = info.value_shape
        if info.aliases and (default_info is None or default_info.aliases != info.aliases):
            payload["aliases"] = list(info.aliases)
        if info.deprecated and (default_info is None or default_info.deprecated != info.deprecated):
            payload["deprecated"] = True
        if info.stability is not None and (
            default_info is None or default_info.stability != info.stability
        ):
            payload["stability"] = info.stability.value
        if info.privacy is not None and (
            default_info is None or default_info.privacy != info.privacy
        ):
            payload["privacy"] = info.privacy.value
        if info.cardinality is not None and (
            default_info is None or default_info.cardinality != info.cardinality
        ):
            payload["cardinality"] = info.cardinality.value
        if info.aggregation is not None and (
            default_info is None or default_info.aggregation != info.aggregation
        ):
            payload["aggregation"] = info.aggregation.value
        if info.tags and (default_info is None or default_info.tags != info.tags):
            payload["tags"] = dict(info.tags)
        types[semantic_type] = payload
    view: dict[str, Any] = {}
    if registry.version != default_registry.version:
        view["version"] = registry.version
    if aliases:
        view["aliases"] = aliases
    if types:
        view["types"] = types
    return view


def _compact_model_dump(model: BaseModel) -> dict[str, Any]:
    return model.model_dump(mode="json", exclude_none=True, exclude_defaults=True)


def _dataset_case_defaults_to_yaml_view(defaults: CaseDefaults) -> dict[str, Any]:
    view: dict[str, Any] = {}
    if defaults.input is not None:
        view["input"] = defaults.input
    if defaults.expected is not None:
        view["expected"] = defaults.expected
    if defaults.metadata:
        view["metadata"] = defaults.metadata
    if defaults.tags:
        view["tags"] = list(defaults.tags)
    if defaults.attachments:
        view["attachments"] = [
            _artifact_ref_to_yaml_view(attachment) for attachment in defaults.attachments
        ]
    return view


def _artifact_ref_to_yaml_view(attachment: Any) -> dict[str, Any]:
    view: dict[str, Any] = {
        "id": attachment.id,
        "name": attachment.name,
    }
    if attachment.media_type is not None:
        view["media_type"] = attachment.media_type
    if attachment.value is not None:
        view["value"] = attachment.value
    if attachment.span_id is not None:
        view["span_id"] = attachment.span_id
    if attachment.tags:
        view["tags"] = attachment.tags
    return view


__all__ = (
    "_normalize_benchmark_dsl",
    "benchmark_spec_to_yaml_view",
)
