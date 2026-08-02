from __future__ import annotations as _annotations

from pydantic import BaseModel, Field

from autobench.metrics.semantics import Semantic, SemanticRegistry, SemanticTypeInfo
from autobench.reports.reporting import MetricAggregation


class MetricPack(BaseModel):
    id: str
    semantic_registry_delta: SemanticRegistry = Field(default_factory=SemanticRegistry)
    scorer_factories: dict[str, str] = Field(default_factory=dict)
    default_report_metrics: tuple[MetricAggregation, ...] = ()
    feedback_extractors: tuple[str, ...] = ()


class MetricPackRegistry(BaseModel):
    packs: dict[str, MetricPack] = Field(default_factory=dict)

    def register(self, pack: MetricPack) -> None:
        self.packs[pack.id] = pack

    def get(self, pack_id: str) -> MetricPack | None:
        return self.packs.get(pack_id)

    def require(self, pack_id: str) -> MetricPack:
        pack = self.get(pack_id)
        if pack is None:
            raise KeyError(f"Unknown metric pack: {pack_id}")
        return pack

    def names(self) -> tuple[str, ...]:
        return tuple(sorted(self.packs))

    def semantic_registry_for(self, pack_ids: list[str]) -> SemanticRegistry:
        merged = SemanticRegistry()
        for pack_id in pack_ids:
            pack = self.require(pack_id)
            merged.types.update(pack.semantic_registry_delta.types)
            merged.aliases.update(pack.semantic_registry_delta.aliases)
        return merged


def builtin_metric_pack_registry() -> MetricPackRegistry:
    registry = MetricPackRegistry()
    for pack in (
        _agentic_pack(),
        _structured_output_pack(),
        _llm_usage_pack(),
        _performance_pack(),
    ):
        registry.register(pack)
    return registry


def _agentic_pack() -> MetricPack:
    return MetricPack(
        id="agentic",
        semantic_registry_delta=SemanticRegistry(
            types={
                Semantic.AGENT_TASK_COMPLETION: SemanticTypeInfo(
                    id=Semantic.AGENT_TASK_COMPLETION,
                    parent=Semantic.RESULT_SUCCESS,
                    value_shape="boolean",
                ),
                Semantic.AGENT_TOOL_SELECTION_CORRECTNESS: SemanticTypeInfo(
                    id=Semantic.AGENT_TOOL_SELECTION_CORRECTNESS,
                    parent=Semantic.QUALITY_CORRECTNESS,
                    value_shape="number",
                ),
                Semantic.AGENT_TOOL_ARGUMENT_CORRECTNESS: SemanticTypeInfo(
                    id=Semantic.AGENT_TOOL_ARGUMENT_CORRECTNESS,
                    parent=Semantic.QUALITY_CORRECTNESS,
                    value_shape="number",
                ),
                Semantic.AGENT_TOOL_SEQUENCE_CORRECTNESS: SemanticTypeInfo(
                    id=Semantic.AGENT_TOOL_SEQUENCE_CORRECTNESS,
                    parent=Semantic.QUALITY_CORRECTNESS,
                    value_shape="number",
                ),
            },
        ),
        scorer_factories={
            "tool_selection": "expected_action.selection",
            "tool_arguments": "expected_action.arguments",
            "tool_sequence": "expected_action.sequence",
        },
        default_report_metrics=(
            MetricAggregation(
                name="task_completion",
                semantic_type=Semantic.AGENT_TASK_COMPLETION,
                fn="mean",
            ),
            MetricAggregation(
                name="tool_selection",
                semantic_type=Semantic.AGENT_TOOL_SELECTION_CORRECTNESS,
                fn="mean",
            ),
        ),
        feedback_extractors=("assertion", "score_reason", "span_error"),
    )


def _structured_output_pack() -> MetricPack:
    return MetricPack(
        id="structured_output",
        default_report_metrics=(
            MetricAggregation(
                name="output_validity",
                semantic_type=Semantic.AGENT_OUTPUT_STRUCTURE_VALIDITY,
                fn="mean",
            ),
        ),
    )


def _llm_usage_pack() -> MetricPack:
    return MetricPack(
        id="llm_usage",
        default_report_metrics=(
            MetricAggregation(
                name="input_tokens",
                semantic_type=Semantic.LLM_TOKENS_INPUT,
                fn="sum",
            ),
            MetricAggregation(
                name="output_tokens",
                semantic_type=Semantic.LLM_TOKENS_OUTPUT,
                fn="sum",
            ),
            MetricAggregation(name="cost", semantic_type=Semantic.MONEY_COST, fn="sum"),
        ),
    )


def _performance_pack() -> MetricPack:
    return MetricPack(
        id="performance",
        default_report_metrics=(
            MetricAggregation(name="latency", semantic_type=Semantic.TIME_LATENCY, fn="mean"),
            MetricAggregation(name="success", semantic_type=Semantic.RESULT_SUCCESS, fn="mean"),
        ),
    )


DEFAULT_METRIC_PACKS = builtin_metric_pack_registry()


__all__ = (
    "DEFAULT_METRIC_PACKS",
    "MetricPack",
    "MetricPackRegistry",
    "builtin_metric_pack_registry",
)
