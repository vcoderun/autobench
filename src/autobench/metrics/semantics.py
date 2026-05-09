from __future__ import annotations as _annotations

from typing import Any, Final, Literal, TypeAlias

from pydantic import BaseModel, Field

LLMSemanticType: TypeAlias = Literal[
    "llm.tokens.input",
    "llm.tokens.output",
    "llm.tokens.total",
    "llm.model.name",
    "llm.provider",
    "llm.temperature",
    "llm.optimizer.model",
    "llm.student.model",
]

CostSemanticType: TypeAlias = Literal[
    "money.cost",
    "optimization.cost",
    "serving.cost",
    "lifetime.cost",
]

TimeSemanticType: TypeAlias = Literal["time.latency"]

ResultSemanticType: TypeAlias = Literal["result.success"]

QualitySemanticType: TypeAlias = Literal[
    "quality.score",
    "quality.correctness",
    "coverage.ratio",
]

AgentSemanticType: TypeAlias = Literal[
    "agent.version",
    "agent.orchestration.quality",
    "agent.tool.name",
    "agent.tool.version",
    "agent.tool_call.quality",
    "agent.serving.volume",
]

PromptSemanticType: TypeAlias = Literal["prompt.version"]

DatasetSemanticType: TypeAlias = Literal["dataset.version"]

KnownSemanticType: TypeAlias = (
    LLMSemanticType
    | CostSemanticType
    | TimeSemanticType
    | ResultSemanticType
    | QualitySemanticType
    | AgentSemanticType
    | PromptSemanticType
    | DatasetSemanticType
)

SemanticType: TypeAlias = KnownSemanticType | str


class Semantic:
    LLM_TOKENS_INPUT: Final[str] = "llm.tokens.input"
    LLM_TOKENS_OUTPUT: Final[str] = "llm.tokens.output"
    LLM_TOKENS_TOTAL: Final[str] = "llm.tokens.total"
    LLM_MODEL_NAME: Final[str] = "llm.model.name"
    LLM_PROVIDER: Final[str] = "llm.provider"
    LLM_TEMPERATURE: Final[str] = "llm.temperature"
    LLM_OPTIMIZER_MODEL: Final[str] = "llm.optimizer.model"
    LLM_STUDENT_MODEL: Final[str] = "llm.student.model"
    MONEY_COST: Final[str] = "money.cost"
    OPTIMIZATION_COST: Final[str] = "optimization.cost"
    SERVING_COST: Final[str] = "serving.cost"
    LIFETIME_COST: Final[str] = "lifetime.cost"
    TIME_LATENCY: Final[str] = "time.latency"
    RESULT_SUCCESS: Final[str] = "result.success"
    QUALITY_SCORE: Final[str] = "quality.score"
    QUALITY_CORRECTNESS: Final[str] = "quality.correctness"
    COVERAGE_RATIO: Final[str] = "coverage.ratio"
    AGENT_VERSION: Final[str] = "agent.version"
    AGENT_ORCHESTRATION_QUALITY: Final[str] = "agent.orchestration.quality"
    AGENT_TOOL_NAME: Final[str] = "agent.tool.name"
    AGENT_TOOL_VERSION: Final[str] = "agent.tool.version"
    AGENT_TOOL_CALL_QUALITY: Final[str] = "agent.tool_call.quality"
    AGENT_SERVING_VOLUME: Final[str] = "agent.serving.volume"
    PROMPT_VERSION: Final[str] = "prompt.version"
    DATASET_VERSION: Final[str] = "dataset.version"


class SemanticTypeInfo(BaseModel):
    id: str
    parent: SemanticType | None = None
    unit: str | None = None
    value_shape: str | None = None
    aliases: list[str] = Field(default_factory=list)
    deprecated: bool = False
    tags: dict[str, str] = Field(default_factory=dict)


class SemanticRegistry(BaseModel):
    version: int = 1
    types: dict[str, SemanticTypeInfo] = Field(default_factory=dict)
    aliases: dict[str, str] = Field(default_factory=dict)

    @classmethod
    def with_defaults(cls) -> SemanticRegistry:
        types = {
            Semantic.LLM_TOKENS_INPUT: SemanticTypeInfo(
                id=Semantic.LLM_TOKENS_INPUT,
                unit="tokens",
                value_shape="integer",
            ),
            Semantic.LLM_TOKENS_OUTPUT: SemanticTypeInfo(
                id=Semantic.LLM_TOKENS_OUTPUT,
                unit="tokens",
                value_shape="integer",
            ),
            Semantic.LLM_TOKENS_TOTAL: SemanticTypeInfo(
                id=Semantic.LLM_TOKENS_TOTAL,
                unit="tokens",
                value_shape="integer",
            ),
            Semantic.LLM_MODEL_NAME: SemanticTypeInfo(
                id=Semantic.LLM_MODEL_NAME,
                value_shape="string",
            ),
            Semantic.LLM_PROVIDER: SemanticTypeInfo(
                id=Semantic.LLM_PROVIDER,
                value_shape="string",
            ),
            Semantic.LLM_TEMPERATURE: SemanticTypeInfo(
                id=Semantic.LLM_TEMPERATURE,
                value_shape="number",
            ),
            Semantic.LLM_OPTIMIZER_MODEL: SemanticTypeInfo(
                id=Semantic.LLM_OPTIMIZER_MODEL,
                parent=Semantic.LLM_MODEL_NAME,
                value_shape="string",
            ),
            Semantic.LLM_STUDENT_MODEL: SemanticTypeInfo(
                id=Semantic.LLM_STUDENT_MODEL,
                parent=Semantic.LLM_MODEL_NAME,
                value_shape="string",
            ),
            Semantic.MONEY_COST: SemanticTypeInfo(
                id=Semantic.MONEY_COST,
                unit="usd",
                value_shape="number",
            ),
            Semantic.OPTIMIZATION_COST: SemanticTypeInfo(
                id=Semantic.OPTIMIZATION_COST,
                parent=Semantic.MONEY_COST,
                unit="usd",
                value_shape="number",
            ),
            Semantic.SERVING_COST: SemanticTypeInfo(
                id=Semantic.SERVING_COST,
                parent=Semantic.MONEY_COST,
                unit="usd",
                value_shape="number",
            ),
            Semantic.LIFETIME_COST: SemanticTypeInfo(
                id=Semantic.LIFETIME_COST,
                parent=Semantic.MONEY_COST,
                unit="usd",
                value_shape="number",
            ),
            Semantic.TIME_LATENCY: SemanticTypeInfo(
                id=Semantic.TIME_LATENCY,
                unit="s",
                value_shape="number",
            ),
            Semantic.RESULT_SUCCESS: SemanticTypeInfo(
                id=Semantic.RESULT_SUCCESS,
                value_shape="boolean",
            ),
            Semantic.QUALITY_SCORE: SemanticTypeInfo(
                id=Semantic.QUALITY_SCORE,
                value_shape="number",
            ),
            Semantic.QUALITY_CORRECTNESS: SemanticTypeInfo(
                id=Semantic.QUALITY_CORRECTNESS,
                parent=Semantic.QUALITY_SCORE,
                value_shape="number",
            ),
            Semantic.COVERAGE_RATIO: SemanticTypeInfo(
                id=Semantic.COVERAGE_RATIO,
                value_shape="number",
            ),
            Semantic.AGENT_VERSION: SemanticTypeInfo(
                id=Semantic.AGENT_VERSION,
                value_shape="string",
            ),
            Semantic.AGENT_ORCHESTRATION_QUALITY: SemanticTypeInfo(
                id=Semantic.AGENT_ORCHESTRATION_QUALITY,
                parent=Semantic.QUALITY_SCORE,
                value_shape="number",
            ),
            Semantic.AGENT_TOOL_NAME: SemanticTypeInfo(
                id=Semantic.AGENT_TOOL_NAME,
                value_shape="string",
            ),
            Semantic.AGENT_TOOL_VERSION: SemanticTypeInfo(
                id=Semantic.AGENT_TOOL_VERSION,
                value_shape="string",
            ),
            Semantic.AGENT_TOOL_CALL_QUALITY: SemanticTypeInfo(
                id=Semantic.AGENT_TOOL_CALL_QUALITY,
                parent=Semantic.QUALITY_SCORE,
                value_shape="number",
            ),
            Semantic.AGENT_SERVING_VOLUME: SemanticTypeInfo(
                id=Semantic.AGENT_SERVING_VOLUME,
                value_shape="integer",
            ),
            Semantic.PROMPT_VERSION: SemanticTypeInfo(
                id=Semantic.PROMPT_VERSION,
                value_shape="string",
            ),
            Semantic.DATASET_VERSION: SemanticTypeInfo(
                id=Semantic.DATASET_VERSION,
                value_shape="string",
            ),
            "ai.codegen.spec_model": SemanticTypeInfo(
                id="ai.codegen.spec_model",
                parent=Semantic.LLM_MODEL_NAME,
                value_shape="string",
                tags={"role": "spec_generator"},
            ),
            "ai.codegen.exploration_model": SemanticTypeInfo(
                id="ai.codegen.exploration_model",
                parent=Semantic.LLM_MODEL_NAME,
                value_shape="string",
                tags={"role": "explorer"},
            ),
        }
        aliases = {
            "quality.answer": Semantic.QUALITY_SCORE,
            "agent.tool_call.correctness": Semantic.AGENT_TOOL_CALL_QUALITY,
        }
        return cls(types=types, aliases=aliases)

    def info_for(self, semantic_type: str) -> SemanticTypeInfo | None:
        normalized = self.normalize(semantic_type)
        if normalized is None:
            return None
        return self.types.get(normalized)

    def normalize(self, semantic_type: str | None) -> str | None:
        if semantic_type is None:
            return None
        alias_target = self.aliases.get(semantic_type)
        if alias_target is not None:
            return alias_target
        info = self.types.get(semantic_type)
        if info is not None and info.deprecated and info.parent is not None:
            return str(info.parent)
        return semantic_type

    def parent_of(self, semantic_type: str | None) -> str | None:
        normalized = self.normalize(semantic_type)
        if normalized is None:
            return None
        info = self.types.get(normalized)
        if info is None or info.parent is None:
            return None
        return self.normalize(str(info.parent))

    def is_a(self, child: str | None, parent: str | None) -> bool:
        if child is None or parent is None:
            return False
        normalized_child = self.normalize(child)
        normalized_parent = self.normalize(parent)
        if normalized_child == normalized_parent:
            return True

        current = self.parent_of(normalized_child)
        while current is not None:
            if current == normalized_parent:
                return True
            current = self.parent_of(current)
        return False


DEFAULT_SEMANTIC_REGISTRY: Final[SemanticRegistry] = SemanticRegistry.with_defaults()


def semantic_registry_to_yaml_view(registry: SemanticRegistry) -> dict[str, Any]:
    types_view = {
        semantic_id: _semantic_type_yaml_view(info) for semantic_id, info in registry.types.items()
    }
    return {
        "record": {
            "type": "semantic_registry",
            "version": registry.version,
        },
        "semantic_registry": {
            "version": registry.version,
            "types": types_view,
            "aliases": dict(registry.aliases),
        },
    }


def semantic_registry_payload_from_yaml_view(raw: Any) -> dict[str, Any]:
    registry = raw
    if isinstance(raw, dict):
        record_header = raw.get("record")
        if isinstance(record_header, dict) and record_header.get("type") == "semantic_registry":
            registry = raw.get("semantic_registry")
    if not isinstance(registry, dict):
        raise TypeError("semantic_registry must be a mapping")

    raw_types = registry.get("types", {})
    raw_aliases = registry.get("aliases", {})
    if not isinstance(raw_types, dict):
        raise TypeError("semantic_registry.types must be a mapping")
    if not isinstance(raw_aliases, dict):
        raise TypeError("semantic_registry.aliases must be a mapping")

    resolved_types: dict[str, dict[str, Any]] = {}
    for semantic_id, raw_type in raw_types.items():
        if not isinstance(raw_type, dict):
            raise TypeError(f"semantic_registry.types.{semantic_id} must be a mapping")
        payload = dict(raw_type)
        payload["id"] = str(payload.get("id", semantic_id))
        if "shape" in payload and "value_shape" not in payload:
            payload["value_shape"] = payload.pop("shape")
        resolved_types[str(semantic_id)] = payload

    return {
        "version": registry.get("version", 1),
        "types": resolved_types,
        "aliases": dict(raw_aliases),
    }


def _semantic_type_yaml_view(info: SemanticTypeInfo) -> dict[str, Any]:
    view: dict[str, Any] = {}
    if info.parent is not None:
        view["parent"] = info.parent
    if info.unit is not None:
        view["unit"] = info.unit
    if info.value_shape is not None:
        view["shape"] = info.value_shape
    if info.aliases:
        view["aliases"] = list(info.aliases)
    if info.deprecated:
        view["deprecated"] = True
    if info.tags:
        view["tags"] = dict(info.tags)
    return view


__all__ = (
    "AgentSemanticType",
    "CostSemanticType",
    "DEFAULT_SEMANTIC_REGISTRY",
    "DatasetSemanticType",
    "KnownSemanticType",
    "LLMSemanticType",
    "PromptSemanticType",
    "QualitySemanticType",
    "ResultSemanticType",
    "Semantic",
    "SemanticRegistry",
    "SemanticType",
    "SemanticTypeInfo",
    "semantic_registry_payload_from_yaml_view",
    "semantic_registry_to_yaml_view",
    "TimeSemanticType",
)
