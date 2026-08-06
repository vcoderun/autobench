from __future__ import annotations as _annotations

from enum import StrEnum
from typing import Any, Final, Literal, TypeAlias

from pydantic import BaseModel, Field

LLMSemanticType: TypeAlias = Literal[
    "llm.tokens.input",
    "llm.tokens.output",
    "llm.tokens.total",
    "llm.tokens.cached_input",
    "llm.tokens.cache_write",
    "llm.tokens.reasoning_output",
    "llm.model.name",
    "llm.model.requested",
    "llm.model.response",
    "llm.provider",
    "llm.provider.name",
    "llm.temperature",
    "llm.optimizer.model",
    "llm.student.model",
    "llm.request.count",
]

CostSemanticType: TypeAlias = Literal[
    "money.cost",
    "optimization.cost",
    "serving.cost",
    "lifetime.cost",
]

TimeSemanticType: TypeAlias = Literal[
    "time.latency",
    "time.first_chunk",
    "time.critical_path",
]

HTTPSemanticType: TypeAlias = Literal[
    "http.request.method",
    "http.request.scheme",
    "http.request.host",
    "http.request.port",
    "http.request.path",
    "http.request.path_hash",
    "http.request.headers",
    "http.request.body.size",
    "http.response.status_code",
    "http.response.headers",
    "http.response.body.size",
]

NetworkSemanticType: TypeAlias = Literal["network.protocol.version"]

ErrorSemanticType: TypeAlias = Literal["error.type"]

OperationEvidenceSemanticType: TypeAlias = Literal[
    "operation.count",
    "operation.depth.max",
    "operation.fan_out.max",
    "operation.incomplete.count",
    "operation.parallelism",
    "operation.retry.count",
    "operation.retry.recovered.count",
    "operation.first_attempt.success",
]

WorkflowEvidenceSemanticType: TypeAlias = Literal[
    "validation.count",
    "validation.failure.count",
    "validation.failure.rate",
    "approval.count",
    "approval.wait",
    "tool.call.count",
    "tool.call.success.count",
    "tool.call.failure.count",
    "tool.call.arguments.present.count",
]

EvidenceReferenceSemanticType: TypeAlias = Literal[
    "artifact.reference.count",
    "asset.reference.count",
    "message.input.count",
    "message.output.count",
    "message.growth",
]

ResultSemanticType: TypeAlias = Literal["result.success"]

QualitySemanticType: TypeAlias = Literal[
    "quality.score",
    "quality.correctness",
    "coverage.ratio",
]

AgentSemanticType: TypeAlias = Literal[
    "agent.id",
    "agent.name",
    "agent.version",
    "agent.task.completion",
    "agent.goal.accuracy",
    "agent.plan.quality",
    "agent.plan.adherence",
    "agent.step.efficiency",
    "agent.orchestration.quality",
    "agent.tool.name",
    "agent.tool.version",
    "agent.tool.selection.correctness",
    "agent.tool.argument.correctness",
    "agent.tool.sequence.correctness",
    "agent.tool_call.quality",
    "agent.serving.volume",
    "agent.output.correctness",
    "agent.output.structure.validity",
]

ToolSemanticType: TypeAlias = Literal[
    "tool.name",
    "tool.type",
    "tool.version",
    "tool.definitions",
    "tool.call.id",
    "tool.call.arguments",
    "tool.call.result",
    "tool.call.quality",
]

WorkflowSemanticType: TypeAlias = Literal["workflow.name"]

ConversationSemanticType: TypeAlias = Literal["conversation.id"]

RetrievalSemanticType: TypeAlias = Literal[
    "retrieval.query",
    "retrieval.documents",
    "retrieval.documents.count",
]

EvaluationSemanticType: TypeAlias = Literal[
    "evaluation.name",
    "evaluation.score",
    "evaluation.label",
    "evaluation.explanation",
]

ContentSemanticType: TypeAlias = Literal[
    "message.input",
    "message.output",
    "prompt.system",
    "artifact.content",
]

OperationSemanticType: TypeAlias = Literal[
    "operation.name",
    "operation.input",
    "operation.output",
]

LifecycleEventSemanticType: TypeAlias = Literal[
    "stream.first_chunk",
    "stream.completed",
    "stream.partial",
    "stream.failed",
    "operation.retry",
    "operation.repair",
    "operation.deferred",
    "operation.deferred.resolved",
    "validation.failure",
    "approval.requested",
    "tool.call.requested",
]

RuntimeSemanticType: TypeAlias = Literal[
    "factor.value",
    "event.occurrence",
    "diagnostic.event",
    "error.exception",
]

PromptSemanticType: TypeAlias = Literal["prompt.version"]

DatasetSemanticType: TypeAlias = Literal["dataset.version"]

AssetSemanticType: TypeAlias = Literal[
    "output_schema.version",
    "capability.version",
    "guardrail.version",
    "handoff.version",
    "policy.version",
    "toolset.version",
    "asset.rendering.version",
    "asset.external.version",
    "asset.deployment.label",
]

KnownSemanticType: TypeAlias = (
    LLMSemanticType
    | CostSemanticType
    | TimeSemanticType
    | ResultSemanticType
    | QualitySemanticType
    | AgentSemanticType
    | ToolSemanticType
    | WorkflowSemanticType
    | ConversationSemanticType
    | RetrievalSemanticType
    | EvaluationSemanticType
    | ContentSemanticType
    | OperationSemanticType
    | LifecycleEventSemanticType
    | RuntimeSemanticType
    | PromptSemanticType
    | DatasetSemanticType
    | AssetSemanticType
    | OperationEvidenceSemanticType
    | WorkflowEvidenceSemanticType
    | EvidenceReferenceSemanticType
    | HTTPSemanticType
    | NetworkSemanticType
    | ErrorSemanticType
)

SemanticType: TypeAlias = KnownSemanticType | str


class Semantic:
    LLM_TOKENS_INPUT: Final[str] = "llm.tokens.input"
    LLM_TOKENS_OUTPUT: Final[str] = "llm.tokens.output"
    LLM_TOKENS_TOTAL: Final[str] = "llm.tokens.total"
    LLM_TOKENS_CACHED_INPUT: Final[str] = "llm.tokens.cached_input"
    LLM_TOKENS_CACHE_WRITE: Final[str] = "llm.tokens.cache_write"
    LLM_TOKENS_REASONING_OUTPUT: Final[str] = "llm.tokens.reasoning_output"
    LLM_MODEL_NAME: Final[str] = "llm.model.name"
    LLM_MODEL_REQUESTED: Final[str] = "llm.model.requested"
    LLM_MODEL_RESPONSE: Final[str] = "llm.model.response"
    LLM_PROVIDER: Final[str] = "llm.provider"
    LLM_PROVIDER_NAME: Final[str] = "llm.provider.name"
    LLM_TEMPERATURE: Final[str] = "llm.temperature"
    LLM_OPTIMIZER_MODEL: Final[str] = "llm.optimizer.model"
    LLM_STUDENT_MODEL: Final[str] = "llm.student.model"
    LLM_REQUEST_COUNT: Final[str] = "llm.request.count"
    MONEY_COST: Final[str] = "money.cost"
    OPTIMIZATION_COST: Final[str] = "optimization.cost"
    SERVING_COST: Final[str] = "serving.cost"
    LIFETIME_COST: Final[str] = "lifetime.cost"
    TIME_LATENCY: Final[str] = "time.latency"
    TIME_FIRST_CHUNK: Final[str] = "time.first_chunk"
    TIME_CRITICAL_PATH: Final[str] = "time.critical_path"
    HTTP_REQUEST_METHOD: Final[str] = "http.request.method"
    HTTP_REQUEST_SCHEME: Final[str] = "http.request.scheme"
    HTTP_REQUEST_HOST: Final[str] = "http.request.host"
    HTTP_REQUEST_PORT: Final[str] = "http.request.port"
    HTTP_REQUEST_PATH: Final[str] = "http.request.path"
    HTTP_REQUEST_PATH_HASH: Final[str] = "http.request.path_hash"
    HTTP_REQUEST_HEADERS: Final[str] = "http.request.headers"
    HTTP_REQUEST_BODY_SIZE: Final[str] = "http.request.body.size"
    HTTP_RESPONSE_STATUS_CODE: Final[str] = "http.response.status_code"
    HTTP_RESPONSE_HEADERS: Final[str] = "http.response.headers"
    HTTP_RESPONSE_BODY_SIZE: Final[str] = "http.response.body.size"
    NETWORK_PROTOCOL_VERSION: Final[str] = "network.protocol.version"
    ERROR_TYPE: Final[str] = "error.type"
    RESULT_SUCCESS: Final[str] = "result.success"
    QUALITY_SCORE: Final[str] = "quality.score"
    QUALITY_CORRECTNESS: Final[str] = "quality.correctness"
    COVERAGE_RATIO: Final[str] = "coverage.ratio"
    AGENT_VERSION: Final[str] = "agent.version"
    AGENT_ID: Final[str] = "agent.id"
    AGENT_NAME: Final[str] = "agent.name"
    AGENT_ORCHESTRATION_QUALITY: Final[str] = "agent.orchestration.quality"
    AGENT_TOOL_NAME: Final[str] = "agent.tool.name"
    AGENT_TOOL_VERSION: Final[str] = "agent.tool.version"
    AGENT_TOOL_CALL_QUALITY: Final[str] = "agent.tool_call.quality"
    AGENT_SERVING_VOLUME: Final[str] = "agent.serving.volume"
    AGENT_TASK_COMPLETION: Final[str] = "agent.task.completion"
    AGENT_GOAL_ACCURACY: Final[str] = "agent.goal.accuracy"
    AGENT_PLAN_QUALITY: Final[str] = "agent.plan.quality"
    AGENT_PLAN_ADHERENCE: Final[str] = "agent.plan.adherence"
    AGENT_STEP_EFFICIENCY: Final[str] = "agent.step.efficiency"
    AGENT_TOOL_SELECTION_CORRECTNESS: Final[str] = "agent.tool.selection.correctness"
    AGENT_TOOL_ARGUMENT_CORRECTNESS: Final[str] = "agent.tool.argument.correctness"
    AGENT_TOOL_SEQUENCE_CORRECTNESS: Final[str] = "agent.tool.sequence.correctness"
    AGENT_OUTPUT_CORRECTNESS: Final[str] = "agent.output.correctness"
    AGENT_OUTPUT_STRUCTURE_VALIDITY: Final[str] = "agent.output.structure.validity"
    PROMPT_VERSION: Final[str] = "prompt.version"
    DATASET_VERSION: Final[str] = "dataset.version"
    OUTPUT_SCHEMA_VERSION: Final[str] = "output_schema.version"
    CAPABILITY_VERSION: Final[str] = "capability.version"
    GUARDRAIL_VERSION: Final[str] = "guardrail.version"
    HANDOFF_VERSION: Final[str] = "handoff.version"
    POLICY_VERSION: Final[str] = "policy.version"
    TOOLSET_VERSION: Final[str] = "toolset.version"
    ASSET_RENDERING_VERSION: Final[str] = "asset.rendering.version"
    ASSET_EXTERNAL_VERSION: Final[str] = "asset.external.version"
    ASSET_DEPLOYMENT_LABEL: Final[str] = "asset.deployment.label"
    TOOL_NAME: Final[str] = "tool.name"
    TOOL_TYPE: Final[str] = "tool.type"
    TOOL_VERSION: Final[str] = "tool.version"
    TOOL_DEFINITIONS: Final[str] = "tool.definitions"
    TOOL_CALL_ID: Final[str] = "tool.call.id"
    TOOL_CALL_ARGUMENTS: Final[str] = "tool.call.arguments"
    TOOL_CALL_RESULT: Final[str] = "tool.call.result"
    TOOL_CALL_QUALITY: Final[str] = "tool.call.quality"
    WORKFLOW_NAME: Final[str] = "workflow.name"
    CONVERSATION_ID: Final[str] = "conversation.id"
    RETRIEVAL_QUERY: Final[str] = "retrieval.query"
    RETRIEVAL_DOCUMENTS: Final[str] = "retrieval.documents"
    RETRIEVAL_DOCUMENTS_COUNT: Final[str] = "retrieval.documents.count"
    EVALUATION_NAME: Final[str] = "evaluation.name"
    EVALUATION_SCORE: Final[str] = "evaluation.score"
    EVALUATION_LABEL: Final[str] = "evaluation.label"
    EVALUATION_EXPLANATION: Final[str] = "evaluation.explanation"
    MESSAGE_INPUT: Final[str] = "message.input"
    MESSAGE_OUTPUT: Final[str] = "message.output"
    PROMPT_SYSTEM: Final[str] = "prompt.system"
    ARTIFACT_CONTENT: Final[str] = "artifact.content"
    OPERATION_NAME: Final[str] = "operation.name"
    OPERATION_INPUT: Final[str] = "operation.input"
    OPERATION_OUTPUT: Final[str] = "operation.output"
    STREAM_FIRST_CHUNK: Final[str] = "stream.first_chunk"
    STREAM_COMPLETED: Final[str] = "stream.completed"
    STREAM_PARTIAL: Final[str] = "stream.partial"
    STREAM_FAILED: Final[str] = "stream.failed"
    OPERATION_RETRY: Final[str] = "operation.retry"
    OPERATION_REPAIR: Final[str] = "operation.repair"
    OPERATION_DEFERRED: Final[str] = "operation.deferred"
    OPERATION_DEFERRED_RESOLVED: Final[str] = "operation.deferred.resolved"
    VALIDATION_FAILURE: Final[str] = "validation.failure"
    APPROVAL_REQUESTED: Final[str] = "approval.requested"
    TOOL_CALL_REQUESTED: Final[str] = "tool.call.requested"
    FACTOR_VALUE: Final[str] = "factor.value"
    EVENT_OCCURRENCE: Final[str] = "event.occurrence"
    DIAGNOSTIC_EVENT: Final[str] = "diagnostic.event"
    ERROR_EXCEPTION: Final[str] = "error.exception"
    OPERATION_COUNT: Final[str] = "operation.count"
    OPERATION_DEPTH_MAX: Final[str] = "operation.depth.max"
    OPERATION_FAN_OUT_MAX: Final[str] = "operation.fan_out.max"
    OPERATION_INCOMPLETE_COUNT: Final[str] = "operation.incomplete.count"
    OPERATION_PARALLELISM: Final[str] = "operation.parallelism"
    OPERATION_RETRY_COUNT: Final[str] = "operation.retry.count"
    OPERATION_RETRY_RECOVERED_COUNT: Final[str] = "operation.retry.recovered.count"
    OPERATION_FIRST_ATTEMPT_SUCCESS: Final[str] = "operation.first_attempt.success"
    VALIDATION_COUNT: Final[str] = "validation.count"
    VALIDATION_FAILURE_COUNT: Final[str] = "validation.failure.count"
    VALIDATION_FAILURE_RATE: Final[str] = "validation.failure.rate"
    APPROVAL_COUNT: Final[str] = "approval.count"
    APPROVAL_WAIT: Final[str] = "approval.wait"
    TOOL_CALL_COUNT: Final[str] = "tool.call.count"
    TOOL_CALL_SUCCESS_COUNT: Final[str] = "tool.call.success.count"
    TOOL_CALL_FAILURE_COUNT: Final[str] = "tool.call.failure.count"
    TOOL_CALL_ARGUMENTS_PRESENT_COUNT: Final[str] = "tool.call.arguments.present.count"
    ARTIFACT_REFERENCE_COUNT: Final[str] = "artifact.reference.count"
    ASSET_REFERENCE_COUNT: Final[str] = "asset.reference.count"
    MESSAGE_INPUT_COUNT: Final[str] = "message.input.count"
    MESSAGE_OUTPUT_COUNT: Final[str] = "message.output.count"
    MESSAGE_GROWTH: Final[str] = "message.growth"


class SemanticStability(StrEnum):
    STABLE = "stable"
    EVOLVING = "evolving"
    EXPERIMENTAL = "experimental"


class SemanticPrivacy(StrEnum):
    PUBLIC = "public"
    INTERNAL = "internal"
    SENSITIVE = "sensitive"
    SECRET = "secret"


class SemanticCardinality(StrEnum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    UNBOUNDED = "unbounded"


class SemanticAggregation(StrEnum):
    NONE = "none"
    SUM = "sum"
    MEAN = "mean"
    LATEST = "latest"
    ANY = "any"


class SemanticTypeInfo(BaseModel):
    id: str
    parent: SemanticType | None = None
    description: str | None = None
    unit: str | None = None
    value_shape: str | None = None
    aliases: list[str] = Field(default_factory=list)
    deprecated: bool = False
    stability: SemanticStability | None = None
    privacy: SemanticPrivacy | None = None
    cardinality: SemanticCardinality | None = None
    aggregation: SemanticAggregation | None = None
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
                description="Total input tokens reported for one model operation.",
                unit="tokens",
                value_shape="integer",
                stability=SemanticStability.STABLE,
                privacy=SemanticPrivacy.INTERNAL,
                cardinality=SemanticCardinality.LOW,
                aggregation=SemanticAggregation.SUM,
            ),
            Semantic.LLM_TOKENS_OUTPUT: SemanticTypeInfo(
                id=Semantic.LLM_TOKENS_OUTPUT,
                description="Total output tokens reported for one model operation.",
                unit="tokens",
                value_shape="integer",
                stability=SemanticStability.STABLE,
                privacy=SemanticPrivacy.INTERNAL,
                cardinality=SemanticCardinality.LOW,
                aggregation=SemanticAggregation.SUM,
            ),
            Semantic.LLM_TOKENS_TOTAL: SemanticTypeInfo(
                id=Semantic.LLM_TOKENS_TOTAL,
                description="Provider-reported total tokens when available.",
                unit="tokens",
                value_shape="integer",
                stability=SemanticStability.STABLE,
                privacy=SemanticPrivacy.INTERNAL,
                cardinality=SemanticCardinality.LOW,
                aggregation=SemanticAggregation.SUM,
            ),
            Semantic.LLM_TOKENS_CACHED_INPUT: SemanticTypeInfo(
                id=Semantic.LLM_TOKENS_CACHED_INPUT,
                parent=Semantic.LLM_TOKENS_INPUT,
                description="Input tokens read from a provider-managed cache.",
                unit="tokens",
                value_shape="integer",
                stability=SemanticStability.EVOLVING,
                privacy=SemanticPrivacy.INTERNAL,
                cardinality=SemanticCardinality.LOW,
                aggregation=SemanticAggregation.SUM,
            ),
            Semantic.LLM_TOKENS_CACHE_WRITE: SemanticTypeInfo(
                id=Semantic.LLM_TOKENS_CACHE_WRITE,
                parent=Semantic.LLM_TOKENS_INPUT,
                description="Input tokens written to a provider-managed cache.",
                unit="tokens",
                value_shape="integer",
                stability=SemanticStability.EVOLVING,
                privacy=SemanticPrivacy.INTERNAL,
                cardinality=SemanticCardinality.LOW,
                aggregation=SemanticAggregation.SUM,
            ),
            Semantic.LLM_TOKENS_REASONING_OUTPUT: SemanticTypeInfo(
                id=Semantic.LLM_TOKENS_REASONING_OUTPUT,
                parent=Semantic.LLM_TOKENS_OUTPUT,
                description="Output tokens used for provider-reported reasoning.",
                unit="tokens",
                value_shape="integer",
                stability=SemanticStability.EVOLVING,
                privacy=SemanticPrivacy.INTERNAL,
                cardinality=SemanticCardinality.LOW,
                aggregation=SemanticAggregation.SUM,
            ),
            Semantic.LLM_MODEL_NAME: SemanticTypeInfo(
                id=Semantic.LLM_MODEL_NAME,
                description="Model identity when request and response roles are not distinguished.",
                value_shape="string",
                stability=SemanticStability.STABLE,
                privacy=SemanticPrivacy.PUBLIC,
                cardinality=SemanticCardinality.MEDIUM,
                aggregation=SemanticAggregation.LATEST,
            ),
            Semantic.LLM_MODEL_REQUESTED: SemanticTypeInfo(
                id=Semantic.LLM_MODEL_REQUESTED,
                parent=Semantic.LLM_MODEL_NAME,
                description="Model requested by the caller before provider routing.",
                value_shape="string",
                stability=SemanticStability.STABLE,
                privacy=SemanticPrivacy.PUBLIC,
                cardinality=SemanticCardinality.MEDIUM,
                aggregation=SemanticAggregation.LATEST,
            ),
            Semantic.LLM_MODEL_RESPONSE: SemanticTypeInfo(
                id=Semantic.LLM_MODEL_RESPONSE,
                parent=Semantic.LLM_MODEL_NAME,
                description="Model identity reported by the serving response.",
                value_shape="string",
                stability=SemanticStability.STABLE,
                privacy=SemanticPrivacy.PUBLIC,
                cardinality=SemanticCardinality.MEDIUM,
                aggregation=SemanticAggregation.LATEST,
            ),
            Semantic.LLM_PROVIDER: SemanticTypeInfo(
                id=Semantic.LLM_PROVIDER,
                parent=Semantic.LLM_PROVIDER_NAME,
                description="Deprecated provider identity semantic.",
                value_shape="string",
                deprecated=True,
            ),
            Semantic.LLM_PROVIDER_NAME: SemanticTypeInfo(
                id=Semantic.LLM_PROVIDER_NAME,
                description="Provider or serving platform identity.",
                value_shape="string",
                stability=SemanticStability.STABLE,
                privacy=SemanticPrivacy.PUBLIC,
                cardinality=SemanticCardinality.LOW,
                aggregation=SemanticAggregation.LATEST,
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
            Semantic.LLM_REQUEST_COUNT: SemanticTypeInfo(
                id=Semantic.LLM_REQUEST_COUNT,
                description="Direct model request count at one accounting boundary.",
                unit="requests",
                value_shape="integer",
                stability=SemanticStability.STABLE,
                privacy=SemanticPrivacy.PUBLIC,
                cardinality=SemanticCardinality.LOW,
                aggregation=SemanticAggregation.SUM,
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
                description="Elapsed operation duration.",
                unit="s",
                value_shape="number",
                stability=SemanticStability.STABLE,
                privacy=SemanticPrivacy.PUBLIC,
                cardinality=SemanticCardinality.LOW,
                aggregation=SemanticAggregation.MEAN,
            ),
            Semantic.TIME_FIRST_CHUNK: SemanticTypeInfo(
                id=Semantic.TIME_FIRST_CHUNK,
                parent=Semantic.TIME_LATENCY,
                description="Elapsed time until the first streamed response chunk.",
                unit="s",
                value_shape="number",
                stability=SemanticStability.EVOLVING,
                privacy=SemanticPrivacy.PUBLIC,
                cardinality=SemanticCardinality.LOW,
                aggregation=SemanticAggregation.MEAN,
            ),
            Semantic.TIME_CRITICAL_PATH: SemanticTypeInfo(
                id=Semantic.TIME_CRITICAL_PATH,
                parent=Semantic.TIME_LATENCY,
                description="Observed monotonic trace makespan across complete operations.",
                unit="s",
                value_shape="number",
                stability=SemanticStability.STABLE,
                privacy=SemanticPrivacy.PUBLIC,
                cardinality=SemanticCardinality.LOW,
                aggregation=SemanticAggregation.MEAN,
            ),
            Semantic.HTTP_REQUEST_METHOD: SemanticTypeInfo(
                id=Semantic.HTTP_REQUEST_METHOD,
                description="HTTP request method.",
                value_shape="string",
                stability=SemanticStability.STABLE,
                privacy=SemanticPrivacy.PUBLIC,
                cardinality=SemanticCardinality.LOW,
                aggregation=SemanticAggregation.LATEST,
            ),
            Semantic.HTTP_REQUEST_SCHEME: SemanticTypeInfo(
                id=Semantic.HTTP_REQUEST_SCHEME,
                description="HTTP request URL scheme.",
                value_shape="string",
                stability=SemanticStability.STABLE,
                privacy=SemanticPrivacy.PUBLIC,
                cardinality=SemanticCardinality.LOW,
                aggregation=SemanticAggregation.LATEST,
            ),
            Semantic.HTTP_REQUEST_HOST: SemanticTypeInfo(
                id=Semantic.HTTP_REQUEST_HOST,
                description="HTTP request host without user information.",
                value_shape="string",
                stability=SemanticStability.STABLE,
                privacy=SemanticPrivacy.INTERNAL,
                cardinality=SemanticCardinality.MEDIUM,
                aggregation=SemanticAggregation.LATEST,
            ),
            Semantic.HTTP_REQUEST_PORT: SemanticTypeInfo(
                id=Semantic.HTTP_REQUEST_PORT,
                description="HTTP request destination port.",
                value_shape="integer",
                stability=SemanticStability.STABLE,
                privacy=SemanticPrivacy.PUBLIC,
                cardinality=SemanticCardinality.LOW,
                aggregation=SemanticAggregation.LATEST,
            ),
            Semantic.HTTP_REQUEST_PATH: SemanticTypeInfo(
                id=Semantic.HTTP_REQUEST_PATH,
                description="HTTP path captured only by explicit policy; query is excluded.",
                value_shape="string",
                stability=SemanticStability.EVOLVING,
                privacy=SemanticPrivacy.SENSITIVE,
                cardinality=SemanticCardinality.UNBOUNDED,
                aggregation=SemanticAggregation.NONE,
            ),
            Semantic.HTTP_REQUEST_PATH_HASH: SemanticTypeInfo(
                id=Semantic.HTTP_REQUEST_PATH_HASH,
                description="SHA-256 of the query-free HTTP request path.",
                value_shape="string",
                stability=SemanticStability.EVOLVING,
                privacy=SemanticPrivacy.INTERNAL,
                cardinality=SemanticCardinality.HIGH,
                aggregation=SemanticAggregation.NONE,
            ),
            Semantic.HTTP_REQUEST_HEADERS: SemanticTypeInfo(
                id=Semantic.HTTP_REQUEST_HEADERS,
                description="Explicitly selected request headers with mandatory secret redaction.",
                value_shape="mapping",
                stability=SemanticStability.EVOLVING,
                privacy=SemanticPrivacy.SENSITIVE,
                cardinality=SemanticCardinality.HIGH,
                aggregation=SemanticAggregation.NONE,
            ),
            Semantic.HTTP_REQUEST_BODY_SIZE: SemanticTypeInfo(
                id=Semantic.HTTP_REQUEST_BODY_SIZE,
                description="HTTP request body size when available without consuming a stream.",
                unit="By",
                value_shape="integer",
                stability=SemanticStability.STABLE,
                privacy=SemanticPrivacy.PUBLIC,
                cardinality=SemanticCardinality.LOW,
                aggregation=SemanticAggregation.SUM,
            ),
            Semantic.HTTP_RESPONSE_STATUS_CODE: SemanticTypeInfo(
                id=Semantic.HTTP_RESPONSE_STATUS_CODE,
                description="HTTP response status code.",
                value_shape="integer",
                stability=SemanticStability.STABLE,
                privacy=SemanticPrivacy.PUBLIC,
                cardinality=SemanticCardinality.LOW,
                aggregation=SemanticAggregation.LATEST,
            ),
            Semantic.HTTP_RESPONSE_HEADERS: SemanticTypeInfo(
                id=Semantic.HTTP_RESPONSE_HEADERS,
                description="Explicitly selected response headers with mandatory secret redaction.",
                value_shape="mapping",
                stability=SemanticStability.EVOLVING,
                privacy=SemanticPrivacy.SENSITIVE,
                cardinality=SemanticCardinality.HIGH,
                aggregation=SemanticAggregation.NONE,
            ),
            Semantic.HTTP_RESPONSE_BODY_SIZE: SemanticTypeInfo(
                id=Semantic.HTTP_RESPONSE_BODY_SIZE,
                description="Bytes consumed from an HTTP response body.",
                unit="By",
                value_shape="integer",
                stability=SemanticStability.STABLE,
                privacy=SemanticPrivacy.PUBLIC,
                cardinality=SemanticCardinality.LOW,
                aggregation=SemanticAggregation.SUM,
            ),
            Semantic.NETWORK_PROTOCOL_VERSION: SemanticTypeInfo(
                id=Semantic.NETWORK_PROTOCOL_VERSION,
                description="Transport-reported network protocol version.",
                value_shape="string",
                stability=SemanticStability.EVOLVING,
                privacy=SemanticPrivacy.PUBLIC,
                cardinality=SemanticCardinality.LOW,
                aggregation=SemanticAggregation.LATEST,
            ),
            Semantic.ERROR_TYPE: SemanticTypeInfo(
                id=Semantic.ERROR_TYPE,
                description="Exception or error type without an error message payload.",
                value_shape="string",
                stability=SemanticStability.STABLE,
                privacy=SemanticPrivacy.PUBLIC,
                cardinality=SemanticCardinality.MEDIUM,
                aggregation=SemanticAggregation.LATEST,
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
            Semantic.AGENT_ID: SemanticTypeInfo(
                id=Semantic.AGENT_ID,
                description="Run-local or provider agent identifier.",
                value_shape="string",
                stability=SemanticStability.EVOLVING,
                privacy=SemanticPrivacy.INTERNAL,
                cardinality=SemanticCardinality.HIGH,
                aggregation=SemanticAggregation.NONE,
            ),
            Semantic.AGENT_NAME: SemanticTypeInfo(
                id=Semantic.AGENT_NAME,
                description="Human-readable agent name.",
                value_shape="string",
                stability=SemanticStability.EVOLVING,
                privacy=SemanticPrivacy.PUBLIC,
                cardinality=SemanticCardinality.MEDIUM,
                aggregation=SemanticAggregation.LATEST,
            ),
            Semantic.AGENT_TASK_COMPLETION: SemanticTypeInfo(
                id=Semantic.AGENT_TASK_COMPLETION,
                parent=Semantic.RESULT_SUCCESS,
                value_shape="boolean",
            ),
            Semantic.AGENT_GOAL_ACCURACY: SemanticTypeInfo(
                id=Semantic.AGENT_GOAL_ACCURACY,
                parent=Semantic.QUALITY_SCORE,
                value_shape="number",
            ),
            Semantic.AGENT_PLAN_QUALITY: SemanticTypeInfo(
                id=Semantic.AGENT_PLAN_QUALITY,
                parent=Semantic.QUALITY_SCORE,
                value_shape="number",
            ),
            Semantic.AGENT_PLAN_ADHERENCE: SemanticTypeInfo(
                id=Semantic.AGENT_PLAN_ADHERENCE,
                parent=Semantic.QUALITY_SCORE,
                value_shape="number",
            ),
            Semantic.AGENT_STEP_EFFICIENCY: SemanticTypeInfo(
                id=Semantic.AGENT_STEP_EFFICIENCY,
                parent=Semantic.TIME_LATENCY,
                value_shape="number",
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
            Semantic.AGENT_TOOL_CALL_QUALITY: SemanticTypeInfo(
                id=Semantic.AGENT_TOOL_CALL_QUALITY,
                parent=Semantic.QUALITY_SCORE,
                value_shape="number",
            ),
            Semantic.AGENT_SERVING_VOLUME: SemanticTypeInfo(
                id=Semantic.AGENT_SERVING_VOLUME,
                value_shape="integer",
            ),
            Semantic.AGENT_OUTPUT_CORRECTNESS: SemanticTypeInfo(
                id=Semantic.AGENT_OUTPUT_CORRECTNESS,
                parent=Semantic.QUALITY_CORRECTNESS,
                value_shape="number",
            ),
            Semantic.AGENT_OUTPUT_STRUCTURE_VALIDITY: SemanticTypeInfo(
                id=Semantic.AGENT_OUTPUT_STRUCTURE_VALIDITY,
                parent=Semantic.QUALITY_CORRECTNESS,
                value_shape="boolean",
            ),
            Semantic.PROMPT_VERSION: SemanticTypeInfo(
                id=Semantic.PROMPT_VERSION,
                value_shape="string",
            ),
            Semantic.DATASET_VERSION: SemanticTypeInfo(
                id=Semantic.DATASET_VERSION,
                value_shape="string",
            ),
            Semantic.OUTPUT_SCHEMA_VERSION: SemanticTypeInfo(
                id=Semantic.OUTPUT_SCHEMA_VERSION,
                value_shape="string",
            ),
            Semantic.CAPABILITY_VERSION: SemanticTypeInfo(
                id=Semantic.CAPABILITY_VERSION,
                value_shape="string",
            ),
            Semantic.GUARDRAIL_VERSION: SemanticTypeInfo(
                id=Semantic.GUARDRAIL_VERSION,
                value_shape="string",
            ),
            Semantic.HANDOFF_VERSION: SemanticTypeInfo(
                id=Semantic.HANDOFF_VERSION,
                value_shape="string",
            ),
            Semantic.POLICY_VERSION: SemanticTypeInfo(
                id=Semantic.POLICY_VERSION,
                value_shape="string",
            ),
            Semantic.TOOLSET_VERSION: SemanticTypeInfo(
                id=Semantic.TOOLSET_VERSION,
                value_shape="string",
            ),
            Semantic.ASSET_RENDERING_VERSION: SemanticTypeInfo(
                id=Semantic.ASSET_RENDERING_VERSION,
                value_shape="string",
            ),
            Semantic.ASSET_EXTERNAL_VERSION: SemanticTypeInfo(
                id=Semantic.ASSET_EXTERNAL_VERSION,
                value_shape="string",
            ),
            Semantic.ASSET_DEPLOYMENT_LABEL: SemanticTypeInfo(
                id=Semantic.ASSET_DEPLOYMENT_LABEL,
                value_shape="string",
            ),
            Semantic.TOOL_NAME: SemanticTypeInfo(
                id=Semantic.TOOL_NAME,
                description="Canonical tool name.",
                value_shape="string",
                stability=SemanticStability.STABLE,
                privacy=SemanticPrivacy.PUBLIC,
                cardinality=SemanticCardinality.MEDIUM,
                aggregation=SemanticAggregation.LATEST,
            ),
            Semantic.TOOL_TYPE: SemanticTypeInfo(
                id=Semantic.TOOL_TYPE,
                description="Tool execution category such as function or datastore.",
                value_shape="string",
                stability=SemanticStability.EVOLVING,
                privacy=SemanticPrivacy.PUBLIC,
                cardinality=SemanticCardinality.LOW,
                aggregation=SemanticAggregation.LATEST,
            ),
            Semantic.TOOL_VERSION: SemanticTypeInfo(
                id=Semantic.TOOL_VERSION,
                description="Tracked tool version.",
                value_shape="string",
                stability=SemanticStability.STABLE,
                privacy=SemanticPrivacy.PUBLIC,
                cardinality=SemanticCardinality.MEDIUM,
                aggregation=SemanticAggregation.LATEST,
            ),
            Semantic.TOOL_DEFINITIONS: SemanticTypeInfo(
                id=Semantic.TOOL_DEFINITIONS,
                description="Definitions made available to a model or agent.",
                value_shape="array",
                stability=SemanticStability.EVOLVING,
                privacy=SemanticPrivacy.SENSITIVE,
                cardinality=SemanticCardinality.HIGH,
                aggregation=SemanticAggregation.NONE,
            ),
            Semantic.TOOL_CALL_ID: SemanticTypeInfo(
                id=Semantic.TOOL_CALL_ID,
                description="Run-local tool call correlation identifier.",
                value_shape="string",
                stability=SemanticStability.STABLE,
                privacy=SemanticPrivacy.INTERNAL,
                cardinality=SemanticCardinality.HIGH,
                aggregation=SemanticAggregation.NONE,
            ),
            Semantic.TOOL_CALL_ARGUMENTS: SemanticTypeInfo(
                id=Semantic.TOOL_CALL_ARGUMENTS,
                description="Arguments supplied to one tool call.",
                value_shape="mapping",
                stability=SemanticStability.STABLE,
                privacy=SemanticPrivacy.SENSITIVE,
                cardinality=SemanticCardinality.UNBOUNDED,
                aggregation=SemanticAggregation.NONE,
            ),
            Semantic.TOOL_CALL_RESULT: SemanticTypeInfo(
                id=Semantic.TOOL_CALL_RESULT,
                description="Result returned by one tool call.",
                value_shape="any",
                stability=SemanticStability.STABLE,
                privacy=SemanticPrivacy.SENSITIVE,
                cardinality=SemanticCardinality.UNBOUNDED,
                aggregation=SemanticAggregation.NONE,
            ),
            Semantic.TOOL_CALL_QUALITY: SemanticTypeInfo(
                id=Semantic.TOOL_CALL_QUALITY,
                parent=Semantic.QUALITY_SCORE,
                description="Quality score assigned to one tool call.",
                value_shape="number",
                stability=SemanticStability.STABLE,
                privacy=SemanticPrivacy.INTERNAL,
                cardinality=SemanticCardinality.LOW,
                aggregation=SemanticAggregation.MEAN,
            ),
            Semantic.WORKFLOW_NAME: SemanticTypeInfo(
                id=Semantic.WORKFLOW_NAME,
                description="Human-readable workflow name.",
                value_shape="string",
                stability=SemanticStability.EVOLVING,
                privacy=SemanticPrivacy.PUBLIC,
                cardinality=SemanticCardinality.MEDIUM,
                aggregation=SemanticAggregation.LATEST,
            ),
            Semantic.CONVERSATION_ID: SemanticTypeInfo(
                id=Semantic.CONVERSATION_ID,
                description="Run-local conversation or thread correlation identifier.",
                value_shape="string",
                stability=SemanticStability.EVOLVING,
                privacy=SemanticPrivacy.INTERNAL,
                cardinality=SemanticCardinality.HIGH,
                aggregation=SemanticAggregation.NONE,
            ),
            Semantic.RETRIEVAL_QUERY: SemanticTypeInfo(
                id=Semantic.RETRIEVAL_QUERY,
                description="Query supplied to a retrieval operation.",
                value_shape="string",
                stability=SemanticStability.EVOLVING,
                privacy=SemanticPrivacy.SENSITIVE,
                cardinality=SemanticCardinality.UNBOUNDED,
                aggregation=SemanticAggregation.NONE,
            ),
            Semantic.RETRIEVAL_DOCUMENTS: SemanticTypeInfo(
                id=Semantic.RETRIEVAL_DOCUMENTS,
                description="Documents returned by a retrieval operation.",
                value_shape="array",
                stability=SemanticStability.EVOLVING,
                privacy=SemanticPrivacy.SENSITIVE,
                cardinality=SemanticCardinality.UNBOUNDED,
                aggregation=SemanticAggregation.NONE,
            ),
            Semantic.RETRIEVAL_DOCUMENTS_COUNT: SemanticTypeInfo(
                id=Semantic.RETRIEVAL_DOCUMENTS_COUNT,
                description="Number of documents returned by retrieval.",
                value_shape="integer",
                stability=SemanticStability.EVOLVING,
                privacy=SemanticPrivacy.PUBLIC,
                cardinality=SemanticCardinality.LOW,
                aggregation=SemanticAggregation.SUM,
            ),
            Semantic.EVALUATION_NAME: SemanticTypeInfo(
                id=Semantic.EVALUATION_NAME,
                description="Evaluator or metric name.",
                value_shape="string",
                stability=SemanticStability.EVOLVING,
                privacy=SemanticPrivacy.PUBLIC,
                cardinality=SemanticCardinality.MEDIUM,
                aggregation=SemanticAggregation.LATEST,
            ),
            Semantic.EVALUATION_SCORE: SemanticTypeInfo(
                id=Semantic.EVALUATION_SCORE,
                parent=Semantic.QUALITY_SCORE,
                description="Numeric evaluator result.",
                value_shape="number",
                stability=SemanticStability.EVOLVING,
                privacy=SemanticPrivacy.INTERNAL,
                cardinality=SemanticCardinality.LOW,
                aggregation=SemanticAggregation.MEAN,
            ),
            Semantic.EVALUATION_LABEL: SemanticTypeInfo(
                id=Semantic.EVALUATION_LABEL,
                description="Human-readable evaluator result label.",
                value_shape="string",
                stability=SemanticStability.EVOLVING,
                privacy=SemanticPrivacy.INTERNAL,
                cardinality=SemanticCardinality.LOW,
                aggregation=SemanticAggregation.LATEST,
            ),
            Semantic.EVALUATION_EXPLANATION: SemanticTypeInfo(
                id=Semantic.EVALUATION_EXPLANATION,
                description="Evaluator explanation or feedback.",
                value_shape="string",
                stability=SemanticStability.EVOLVING,
                privacy=SemanticPrivacy.SENSITIVE,
                cardinality=SemanticCardinality.UNBOUNDED,
                aggregation=SemanticAggregation.NONE,
            ),
            Semantic.MESSAGE_INPUT: SemanticTypeInfo(
                id=Semantic.MESSAGE_INPUT,
                description="Messages supplied to a model operation.",
                value_shape="array",
                stability=SemanticStability.EVOLVING,
                privacy=SemanticPrivacy.SENSITIVE,
                cardinality=SemanticCardinality.UNBOUNDED,
                aggregation=SemanticAggregation.NONE,
            ),
            Semantic.MESSAGE_OUTPUT: SemanticTypeInfo(
                id=Semantic.MESSAGE_OUTPUT,
                description="Messages returned by a model operation.",
                value_shape="array",
                stability=SemanticStability.EVOLVING,
                privacy=SemanticPrivacy.SENSITIVE,
                cardinality=SemanticCardinality.UNBOUNDED,
                aggregation=SemanticAggregation.NONE,
            ),
            Semantic.PROMPT_SYSTEM: SemanticTypeInfo(
                id=Semantic.PROMPT_SYSTEM,
                description="System instructions supplied to a model or agent.",
                value_shape="any",
                stability=SemanticStability.EVOLVING,
                privacy=SemanticPrivacy.SENSITIVE,
                cardinality=SemanticCardinality.UNBOUNDED,
                aggregation=SemanticAggregation.NONE,
            ),
            Semantic.ARTIFACT_CONTENT: SemanticTypeInfo(
                id=Semantic.ARTIFACT_CONTENT,
                description="Content retained as benchmark evidence.",
                value_shape="any",
                stability=SemanticStability.STABLE,
                privacy=SemanticPrivacy.SENSITIVE,
                cardinality=SemanticCardinality.UNBOUNDED,
                aggregation=SemanticAggregation.NONE,
            ),
            Semantic.OPERATION_NAME: SemanticTypeInfo(
                id=Semantic.OPERATION_NAME,
                description="Normalized runtime operation name.",
                value_shape="string",
                stability=SemanticStability.STABLE,
                privacy=SemanticPrivacy.PUBLIC,
                cardinality=SemanticCardinality.LOW,
                aggregation=SemanticAggregation.LATEST,
            ),
            Semantic.OPERATION_INPUT: SemanticTypeInfo(
                id=Semantic.OPERATION_INPUT,
                description="Input captured for a runtime operation.",
                value_shape="any",
                stability=SemanticStability.STABLE,
                privacy=SemanticPrivacy.SENSITIVE,
                cardinality=SemanticCardinality.UNBOUNDED,
                aggregation=SemanticAggregation.NONE,
            ),
            Semantic.OPERATION_OUTPUT: SemanticTypeInfo(
                id=Semantic.OPERATION_OUTPUT,
                description="Output captured for a runtime operation.",
                value_shape="any",
                stability=SemanticStability.STABLE,
                privacy=SemanticPrivacy.SENSITIVE,
                cardinality=SemanticCardinality.UNBOUNDED,
                aggregation=SemanticAggregation.NONE,
            ),
            Semantic.STREAM_FIRST_CHUNK: SemanticTypeInfo(
                id=Semantic.STREAM_FIRST_CHUNK,
                parent=Semantic.EVENT_OCCURRENCE,
                description="The first response chunk became available.",
                value_shape="event",
                stability=SemanticStability.EVOLVING,
                privacy=SemanticPrivacy.PUBLIC,
                cardinality=SemanticCardinality.HIGH,
                aggregation=SemanticAggregation.NONE,
            ),
            Semantic.STREAM_COMPLETED: SemanticTypeInfo(
                id=Semantic.STREAM_COMPLETED,
                parent=Semantic.EVENT_OCCURRENCE,
                description="A response stream completed normally.",
                value_shape="event",
                stability=SemanticStability.EVOLVING,
                privacy=SemanticPrivacy.PUBLIC,
                cardinality=SemanticCardinality.HIGH,
                aggregation=SemanticAggregation.NONE,
            ),
            Semantic.STREAM_PARTIAL: SemanticTypeInfo(
                id=Semantic.STREAM_PARTIAL,
                parent=Semantic.EVENT_OCCURRENCE,
                description="A response stream ended after producing partial evidence.",
                value_shape="event",
                stability=SemanticStability.EVOLVING,
                privacy=SemanticPrivacy.PUBLIC,
                cardinality=SemanticCardinality.HIGH,
                aggregation=SemanticAggregation.NONE,
            ),
            Semantic.STREAM_FAILED: SemanticTypeInfo(
                id=Semantic.STREAM_FAILED,
                parent=Semantic.EVENT_OCCURRENCE,
                description="A response stream failed before normal completion.",
                value_shape="event",
                stability=SemanticStability.EVOLVING,
                privacy=SemanticPrivacy.PUBLIC,
                cardinality=SemanticCardinality.HIGH,
                aggregation=SemanticAggregation.NONE,
            ),
            Semantic.OPERATION_RETRY: SemanticTypeInfo(
                id=Semantic.OPERATION_RETRY,
                parent=Semantic.EVENT_OCCURRENCE,
                description="An operation requested another attempt.",
                value_shape="event",
                stability=SemanticStability.STABLE,
                privacy=SemanticPrivacy.PUBLIC,
                cardinality=SemanticCardinality.HIGH,
                aggregation=SemanticAggregation.NONE,
            ),
            Semantic.OPERATION_REPAIR: SemanticTypeInfo(
                id=Semantic.OPERATION_REPAIR,
                parent=Semantic.OPERATION_RETRY,
                description="An operation requested a corrective attempt.",
                value_shape="event",
                stability=SemanticStability.EVOLVING,
                privacy=SemanticPrivacy.PUBLIC,
                cardinality=SemanticCardinality.HIGH,
                aggregation=SemanticAggregation.NONE,
            ),
            Semantic.OPERATION_DEFERRED: SemanticTypeInfo(
                id=Semantic.OPERATION_DEFERRED,
                parent=Semantic.EVENT_OCCURRENCE,
                description="An operation paused for external completion or approval.",
                value_shape="event",
                stability=SemanticStability.EVOLVING,
                privacy=SemanticPrivacy.PUBLIC,
                cardinality=SemanticCardinality.HIGH,
                aggregation=SemanticAggregation.NONE,
            ),
            Semantic.OPERATION_DEFERRED_RESOLVED: SemanticTypeInfo(
                id=Semantic.OPERATION_DEFERRED_RESOLVED,
                parent=Semantic.OPERATION_DEFERRED,
                description="A previously deferred operation received a result.",
                value_shape="event",
                stability=SemanticStability.EVOLVING,
                privacy=SemanticPrivacy.PUBLIC,
                cardinality=SemanticCardinality.HIGH,
                aggregation=SemanticAggregation.NONE,
            ),
            Semantic.VALIDATION_FAILURE: SemanticTypeInfo(
                id=Semantic.VALIDATION_FAILURE,
                parent=Semantic.EVENT_OCCURRENCE,
                description="Input, tool, or output validation rejected a value.",
                value_shape="event",
                stability=SemanticStability.STABLE,
                privacy=SemanticPrivacy.SENSITIVE,
                cardinality=SemanticCardinality.HIGH,
                aggregation=SemanticAggregation.NONE,
            ),
            Semantic.APPROVAL_REQUESTED: SemanticTypeInfo(
                id=Semantic.APPROVAL_REQUESTED,
                parent=Semantic.EVENT_OCCURRENCE,
                description="An operation requested external approval.",
                value_shape="event",
                stability=SemanticStability.EVOLVING,
                privacy=SemanticPrivacy.INTERNAL,
                cardinality=SemanticCardinality.HIGH,
                aggregation=SemanticAggregation.NONE,
            ),
            Semantic.TOOL_CALL_REQUESTED: SemanticTypeInfo(
                id=Semantic.TOOL_CALL_REQUESTED,
                parent=Semantic.EVENT_OCCURRENCE,
                description="A model or agent requested a tool call.",
                value_shape="event",
                stability=SemanticStability.STABLE,
                privacy=SemanticPrivacy.INTERNAL,
                cardinality=SemanticCardinality.HIGH,
                aggregation=SemanticAggregation.NONE,
            ),
            Semantic.FACTOR_VALUE: SemanticTypeInfo(
                id=Semantic.FACTOR_VALUE,
                description="Unclassified factor value that may influence an outcome.",
                value_shape="any",
                stability=SemanticStability.STABLE,
                privacy=SemanticPrivacy.INTERNAL,
                cardinality=SemanticCardinality.HIGH,
                aggregation=SemanticAggregation.NONE,
            ),
            Semantic.EVENT_OCCURRENCE: SemanticTypeInfo(
                id=Semantic.EVENT_OCCURRENCE,
                description="Unclassified runtime event occurrence.",
                value_shape="any",
                stability=SemanticStability.STABLE,
                privacy=SemanticPrivacy.INTERNAL,
                cardinality=SemanticCardinality.HIGH,
                aggregation=SemanticAggregation.NONE,
            ),
            Semantic.DIAGNOSTIC_EVENT: SemanticTypeInfo(
                id=Semantic.DIAGNOSTIC_EVENT,
                description="Diagnostic runtime event retained for analysis.",
                value_shape="any",
                stability=SemanticStability.STABLE,
                privacy=SemanticPrivacy.INTERNAL,
                cardinality=SemanticCardinality.HIGH,
                aggregation=SemanticAggregation.NONE,
            ),
            Semantic.ERROR_EXCEPTION: SemanticTypeInfo(
                id=Semantic.ERROR_EXCEPTION,
                description="Structured runtime exception evidence.",
                value_shape="mapping",
                stability=SemanticStability.STABLE,
                privacy=SemanticPrivacy.SENSITIVE,
                cardinality=SemanticCardinality.HIGH,
                aggregation=SemanticAggregation.NONE,
            ),
            Semantic.OPERATION_COUNT: SemanticTypeInfo(
                id=Semantic.OPERATION_COUNT,
                description="Number of materialized operations in a selected grouping.",
                unit="operations",
                value_shape="integer",
                stability=SemanticStability.STABLE,
                privacy=SemanticPrivacy.PUBLIC,
                cardinality=SemanticCardinality.LOW,
                aggregation=SemanticAggregation.SUM,
            ),
            Semantic.OPERATION_DEPTH_MAX: SemanticTypeInfo(
                id=Semantic.OPERATION_DEPTH_MAX,
                description="Maximum parent-child operation depth in a trace.",
                unit="levels",
                value_shape="integer",
                stability=SemanticStability.STABLE,
                privacy=SemanticPrivacy.PUBLIC,
                cardinality=SemanticCardinality.LOW,
                aggregation=SemanticAggregation.MEAN,
            ),
            Semantic.OPERATION_FAN_OUT_MAX: SemanticTypeInfo(
                id=Semantic.OPERATION_FAN_OUT_MAX,
                description="Maximum direct child and explicit fan-out count.",
                unit="operations",
                value_shape="integer",
                stability=SemanticStability.STABLE,
                privacy=SemanticPrivacy.PUBLIC,
                cardinality=SemanticCardinality.LOW,
                aggregation=SemanticAggregation.MEAN,
            ),
            Semantic.OPERATION_INCOMPLETE_COUNT: SemanticTypeInfo(
                id=Semantic.OPERATION_INCOMPLETE_COUNT,
                parent=Semantic.OPERATION_COUNT,
                description="Partial or abandoned operation count.",
                unit="operations",
                value_shape="integer",
                stability=SemanticStability.STABLE,
                privacy=SemanticPrivacy.PUBLIC,
                cardinality=SemanticCardinality.LOW,
                aggregation=SemanticAggregation.SUM,
            ),
            Semantic.OPERATION_PARALLELISM: SemanticTypeInfo(
                id=Semantic.OPERATION_PARALLELISM,
                description="Completed leaf work divided by observed trace makespan.",
                unit="ratio",
                value_shape="number",
                stability=SemanticStability.STABLE,
                privacy=SemanticPrivacy.PUBLIC,
                cardinality=SemanticCardinality.LOW,
                aggregation=SemanticAggregation.MEAN,
            ),
            Semantic.OPERATION_RETRY_COUNT: SemanticTypeInfo(
                id=Semantic.OPERATION_RETRY_COUNT,
                parent=Semantic.OPERATION_COUNT,
                description="Retry relationships observed in a trace.",
                unit="operations",
                value_shape="integer",
                stability=SemanticStability.STABLE,
                privacy=SemanticPrivacy.PUBLIC,
                cardinality=SemanticCardinality.LOW,
                aggregation=SemanticAggregation.SUM,
            ),
            Semantic.OPERATION_RETRY_RECOVERED_COUNT: SemanticTypeInfo(
                id=Semantic.OPERATION_RETRY_RECOVERED_COUNT,
                parent=Semantic.OPERATION_RETRY_COUNT,
                description="Retries that succeeded after a failed original attempt.",
                unit="operations",
                value_shape="integer",
                stability=SemanticStability.STABLE,
                privacy=SemanticPrivacy.PUBLIC,
                cardinality=SemanticCardinality.LOW,
                aggregation=SemanticAggregation.SUM,
            ),
            Semantic.OPERATION_FIRST_ATTEMPT_SUCCESS: SemanticTypeInfo(
                id=Semantic.OPERATION_FIRST_ATTEMPT_SUCCESS,
                parent=Semantic.RESULT_SUCCESS,
                description="Success ratio of original attempts in retry groups.",
                unit="ratio",
                value_shape="number",
                stability=SemanticStability.STABLE,
                privacy=SemanticPrivacy.PUBLIC,
                cardinality=SemanticCardinality.LOW,
                aggregation=SemanticAggregation.MEAN,
            ),
            Semantic.VALIDATION_COUNT: SemanticTypeInfo(
                id=Semantic.VALIDATION_COUNT,
                parent=Semantic.OPERATION_COUNT,
                unit="operations",
                value_shape="integer",
                aggregation=SemanticAggregation.SUM,
            ),
            Semantic.VALIDATION_FAILURE_COUNT: SemanticTypeInfo(
                id=Semantic.VALIDATION_FAILURE_COUNT,
                parent=Semantic.VALIDATION_COUNT,
                unit="operations",
                value_shape="integer",
                aggregation=SemanticAggregation.SUM,
            ),
            Semantic.VALIDATION_FAILURE_RATE: SemanticTypeInfo(
                id=Semantic.VALIDATION_FAILURE_RATE,
                parent=Semantic.QUALITY_SCORE,
                unit="ratio",
                value_shape="number",
                aggregation=SemanticAggregation.MEAN,
            ),
            Semantic.APPROVAL_COUNT: SemanticTypeInfo(
                id=Semantic.APPROVAL_COUNT,
                parent=Semantic.OPERATION_COUNT,
                unit="operations",
                value_shape="integer",
                aggregation=SemanticAggregation.SUM,
            ),
            Semantic.APPROVAL_WAIT: SemanticTypeInfo(
                id=Semantic.APPROVAL_WAIT,
                parent=Semantic.TIME_LATENCY,
                unit="s",
                value_shape="number",
                aggregation=SemanticAggregation.SUM,
            ),
            Semantic.TOOL_CALL_COUNT: SemanticTypeInfo(
                id=Semantic.TOOL_CALL_COUNT,
                parent=Semantic.OPERATION_COUNT,
                unit="operations",
                value_shape="integer",
                aggregation=SemanticAggregation.SUM,
            ),
            Semantic.TOOL_CALL_SUCCESS_COUNT: SemanticTypeInfo(
                id=Semantic.TOOL_CALL_SUCCESS_COUNT,
                parent=Semantic.TOOL_CALL_COUNT,
                unit="operations",
                value_shape="integer",
                aggregation=SemanticAggregation.SUM,
            ),
            Semantic.TOOL_CALL_FAILURE_COUNT: SemanticTypeInfo(
                id=Semantic.TOOL_CALL_FAILURE_COUNT,
                parent=Semantic.TOOL_CALL_COUNT,
                unit="operations",
                value_shape="integer",
                aggregation=SemanticAggregation.SUM,
            ),
            Semantic.TOOL_CALL_ARGUMENTS_PRESENT_COUNT: SemanticTypeInfo(
                id=Semantic.TOOL_CALL_ARGUMENTS_PRESENT_COUNT,
                parent=Semantic.TOOL_CALL_COUNT,
                unit="operations",
                value_shape="integer",
                aggregation=SemanticAggregation.SUM,
            ),
            Semantic.ARTIFACT_REFERENCE_COUNT: SemanticTypeInfo(
                id=Semantic.ARTIFACT_REFERENCE_COUNT,
                unit="references",
                value_shape="integer",
                aggregation=SemanticAggregation.SUM,
            ),
            Semantic.ASSET_REFERENCE_COUNT: SemanticTypeInfo(
                id=Semantic.ASSET_REFERENCE_COUNT,
                unit="references",
                value_shape="integer",
                aggregation=SemanticAggregation.SUM,
            ),
            Semantic.MESSAGE_INPUT_COUNT: SemanticTypeInfo(
                id=Semantic.MESSAGE_INPUT_COUNT,
                unit="messages",
                value_shape="integer",
                aggregation=SemanticAggregation.SUM,
            ),
            Semantic.MESSAGE_OUTPUT_COUNT: SemanticTypeInfo(
                id=Semantic.MESSAGE_OUTPUT_COUNT,
                unit="messages",
                value_shape="integer",
                aggregation=SemanticAggregation.SUM,
            ),
            Semantic.MESSAGE_GROWTH: SemanticTypeInfo(
                id=Semantic.MESSAGE_GROWTH,
                unit="messages",
                value_shape="integer",
                aggregation=SemanticAggregation.MEAN,
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
            "llm.requests": Semantic.LLM_REQUEST_COUNT,
            "quality.answer": Semantic.QUALITY_SCORE,
            "agent.tool_call.correctness": Semantic.AGENT_TOOL_CALL_QUALITY,
            "agent.task_completion": Semantic.AGENT_TASK_COMPLETION,
            "agent.goal_accuracy": Semantic.AGENT_GOAL_ACCURACY,
            "agent.tool.correctness": Semantic.AGENT_TOOL_SELECTION_CORRECTNESS,
            "agent.tool.args.correctness": Semantic.AGENT_TOOL_ARGUMENT_CORRECTNESS,
            "agent.output.valid": Semantic.AGENT_OUTPUT_STRUCTURE_VALIDITY,
            Semantic.LLM_PROVIDER: Semantic.LLM_PROVIDER_NAME,
            Semantic.AGENT_TOOL_NAME: Semantic.TOOL_NAME,
            Semantic.AGENT_TOOL_VERSION: Semantic.TOOL_VERSION,
            Semantic.AGENT_TOOL_CALL_QUALITY: Semantic.TOOL_CALL_QUALITY,
        }
        return cls(types=types, aliases=aliases)

    def info_for(self, semantic_type: str | None) -> SemanticTypeInfo | None:
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
    if info.description is not None:
        view["description"] = info.description
    if info.unit is not None:
        view["unit"] = info.unit
    if info.value_shape is not None:
        view["shape"] = info.value_shape
    if info.aliases:
        view["aliases"] = list(info.aliases)
    if info.deprecated:
        view["deprecated"] = True
    if info.stability is not None:
        view["stability"] = info.stability.value
    if info.privacy is not None:
        view["privacy"] = info.privacy.value
    if info.cardinality is not None:
        view["cardinality"] = info.cardinality.value
    if info.aggregation is not None:
        view["aggregation"] = info.aggregation.value
    if info.tags:
        view["tags"] = dict(info.tags)
    return view


__all__ = (
    "AgentSemanticType",
    "ContentSemanticType",
    "ConversationSemanticType",
    "CostSemanticType",
    "DEFAULT_SEMANTIC_REGISTRY",
    "DatasetSemanticType",
    "EvaluationSemanticType",
    "ErrorSemanticType",
    "HTTPSemanticType",
    "KnownSemanticType",
    "LLMSemanticType",
    "NetworkSemanticType",
    "OperationSemanticType",
    "PromptSemanticType",
    "QualitySemanticType",
    "ResultSemanticType",
    "RetrievalSemanticType",
    "RuntimeSemanticType",
    "Semantic",
    "SemanticAggregation",
    "SemanticCardinality",
    "SemanticPrivacy",
    "SemanticRegistry",
    "SemanticStability",
    "SemanticType",
    "SemanticTypeInfo",
    "semantic_registry_payload_from_yaml_view",
    "semantic_registry_to_yaml_view",
    "TimeSemanticType",
    "ToolSemanticType",
    "WorkflowSemanticType",
)
