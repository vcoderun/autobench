from __future__ import annotations

import asyncio
import builtins
import subprocess
import sys
from collections.abc import AsyncIterable, AsyncIterator, Iterator
from contextlib import contextmanager
from inspect import getattr_static
from typing import Any

import pytest
from pydantic import BaseModel
from pydantic_ai import (
    Agent,
    BinaryContent,
    DeferredToolRequests,
    DeferredToolResults,
    Tool,
)
from pydantic_ai import (
    RunContext as AgentRunContext,
)
from pydantic_ai.agent.wrapper import WrapperAgent
from pydantic_ai.capabilities import AbstractCapability, Instrumentation
from pydantic_ai.exceptions import ApprovalRequired, CallDeferred, ModelRetry
from pydantic_ai.messages import (
    AgentStreamEvent,
    DeferredToolResultsEvent,
    FunctionToolCallEvent,
    FunctionToolResultEvent,
    InstructionPart,
    ModelMessage,
    ModelRequest,
    ModelResponse,
    PartStartEvent,
    SystemPromptPart,
    TextPart,
    ToolCallPart,
    ToolReturnPart,
)
from pydantic_ai.models import ModelRequestContext, ModelRequestParameters
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.models.test import TestModel
from pydantic_ai.native_tools import AbstractNativeTool
from pydantic_ai.output import OutputContext, OutputObjectDefinition
from pydantic_ai.run import AgentRunResult
from pydantic_ai.tools import ToolDefinition
from pydantic_ai.toolsets.external import ExternalToolset
from pydantic_ai.usage import RequestUsage, RunUsage

from autobench import (
    DEFAULT_SEMANTIC_REGISTRY,
    Case,
    CompatibilityStatus,
    CompositeExtractor,
    ExtractionContext,
    InstrumentationManager,
    RunContext,
    Semantic,
    Variant,
    suppress_instrumentation,
)
from autobench.instrumentation.config import AssetDiscoverySettings
from autobench.instrumentation.manager import InstrumentationRuntime
from autobench.instrumentation.patching import InstrumentationConflictError
from autobench.instrumentation.pydantic_ai import PydanticAI
from autobench.instrumentation.pydantic_ai.assets import AssetDiscovery
from autobench.instrumentation.pydantic_ai.capability import AutobenchCapability
from autobench.protocol import CapturePolicy, EndReason, SpanStatus
from autobench.protocol.traces import SpanRecord
from autobench.runtime.instrumentation import reset_active_run_context, set_active_run_context
from autobench.tracking import TrackingRegistry


def _run_context(*, capture_policy: CapturePolicy | None = None) -> RunContext:
    return RunContext(
        benchmark_id="pydantic-ai",
        case=Case(id="case", input="input"),
        variant=Variant(id="variant"),
        capture_policy=capture_policy,
    )


@contextmanager
def _active(context: RunContext) -> Iterator[None]:
    token = set_active_run_context(context)
    try:
        yield
    finally:
        reset_active_run_context(token)


def _spans(context: RunContext, operation: str) -> list[SpanRecord]:
    return [span for span in context.trace.spans if span.operation == operation]


def _extraction_context(context: RunContext) -> ExtractionContext:
    return ExtractionContext(
        run_id=context.run_id,
        benchmark_id=context.benchmark_id,
        experiment_id=context.experiment_id,
        case_id=context.case.id,
        variant_id=context.variant.id,
    )


def _capability() -> AutobenchCapability:
    instrumentor = PydanticAI()
    return AutobenchCapability(
        InstrumentationRuntime(),
        instrumentor.info,
        target_version="2.22.0",
    )


def _agent_context(
    *,
    agent: Agent[None, Any] | None = None,
    partial_output: bool = False,
    run_step: int = 1,
) -> AgentRunContext[None]:
    return AgentRunContext(
        deps=None,
        model=TestModel(),
        usage=RunUsage(),
        agent=agent,
        prompt="prompt",
        run_id="agent-run",
        conversation_id="conversation",
        partial_output=partial_output,
        run_step=run_step,
    )


def test_instrumentor_declares_and_checks_supported_public_capabilities() -> None:
    instrumentor = PydanticAI()

    compatibility = instrumentor.check()

    assert compatibility.status is CompatibilityStatus.COMPATIBLE
    assert instrumentor.info.id == "autobench.pydantic_ai"
    assert instrumentor.info.supported_versions == ">=2.22,<2.23"
    assert instrumentor.info.capabilities.model_dump(by_alias=True) == {
        "sync": True,
        "async": True,
        "streaming": True,
        "native_hooks": True,
        "asset_discovery": True,
        "asset_kinds": (
            "agent",
            "capability",
            "output_schema",
            "policy",
            "prompt",
            "tool",
            "toolset",
        ),
    }


def test_instrumentor_reports_missing_broken_and_unsupported_pydantic_ai(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import autobench.instrumentation.pydantic_ai.instrumentor as module

    monkeypatch.setattr(module, "find_spec", lambda name: None)
    assert PydanticAI().check().status is CompatibilityStatus.UNAVAILABLE

    monkeypatch.setattr(module, "find_spec", lambda name: True)
    original_import = builtins.__import__

    def broken_import(
        name: str,
        globals: dict[str, Any] | None = None,
        locals: dict[str, Any] | None = None,
        fromlist: tuple[str, ...] = (),
        level: int = 0,
    ) -> Any:
        if name == "pydantic_ai":
            raise ImportError("broken installation")
        return original_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", broken_import)
    broken = PydanticAI().check()
    assert broken.status is CompatibilityStatus.UNAVAILABLE
    assert "broken installation" in broken.diagnostics[0]
    monkeypatch.setattr(builtins, "__import__", original_import)

    def incompatible_iter(self: Agent[Any, Any]) -> None:
        del self

    monkeypatch.setattr(Agent, "iter", incompatible_iter)
    unsupported = PydanticAI().check()
    assert unsupported.status is CompatibilityStatus.UNSUPPORTED
    assert "Agent" in unsupported.degraded_features[0]


def test_sync_agent_collects_model_tool_usage_stream_assets_and_extracted_evidence() -> None:
    registry = TrackingRegistry()

    @registry.type
    class Answer(BaseModel):
        value: int

    @registry.tool
    def add(a: int, b: int) -> int:
        """Add two integers."""

        return a + b

    prompt = registry.prompt(name="system", text="Use the available tools.")
    agent = Agent[None, Answer](
        TestModel(),
        name="calculator",
        output_type=Answer,
        tools=[Tool[None](add)],
        toolsets=[ExternalToolset[None]([])],
        instructions=str(prompt),
        deps_type=type(None),
    )
    context = _run_context()

    with InstrumentationManager() as manager:
        manager.install(PydanticAI(assets=[prompt], registry=registry))
        with _active(context):
            result = agent.run_sync("Calculate a value", model_settings={"temperature": 0.2})

    assert isinstance(result.output, Answer)
    trace = context.finalize(output=result.output)
    agent_span = _spans(context, "pydantic_ai.agent.run")[0]
    model_spans = _spans(context, "pydantic_ai.model.request")
    tool_span = _spans(context, "pydantic_ai.tool.execute")[0]
    validation_spans = _spans(context, "pydantic_ai.tool.validate")

    assert agent_span.status is SpanStatus.OK
    assert agent_span.usage["requests"] == len(model_spans)
    assert len(model_spans) == 2
    assert model_spans[0].attributes[Semantic.LLM_MODEL_REQUESTED] == "test:test"
    assert model_spans[0].attributes[Semantic.LLM_MODEL_RESPONSE] == "test"
    assert model_spans[0].attributes[Semantic.LLM_PROVIDER_NAME] == "test"
    assert model_spans[0].attributes["temperature"] == 0.2
    assert model_spans[0].usage["requests"] == 1
    first_input_tokens = model_spans[0].usage["input_tokens"]
    assert isinstance(first_input_tokens, int)
    assert first_input_tokens > 0
    assert all(
        measurement.semantic_type != Semantic.TIME_FIRST_CHUNK
        for span in model_spans
        for measurement in span.measurements
    )
    assert tool_span.attributes[Semantic.TOOL_NAME] == "add"
    assert tool_span.status is SpanStatus.OK
    assert validation_spans[0].status is SpanStatus.OK
    assert {
        "prompt.system",
        "tool.add",
        "type.Answer",
    } <= {version.asset_id for version in context.asset_versions}
    agent_reference_ids = {reference.reference.id for reference in agent_span.references}
    assert {"prompt.system", "tool.add", "type.Answer"} <= agent_reference_ids

    extracted = CompositeExtractor().extract(
        trace,
        registry=DEFAULT_SEMANTIC_REGISTRY,
        context=_extraction_context(context),
    )
    summaries = {
        observation.semantic_type: observation.value
        for observation in extracted.observations
        if observation.tags.get("abp.summary") is True
    }
    assert summaries[Semantic.LLM_REQUEST_COUNT] == 2
    input_token_values = [span.usage["input_tokens"] for span in model_spans]
    assert all(isinstance(value, int) for value in input_token_values)
    assert summaries[Semantic.LLM_TOKENS_INPUT] == sum(
        value for value in input_token_values if isinstance(value, int)
    )
    assert summaries[Semantic.TOOL_CALL_COUNT] == 1
    assert summaries[Semantic.MESSAGE_INPUT_COUNT] == 4
    assert summaries[Semantic.MESSAGE_OUTPUT_COUNT] == 2
    assert summaries[Semantic.ASSET_REFERENCE_COUNT] == len(context.reference_store.assets)


def test_untracked_agent_discovers_scoped_capabilities_tools_prompts_and_output_schema() -> None:
    class Output(BaseModel):
        answer: str

    class RetrievalCapability(AbstractCapability[None]):
        id = "retrieval"
        instruction_calls: int

        def __init__(self) -> None:
            self.instruction_calls = 0

        def get_instructions(self) -> str:
            self.instruction_calls += 1
            return "Use the shared capability instruction."

    class SafetyCapability(AbstractCapability[None]):
        id = "safety"
        instruction_calls: int

        def __init__(self) -> None:
            self.instruction_calls = 0

        def get_instructions(self) -> str:
            self.instruction_calls += 1
            return "Use the shared capability instruction."

    class RuntimeObserver(AbstractCapability[None]):
        id = "runtime-observer"

    def lookup(query: str) -> str:
        return query

    baseline_retrieval = RetrievalCapability()
    baseline_safety = SafetyCapability()
    baseline_capabilities: tuple[AbstractCapability[None], ...] = (
        baseline_retrieval,
        baseline_safety,
    )
    baseline = Agent[None, Output](
        TestModel(),
        name="baseline-agent",
        output_type=Output,
        deps_type=type(None),
        instructions="Answer directly.",
        tools=[Tool[None](lookup)],
        capabilities=baseline_capabilities,
    )
    baseline.run_sync("answer", capabilities=[RuntimeObserver()])
    baseline_calls = [
        baseline_retrieval.instruction_calls,
        baseline_safety.instruction_calls,
    ]

    retrieval = RetrievalCapability()
    safety = SafetyCapability()
    capabilities: tuple[AbstractCapability[None], ...] = (retrieval, safety)
    agent = Agent[None, Output](
        TestModel(),
        name="discovery-agent",
        output_type=Output,
        deps_type=type(None),
        instructions="Answer directly.",
        tools=[Tool[None](lookup)],
        capabilities=capabilities,
    )
    registry = TrackingRegistry()
    context = _run_context()

    with InstrumentationManager() as manager:
        manager.install(PydanticAI(registry=registry))
        with _active(context):
            result = agent.run_sync(
                "answer",
                instructions="Use the runtime instruction too.",
            )

    assert isinstance(result.output, Output)
    assert [retrieval.instruction_calls, safety.instruction_calls] == baseline_calls
    asset_ids = {version.asset_id for version in context.asset_versions}
    assert {
        "pydantic_ai:retrieval:capability:self",
        "pydantic_ai:retrieval:prompt:instructions",
        "pydantic_ai:safety:capability:self",
        "pydantic_ai:safety:prompt:instructions",
        "pydantic_ai:agent:discovery-agent:agent:self",
        "pydantic_ai:agent:discovery-agent:output_schema:output",
        "pydantic_ai:agent:discovery-agent:prompt:runtime_instructions",
        "pydantic_ai:agent:discovery-agent:tool:lookup",
    } <= asset_ids
    assert registry.resolve_locator("retrieval:prompt:instructions").id == (
        "pydantic_ai:retrieval:prompt:instructions"
    )
    effective_uses = [use for use in context.asset_uses if use.representation.value == "effective"]
    assert any(use.definition_asset_id is not None for use in effective_uses)
    assert all(use.span_id is not None for use in context.asset_uses)


def test_pydantic_ai_discovers_resolved_prompt_native_output_and_runtime_overrides() -> None:
    class Output(BaseModel):
        answer: str

    agent = Agent[None, Output](
        TestModel(),
        name="rich-agent",
        output_type=Output,
        deps_type=type(None),
        description="Extract a structured answer.",
    )
    agent_context = _agent_context(agent=agent)
    registry = TrackingRegistry()
    runtime = InstrumentationRuntime(registry=registry)
    instrumentor = PydanticAI(registry=registry)
    discovery = AssetDiscovery(
        runtime,
        instrumentor.info,
        target_version="2.22.0",
        registry=registry,
        settings=AssetDiscoverySettings(),
    )
    benchmark_context = _run_context()
    request_context = ModelRequestContext(
        model=TestModel(),
        messages=[
            ModelRequest(
                parts=[
                    SystemPromptPart("Static system prompt."),
                    SystemPromptPart("Resolved dynamic prompt.", dynamic_ref="dynamic"),
                ]
            )
        ],
        model_settings=None,
        model_request_parameters=ModelRequestParameters(
            native_tools=[AbstractNativeTool(kind="web_search")],
            output_object=OutputObjectDefinition(
                {"type": "object", "properties": {"answer": {"type": "string"}}},
                name="Output",
            ),
            prompted_output_template="Return JSON matching {schema}",
            instruction_parts=[
                InstructionPart("Static instruction."),
                InstructionPart("Resolved instruction.", dynamic=True),
            ],
        ),
    )
    token = set_active_run_context(benchmark_context)
    try:
        discovery.agent(
            agent_context,
            span_id="agent",
            overrides={
                "output_type": str,
                "toolsets": [ExternalToolset[None]([]), "not-a-toolset"],
                "spec": {"retries": 2},
            },
        )
        discovery.request(agent_context, request_context, span_id="model")
        discovery.request(
            agent_context,
            ModelRequestContext(
                model=TestModel(),
                messages=[],
                model_settings=None,
                model_request_parameters=ModelRequestParameters(
                    instruction_parts=[InstructionPart("Only dynamic instructions.", dynamic=True)]
                ),
            ),
            span_id="dynamic-model",
        )
        discovery.output(
            agent_context,
            OutputContext(
                mode="text",
                output_type=None,
                object_def=None,
                has_function=False,
            ),
            span_id="validation",
        )
        unnamed = Agent[None, str](TestModel(), deps_type=type(None))
        discovery.agent(
            _agent_context(agent=unnamed),
            span_id="unnamed",
            overrides={},
        )
        disabled = AssetDiscovery(
            runtime,
            instrumentor.info,
            target_version="2.22.0",
            registry=registry,
            settings=AssetDiscoverySettings(discover=False),
        )
        disabled.agent(
            agent_context,
            span_id="disabled",
            overrides={},
        )
    finally:
        reset_active_run_context(token)

    asset_ids = {asset.id for asset in registry.definitions}
    assert {
        "pydantic_ai:agent:rich-agent:prompt:description",
        "pydantic_ai:agent:rich-agent:output_schema:runtime_output",
        "pydantic_ai:agent:rich-agent:policy:runtime_spec",
        "pydantic_ai:agent:rich-agent:prompt:prompted_output",
        "pydantic_ai:agent:rich-agent:prompt:system_prompt",
        "pydantic_ai:agent:rich-agent:prompt:system_prompt:effective",
        "pydantic_ai:agent:rich-agent:tool:native:AbstractNativeTool",
    } <= asset_ids
    effective_system = next(
        use
        for use in benchmark_context.asset_uses
        if use.source_locator == "pydantic_ai:agent:rich-agent:prompt:system_prompt:effective"
    )
    assert effective_system.definition_asset_id is not None


def test_request_only_model_preserves_response_metadata_and_reasoning_usage() -> None:
    def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        del messages, info
        return ModelResponse(
            parts=[TextPart("reasoned")],
            usage=RequestUsage(
                input_tokens=8,
                output_tokens=3,
                details={"reasoning_tokens": 2},
            ),
            model_name="served-model",
            provider_name="custom-provider",
            provider_response_id="response-1",
            finish_reason="stop",
        )

    agent = Agent[None, str](
        FunctionModel(respond, model_name="requested-model"),
        deps_type=type(None),
    )
    context = _run_context()
    with InstrumentationManager() as manager:
        manager.install(PydanticAI())
        with _active(context):
            result = agent.run_sync("reason", model_settings={"top_p": 0.8})

    assert result.output == "reasoned"
    model_span = _spans(context, "pydantic_ai.model.request")[0]
    assert model_span.attributes["response_id"] == "response-1"
    assert model_span.attributes["finish_reason"] == "stop"
    assert model_span.usage["reasoning_tokens"] == 2
    assert "temperature" not in model_span.attributes


def test_request_only_model_errors_are_re_raised_unchanged_with_partial_evidence() -> None:
    expected = RuntimeError("model unavailable")

    def fail(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        del messages, info
        raise expected

    agent = Agent[None, str](
        FunctionModel(fail, model_name="failing-model"),
        deps_type=type(None),
    )
    context = _run_context()
    with InstrumentationManager() as manager:
        manager.install(PydanticAI())
        with _active(context), pytest.raises(RuntimeError) as raised:
            agent.run_sync("fail")

    assert raised.value is expected
    model_span = _spans(context, "pydantic_ai.model.request")[0]
    assert model_span.status is SpanStatus.ERROR
    assert model_span.end_reason is EndReason.FAILED


@pytest.mark.asyncio
async def test_wrapper_agent_composes_with_user_event_handler_and_existing_instrumentation() -> (
    None
):
    observed: list[str] = []

    async def event_handler(
        ctx: AgentRunContext[None],
        stream: AsyncIterable[AgentStreamEvent],
    ) -> None:
        del ctx
        async for event in stream:
            observed.append(event.event_kind)

    wrapped = WrapperAgent(
        Agent[None, str](
            TestModel(custom_output_text="composed"),
            capabilities=[Instrumentation()],
            deps_type=type(None),
        )
    )
    context = _run_context()

    with InstrumentationManager() as manager:
        manager.install(PydanticAI())
        with _active(context):
            result = await wrapped.run("hello", event_stream_handler=event_handler)

    assert result.output == "composed"
    assert observed
    assert len(_spans(context, "pydantic_ai.agent.run")) == 1
    assert len(_spans(context, "pydantic_ai.model.request")) == 1


def test_injection_is_inactive_without_context_and_respects_suppression() -> None:
    agent = Agent[None, str](
        TestModel(custom_output_text="unchanged"),
        deps_type=type(None),
    )
    context = _run_context()

    with InstrumentationManager() as manager:
        manager.install(PydanticAI())
        assert agent.run_sync("outside").output == "unchanged"
        with _active(context), suppress_instrumentation("autobench.pydantic_ai"):
            assert agent.run_sync("suppressed").output == "unchanged"

    context.finalize()
    assert not _spans(context, "pydantic_ai.agent.run")


def test_tool_retry_and_error_preserve_native_results_and_exceptions() -> None:
    attempts = 0

    def retry_once() -> str:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise ModelRetry("try again")
        return "done"

    retry_agent = Agent[None, str](
        TestModel(),
        tools=[Tool[None](retry_once)],
        deps_type=type(None),
    )
    retry_context = _run_context()
    with InstrumentationManager() as manager:
        manager.install(PydanticAI())
        with _active(retry_context):
            result = retry_agent.run_sync("retry")

    assert result.output
    tool_attempts = _spans(retry_context, "pydantic_ai.tool.execute")
    assert len(tool_attempts) == 2
    assert tool_attempts[0].status is SpanStatus.ERROR
    assert any(event.semantic_type == "operation.retry" for event in tool_attempts[0].events)
    assert tool_attempts[1].status is SpanStatus.OK

    def fail() -> str:
        raise RuntimeError("tool exploded")

    failing_agent = Agent[None, str](
        TestModel(),
        tools=[Tool[None](fail, max_retries=0)],
        deps_type=type(None),
    )
    failing_context = _run_context()
    with InstrumentationManager() as manager:
        manager.install(PydanticAI())
        with _active(failing_context), pytest.raises(Exception, match="tool exploded"):
            failing_agent.run_sync("fail")

    failed_tool = _spans(failing_context, "pydantic_ai.tool.execute")[0]
    assert failed_tool.status is SpanStatus.ERROR
    assert failed_tool.end_reason is EndReason.FAILED


def test_approval_and_deferred_results_are_control_flow_not_failures() -> None:
    def protected(value: int) -> int:
        return value

    agent = Agent[None, str | DeferredToolRequests](
        TestModel(),
        tools=[Tool[None](protected, requires_approval=True)],
        output_type=[str, DeferredToolRequests],
        deps_type=type(None),
    )
    context = _run_context()
    with InstrumentationManager() as manager:
        manager.install(PydanticAI())
        with _active(context):
            result = agent.run_sync("call protected")

    assert isinstance(result.output, DeferredToolRequests)
    assert result.output.approvals
    context.finalize(output=result.output)
    approval_events = [
        event
        for span in context.trace.spans
        for event in span.events
        if event.semantic_type == "approval.requested"
    ]
    assert len(approval_events) == 1
    assert all(span.status is SpanStatus.OK for span in context.trace.spans)


@pytest.mark.asyncio
async def test_multimodal_input_and_streaming_keep_content_policy_and_partial_evidence() -> None:
    context = _run_context(capture_policy=CapturePolicy.metadata())
    agent = Agent[None, str](
        TestModel(custom_output_text="image accepted"),
        deps_type=type(None),
    )
    binary = BinaryContent(data=b"private-image", media_type="image/png")

    with InstrumentationManager() as manager:
        manager.install(PydanticAI())
        with _active(context):
            async with agent.run_stream(["describe", binary]) as stream:
                chunks = [chunk async for chunk in stream.stream_text(delta=True)]

    assert "".join(chunks) == "image accepted"
    context.finalize()
    agent_span = _spans(context, "pydantic_ai.agent.run")[0]
    input_event = next(event for event in agent_span.events if event.name == "input")
    assert input_event.body == {"type": "list", "length": 2}
    assert context.reference_store.artifacts == ()
    assert context.artifacts == []
    assert b"private-image" not in repr(context.trace).encode()
    model_span = _spans(context, "pydantic_ai.model.request")[0]
    assert any(
        measurement.semantic_type == Semantic.TIME_FIRST_CHUNK
        for measurement in model_span.measurements
    )
    assert {Semantic.STREAM_FIRST_CHUNK, Semantic.STREAM_COMPLETED} <= {
        event.semantic_type for event in model_span.events
    }

    full_context = _run_context(capture_policy=CapturePolicy.full())
    with InstrumentationManager() as manager:
        manager.install(PydanticAI())
        with _active(full_context):
            async with agent.run_stream(["describe", binary]) as stream:
                assert "".join([chunk async for chunk in stream.stream_text(delta=True)])

    full_context.finalize()
    [stored] = full_context.reference_store.artifacts
    assert stored.content == b"private-image"
    assert stored.reference.media_type == "image/png"
    assert full_context.artifacts[0].name == "input.binary.1"


@pytest.mark.asyncio
async def test_capability_is_a_noop_without_an_active_autobench_run() -> None:
    capability = _capability()
    agent_context = _agent_context()
    response = ModelResponse(parts=[TextPart("response")])
    request_context = ModelRequestContext(
        model=TestModel(),
        messages=[],
        model_settings=None,
        model_request_parameters=ModelRequestParameters(),
    )
    tool_call = ToolCallPart("tool", {})
    tool_definition = ToolDefinition(name="tool")
    output_context = OutputContext(
        mode="text",
        output_type=None,
        object_def=None,
        has_function=False,
    )

    async def run_handler() -> AgentRunResult[str]:
        return AgentRunResult("run")

    async def model_handler(context: ModelRequestContext) -> ModelResponse:
        assert context is request_context
        return response

    async def validate_handler(args: str | dict[str, Any]) -> dict[str, Any]:
        assert args == {}
        return {}

    async def execute_handler(args: dict[str, Any]) -> str:
        assert args == {}
        return "tool"

    async def output_handler(output: str | dict[str, Any]) -> str:
        assert output == "raw"
        return "output"

    assert AutobenchCapability.get_serialization_name() is None
    assert capability.get_ordering().position == "outermost"
    assert capability.has_wrap_run_event_stream is False
    assert (await capability.for_run(agent_context)) is not capability
    assert (
        await capability.handle_deferred_tool_calls(
            agent_context,
            requests=DeferredToolRequests(),
        )
        is None
    )
    assert (await capability.wrap_run(agent_context, handler=run_handler)).output == "run"
    assert (
        await capability.wrap_model_request(
            agent_context,
            request_context=request_context,
            handler=model_handler,
        )
    ) is response
    assert (
        await capability.wrap_tool_validate(
            agent_context,
            call=tool_call,
            tool_def=tool_definition,
            args={},
            handler=validate_handler,
        )
        == {}
    )
    assert (
        await capability.wrap_tool_execute(
            agent_context,
            call=tool_call,
            tool_def=tool_definition,
            args={},
            handler=execute_handler,
        )
        == "tool"
    )
    assert (
        await capability.wrap_output_validate(
            agent_context,
            output_context=output_context,
            output="raw",
            handler=output_handler,
        )
        == "output"
    )

    async def events() -> AsyncIterator[AgentStreamEvent]:
        yield PartStartEvent(index=0, part=TextPart("chunk"))

    observed = [
        event
        async for event in capability.wrap_run_event_stream(
            agent_context,
            stream=events(),
        )
    ]
    assert len(observed) == 1

    async def cancelled_events() -> AsyncIterator[AgentStreamEvent]:
        raise asyncio.CancelledError
        yield PartStartEvent(index=0, part=TextPart("unreachable"))

    async def failed_events() -> AsyncIterator[AgentStreamEvent]:
        raise RuntimeError("stream broke")
        yield PartStartEvent(index=0, part=TextPart("unreachable"))

    with pytest.raises(asyncio.CancelledError):
        async for _ in capability.wrap_run_event_stream(
            agent_context,
            stream=cancelled_events(),
        ):
            pass
    with pytest.raises(RuntimeError, match="stream broke"):
        async for _ in capability.wrap_run_event_stream(
            agent_context,
            stream=failed_events(),
        ):
            pass


@pytest.mark.asyncio
async def test_validation_deferred_and_stream_failures_emit_control_flow_evidence() -> None:
    capability = _capability()
    context = _run_context()
    agent_context = _agent_context()
    tool_call = ToolCallPart("tool", {})
    tool_definition = ToolDefinition(name="tool")
    output_context = OutputContext(
        mode="native",
        output_type=str,
        object_def=None,
        has_function=False,
    )
    text_output_context = OutputContext(
        mode="text",
        output_type=None,
        object_def=None,
        has_function=False,
    )

    async def deferred_validation(args: str | dict[str, Any]) -> dict[str, Any]:
        del args
        raise ApprovalRequired()

    async def delayed_validation(args: str | dict[str, Any]) -> dict[str, Any]:
        del args
        raise CallDeferred()

    async def rejected_validation(args: str | dict[str, Any]) -> dict[str, Any]:
        del args
        raise ModelRetry("invalid arguments")

    async def broken_validation(args: str | dict[str, Any]) -> dict[str, Any]:
        del args
        raise RuntimeError("validator broke")

    async def deferred_execution(args: dict[str, Any]) -> str:
        del args
        raise CallDeferred()

    async def approval_execution(args: dict[str, Any]) -> str:
        del args
        raise ApprovalRequired()

    async def partial_output(output: str | dict[str, Any]) -> str:
        assert output == "partial"
        return "partial-output"

    async def rejected_output(output: str | dict[str, Any]) -> str:
        del output
        raise ModelRetry("invalid output")

    async def broken_output(output: str | dict[str, Any]) -> str:
        del output
        raise RuntimeError("output validator broke")

    deferred_call = ToolCallPart("later", {})
    approval_call = ToolCallPart("approve", {})
    requests = DeferredToolRequests(
        calls=[deferred_call],
        approvals=[approval_call],
    )

    async def stream_events() -> AsyncIterator[AgentStreamEvent]:
        stream_call = ToolCallPart("streamed", {})
        yield FunctionToolCallEvent(stream_call)
        yield FunctionToolResultEvent(
            ToolReturnPart(
                tool_name="streamed",
                content="done",
                tool_call_id=stream_call.tool_call_id,
            )
        )
        yield DeferredToolResultsEvent(
            DeferredToolResults(
                calls={deferred_call.tool_call_id: "done"},
                approvals={approval_call.tool_call_id: True},
            )
        )

    async def cancelled_events() -> AsyncIterator[AgentStreamEvent]:
        raise asyncio.CancelledError
        yield PartStartEvent(index=0, part=TextPart("unreachable"))

    async def failed_events() -> AsyncIterator[AgentStreamEvent]:
        raise RuntimeError("stream broke")
        yield PartStartEvent(index=0, part=TextPart("unreachable"))

    async def run_handler() -> AgentRunResult[str]:
        assert (
            await capability.handle_deferred_tool_calls(
                agent_context,
                requests=requests,
            )
            is None
        )
        observed = [
            event
            async for event in capability.wrap_run_event_stream(
                agent_context,
                stream=stream_events(),
            )
        ]
        assert len(observed) == 3
        with pytest.raises(asyncio.CancelledError):
            async for _ in capability.wrap_run_event_stream(
                agent_context,
                stream=cancelled_events(),
            ):
                pass
        with pytest.raises(RuntimeError, match="stream broke"):
            async for _ in capability.wrap_run_event_stream(
                agent_context,
                stream=failed_events(),
            ):
                pass
        return AgentRunResult("done")

    with _active(context):
        with pytest.raises(ApprovalRequired):
            await capability.wrap_tool_validate(
                agent_context,
                call=tool_call,
                tool_def=tool_definition,
                args={},
                handler=deferred_validation,
            )
        with pytest.raises(ModelRetry):
            await capability.wrap_tool_validate(
                agent_context,
                call=tool_call,
                tool_def=tool_definition,
                args={},
                handler=rejected_validation,
            )
        with pytest.raises(CallDeferred):
            await capability.wrap_tool_validate(
                agent_context,
                call=tool_call,
                tool_def=tool_definition,
                args={},
                handler=delayed_validation,
            )
        with pytest.raises(RuntimeError, match="validator broke"):
            await capability.wrap_tool_validate(
                agent_context,
                call=tool_call,
                tool_def=tool_definition,
                args={},
                handler=broken_validation,
            )
        with pytest.raises(CallDeferred):
            await capability.wrap_tool_execute(
                agent_context,
                call=tool_call,
                tool_def=tool_definition,
                args={},
                handler=deferred_execution,
            )
        with pytest.raises(ApprovalRequired):
            await capability.wrap_tool_execute(
                agent_context,
                call=tool_call,
                tool_def=tool_definition,
                args={},
                handler=approval_execution,
            )
        assert (
            await capability.wrap_output_validate(
                _agent_context(partial_output=True),
                output_context=output_context,
                output="partial",
                handler=partial_output,
            )
            == "partial-output"
        )
        assert (
            await capability.wrap_output_validate(
                agent_context,
                output_context=text_output_context,
                output="partial",
                handler=partial_output,
            )
            == "partial-output"
        )
        with pytest.raises(ModelRetry):
            await capability.wrap_output_validate(
                agent_context,
                output_context=output_context,
                output="raw",
                handler=rejected_output,
            )
        with pytest.raises(RuntimeError, match="output validator broke"):
            await capability.wrap_output_validate(
                agent_context,
                output_context=output_context,
                output="raw",
                handler=broken_output,
            )
        result = await capability.wrap_run(agent_context, handler=run_handler)

    assert result.output == "done"
    context.finalize(output=result.output)
    semantic_types = {event.semantic_type for span in context.trace.spans for event in span.events}
    assert {
        "operation.deferred",
        "operation.retry",
        "validation.failure",
        "operation.deferred.resolved",
        "stream.partial",
        "stream.failed",
    } <= semantic_types


def test_installation_restores_entry_points_and_reports_external_patch_conflicts() -> None:
    original_agent_iter = getattr_static(Agent, "iter")
    original_wrapper_iter = getattr_static(WrapperAgent, "iter")
    manager = InstrumentationManager()
    handle = manager.install(PydanticAI())

    assert getattr_static(Agent, "iter") is not original_agent_iter
    assert getattr_static(WrapperAgent, "iter") is not original_wrapper_iter
    handle.close()
    assert getattr_static(Agent, "iter") is original_agent_iter
    assert getattr_static(WrapperAgent, "iter") is original_wrapper_iter
    manager.close()

    context = _run_context()
    manager = InstrumentationManager()
    manager.install(PydanticAI())

    def external_iter(*args: Any, **kwargs: Any) -> Any:
        del args, kwargs
        return None

    Agent.iter = external_iter
    try:
        with _active(context):
            manager.close()
    finally:
        Agent.iter = original_agent_iter
    context.finalize()
    assert any(
        diagnostic.code == "pydantic_ai_patch_conflict" for diagnostic in context.trace.diagnostics
    )


def test_a_second_manager_cannot_own_the_same_agent_entry_points() -> None:
    first = InstrumentationManager()
    second = InstrumentationManager()
    first.install(PydanticAI())
    try:
        with pytest.raises(InstrumentationConflictError, match="already instrumented"):
            second.install(PydanticAI())
    finally:
        first.close()
        second.close()


def test_installation_rolls_back_the_first_entry_point_when_the_second_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import autobench.instrumentation.pydantic_ai.instrumentor as module

    original_agent_iter = getattr_static(Agent, "iter")
    install_entry_point = module._install_entry_point

    def fail_wrapper_install(
        target: type[Any],
        capability: AutobenchCapability,
        instrumentor_id: str,
    ) -> None:
        if target is WrapperAgent:
            raise RuntimeError("wrapper patch failed")
        install_entry_point(target, capability, instrumentor_id)

    monkeypatch.setattr(module, "_install_entry_point", fail_wrapper_install)

    with pytest.raises(RuntimeError, match="wrapper patch failed"):
        PydanticAI().install(InstrumentationRuntime())

    assert getattr_static(Agent, "iter") is original_agent_iter


def test_core_replay_import_does_not_require_pydantic_ai() -> None:
    script = """
import sys
from autobench import Benchmark
from autobench.records.replay import replay_experiment

assert Benchmark is not None
assert callable(replay_experiment)
assert "pydantic_ai" not in sys.modules
"""
    completed = subprocess.run(
        [sys.executable, "-c", script],
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
