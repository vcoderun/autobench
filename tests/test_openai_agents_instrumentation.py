from __future__ import annotations

import asyncio
from collections.abc import Iterator
from contextlib import contextmanager
from inspect import getattr_static
from typing import Any

import httpx
import pytest
from agents import (
    Agent,
    GuardrailFunctionOutput,
    InputGuardrail,
    OutputGuardrail,
    RunContextWrapper,
    WebSearchTool,
    function_tool,
    handoff,
)
from agents.items import TResponseInputItem
from agents.mcp import MCPServerStdio
from agents.tracing import (
    Span as AgentSpan,
)
from agents.tracing import (
    Trace as AgentTrace,
)
from agents.tracing import (
    TracingProcessor,
    add_trace_processor,
    agent_span,
    custom_span,
    function_span,
    generation_span,
    get_trace_provider,
    guardrail_span,
    handoff_span,
    response_span,
    set_trace_provider,
    task_span,
    trace,
    transcription_span,
    turn_span,
)
from agents.tracing.provider import DefaultTraceProvider
from openai import OpenAI
from openai.types.responses import Response
from pydantic import BaseModel

from autobench import Case, InstrumentationManager, RunContext, Variant
from autobench.evaluation.extraction import ExtractionContext, UsageExtractor
from autobench.instrumentation.config import AssetDiscoverySettings
from autobench.instrumentation.manager import InstrumentationRuntime
from autobench.instrumentation.models import (
    InstrumentationError,
    InstrumentationHandle,
    InstrumentorInfo,
)
from autobench.instrumentation.openai import OpenAIClient
from autobench.instrumentation.openai_agents import OpenAIAgents
from autobench.instrumentation.openai_agents import assets as agent_assets
from autobench.instrumentation.openai_agents import instrumentor as agents_instrumentation
from autobench.instrumentation.openai_agents.assets import AssetDiscovery
from autobench.instrumentation.patching import CallHandler
from autobench.metrics.semantics import DEFAULT_SEMANTIC_REGISTRY, Semantic
from autobench.protocol.signals import EndReason
from autobench.runtime.context import Span
from autobench.runtime.instrumentation import reset_active_run_context, set_active_run_context
from autobench.tracking import TrackingRegistry


class _Recorder(TracingProcessor):
    def __init__(self) -> None:
        self.events: list[str] = []

    def on_trace_start(self, trace: AgentTrace) -> None:
        self.events.append(f"trace.start:{trace.trace_id}")

    def on_trace_end(self, trace: AgentTrace) -> None:
        self.events.append(f"trace.end:{trace.trace_id}")

    def on_span_start(self, span: AgentSpan[Any]) -> None:
        self.events.append(f"span.start:{span.span_data.type}")

    def on_span_end(self, span: AgentSpan[Any]) -> None:
        self.events.append(f"span.end:{span.span_data.type}")

    def shutdown(self) -> None:
        self.events.append("shutdown")

    def force_flush(self) -> None:
        self.events.append("flush")


@contextmanager
def _isolated_provider() -> Iterator[DefaultTraceProvider]:
    original = get_trace_provider()
    provider = DefaultTraceProvider()
    set_trace_provider(provider)
    try:
        yield provider
    finally:
        set_trace_provider(original)


def _run_context() -> RunContext:
    return RunContext(
        benchmark_id="openai-agents",
        case=Case(id="case", input={"prompt": "hello"}),
        variant=Variant(id="variant"),
    )


def _response() -> Response:
    return Response.model_validate(
        {
            "id": "resp_agents",
            "object": "response",
            "created_at": 1,
            "status": "completed",
            "error": None,
            "incomplete_details": None,
            "instructions": None,
            "max_output_tokens": None,
            "model": "gpt-agent",
            "output": [],
            "parallel_tool_calls": True,
            "previous_response_id": None,
            "reasoning": None,
            "store": True,
            "temperature": 1.0,
            "text": {"format": {"type": "text"}},
            "tool_choice": "auto",
            "tools": [],
            "top_p": 1.0,
            "truncation": "disabled",
            "usage": {
                "input_tokens": 5,
                "input_tokens_details": {"cached_tokens": 0, "cache_write_tokens": 0},
                "output_tokens": 2,
                "output_tokens_details": {"reasoning_tokens": 0},
                "total_tokens": 7,
            },
        }
    )


def test_openai_agents_maps_native_trace_span_types_and_preserves_processors() -> None:
    with _isolated_provider():
        existing = _Recorder()
        add_trace_processor(existing)
        instrumentor = OpenAIAgents()
        assert instrumentor.check().private_seam_supported is True
        manager = InstrumentationManager()
        handle = manager.install(instrumentor)
        late = _Recorder()
        add_trace_processor(late)
        ctx = _run_context()
        token = set_active_run_context(ctx)
        try:
            with (
                trace("research", group_id="group_1", metadata={"tenant": "test"}),
                agent_span(
                    "planner",
                    handoffs=["writer"],
                    tools=["lookup"],
                    output_type="Answer",
                ),
            ):
                with function_span("lookup", input='{"id":1}') as tool:
                    tool.span_data.output = {"name": "record"}
                with handoff_span("planner", "writer"):
                    pass
                with guardrail_span("safe", triggered=False):
                    pass
                with generation_span(
                    input=[{"role": "user", "content": "hello"}],
                    model="gpt-agent",
                    model_config={"temperature": 0.2},
                ) as generation:
                    generation.span_data.output = [{"role": "assistant", "content": "done"}]
                    generation.span_data.usage = {
                        "input_tokens": 5,
                        "output_tokens": 2,
                    }
                with generation_span(model="gpt-agent-no-usage"):
                    pass
                with response_span(_response()) as response_operation:
                    response_operation.span_data.input = "hello"
                    response_operation.span_data.usage = {"total_tokens": 7}
                with response_span():
                    pass
                with task_span("collect") as task:
                    task.span_data.metadata = {"source": "test"}
                    task.span_data.usage = {"requests": 1}
                with task_span("collect-without-usage"):
                    pass
                with turn_span(1, "planner") as turn:
                    turn.span_data.usage = {"requests": 1}
                with turn_span(2, "planner"):
                    pass
                with custom_span("checkpoint", {"step": 1}):
                    pass
                with transcription_span(input="audio", model="whisper-test"):
                    pass
                with function_span("broken") as failed:
                    failed.set_error({"message": "tool failed", "data": {"retryable": False}})
        finally:
            reset_active_run_context(token)
            handle.close()
            manager.close()

        operation_names = {span.operation for span in ctx.trace.spans}
        assert {
            "openai_agents.workflow",
            "openai_agents.agent",
            "openai_agents.tool",
            "openai_agents.handoff",
            "openai_agents.guardrail",
            "openai_agents.generation",
            "openai_agents.response",
            "openai_agents.task",
            "openai_agents.turn",
            "openai_agents.custom",
            "openai_agents.operation",
        } <= operation_names
        assert any(
            span.errors for span in ctx.trace.spans if span.operation == "openai_agents.tool"
        )
        generation_record = next(
            span for span in ctx.spans if span.name == "openai_agents.generation"
        )
        assert generation_record.kind == "generation"
        assert generation_record.attributes["openai_agents.usage"]["input_tokens"] == 5
        response_record = next(span for span in ctx.spans if span.name == "openai_agents.response")
        assert response_record.attributes["llm.model.response"] == "gpt-agent"
        assert existing.events
        assert late.events

        existing_count = len(existing.events)
        late_count = len(late.events)
        with trace("after-close"), custom_span("still-native"):
            pass
        assert len(existing.events) > existing_count
        assert len(late.events) > late_count
        assert len(ctx.spans) == 16


def test_openai_agents_discovers_public_agent_definitions_without_running_callbacks() -> None:
    class Answer(BaseModel):
        value: str

    callback_calls = {"instructions": 0, "input_guardrail": 0, "output_guardrail": 0}

    async def instructions(
        context: RunContextWrapper[None],
        agent: Agent[None],
    ) -> str:
        del context, agent
        callback_calls["instructions"] += 1
        return "Do not execute during discovery."

    async def input_guardrail(
        context: RunContextWrapper[None],
        agent: Agent[Any],
        input_value: str | list[TResponseInputItem],
    ) -> GuardrailFunctionOutput:
        del context, agent, input_value
        callback_calls["input_guardrail"] += 1
        return GuardrailFunctionOutput(output_info=None, tripwire_triggered=False)

    async def output_guardrail(
        context: RunContextWrapper[None],
        agent: Agent[Any],
        output: Any,
    ) -> GuardrailFunctionOutput:
        del context, agent, output
        callback_calls["output_guardrail"] += 1
        return GuardrailFunctionOutput(output_info=None, tripwire_triggered=False)

    @function_tool
    def lookup(query: str) -> str:
        return query

    writer = Agent[None](name="writer", instructions="Write the final answer.")
    reviewer = Agent[None](name="reviewer", instructions="Review the answer.")
    mcp_server = MCPServerStdio(
        params={"command": "echo", "args": ["mcp"]},
        name="local-mcp",
    )
    planner = Agent[None](
        name="planner",
        instructions=instructions,
        prompt={"id": "pmpt_planner", "version": "2"},
        tools=[lookup, WebSearchTool()],
        mcp_servers=[mcp_server],
        handoffs=[writer, writer, handoff(reviewer)],
        output_type=Answer,
        input_guardrails=[InputGuardrail(input_guardrail)],
        output_guardrails=[OutputGuardrail(output_guardrail)],
    )
    registry = TrackingRegistry()
    runtime = InstrumentationRuntime(registry=registry)
    instrumentor = OpenAIAgents(registry=registry)
    discovery = AssetDiscovery(
        runtime,
        instrumentor.info,
        target_version="0.19.2",
        registry=registry,
        settings=AssetDiscoverySettings(),
    )
    handler = agents_instrumentation._RunnerHandler(runtime, instrumentor.info, discovery)
    assert handler.suppression_keys == (
        "autobench.openai_agents",
        "openai_agents",
    )
    context = _run_context()
    token = set_active_run_context(context)
    try:
        handler.begin(
            agents_instrumentation.InstrumentCall(
                instance=None,
                args=(planner, "hello"),
                kwargs={},
            )
        )
        handler.begin(
            agents_instrumentation.InstrumentCall(
                instance=None,
                args=(),
                kwargs={"starting_agent": planner},
            )
        )
        handler.begin(
            agents_instrumentation.InstrumentCall(
                instance=None,
                args=("not-an-agent",),
                kwargs={},
            )
        )
        handler.diagnose("test", RuntimeError("discovery callback"))
        handler.close()
    finally:
        reset_active_run_context(token)

    assert callback_calls == {
        "instructions": 0,
        "input_guardrail": 0,
        "output_guardrail": 0,
    }
    assert {
        "openai_agents:agent:planner:agent:self",
        "openai_agents:agent:planner:guardrail:input:input_guardrail",
        "openai_agents:agent:planner:guardrail:output:output_guardrail",
        "openai_agents:agent:planner:handoff:transfer_to_reviewer",
        "openai_agents:agent:planner:handoff:writer",
        "openai_agents:agent:planner:output_schema:output",
        "openai_agents:agent:planner:policy:tool_use",
        "openai_agents:agent:planner:prompt:instructions",
        "openai_agents:agent:planner:prompt:prompt",
        "openai_agents:agent:planner:tool:lookup",
        "openai_agents:agent:planner:tool:web_search",
        "openai_agents:agent:planner:toolset:mcp:MCPServerStdio",
        "openai_agents:agent:writer:agent:self",
    } <= {version.asset_id for version in context.asset_versions}
    assert any(
        diagnostic.code == "openai_agents_asset_discovery_error"
        for diagnostic in context.trace.diagnostics
    )
    assert agent_assets._public_dataclass("plain") == "plain"

    class TypeNamedTool:
        type = "type_named"

    class NamelessTool:
        pass

    class InvalidTypeTool:
        type = None

    assert agent_assets._tool_name(TypeNamedTool()) == "type_named"
    assert agent_assets._tool_name(NamelessTool()) == "NamelessTool"
    assert agent_assets._tool_name(InvalidTypeTool()) == "InvalidTypeTool"


def test_openai_agents_asset_discovery_can_select_only_agent_composites() -> None:
    registry = TrackingRegistry()
    runtime = InstrumentationRuntime(registry=registry)
    instrumentor = OpenAIAgents(registry=registry)
    discovery = AssetDiscovery(
        runtime,
        instrumentor.info,
        target_version="0.19.2",
        registry=registry,
        settings=AssetDiscoverySettings(include=("agent",)),
    )
    context = _run_context()
    token = set_active_run_context(context)
    try:
        discovery.agent(Agent[None](name="filtered", instructions="Do not persist this prompt."))
        discovery.agent(Agent[None](name="empty"))
    finally:
        reset_active_run_context(token)

    assert [asset.kind for asset in registry.definitions] == ["agent", "agent"]
    assert context.asset_uses[0].source_locator == "openai_agents:agent:filtered:agent:self"


async def test_openai_agents_parallel_tools_keep_the_agent_parent() -> None:
    async def run_tool(name: str) -> None:
        await asyncio.sleep(0)
        with function_span(name, input="{}") as operation:
            await asyncio.sleep(0)
            operation.span_data.output = "ok"

    with _isolated_provider():
        manager = InstrumentationManager()
        handle = manager.install(OpenAIAgents())
        ctx = _run_context()
        token = set_active_run_context(ctx)
        try:
            with trace("parallel"), agent_span("planner"):
                await asyncio.gather(run_tool("first"), run_tool("second"))
        finally:
            reset_active_run_context(token)
            handle.close()
            manager.close()

    agent = next(span for span in ctx.spans if span.name == "openai_agents.agent")
    tools = [span for span in ctx.spans if span.name == "openai_agents.tool"]
    assert len(tools) == 2
    assert {tool.parent_id for tool in tools} == {agent.id}


def test_openai_agents_and_client_compose_without_duplicate_llm_accounting() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "id": "chatcmpl-agent",
                "object": "chat.completion",
                "created": 1,
                "model": "gpt-agent-2026",
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "done"},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"prompt_tokens": 5, "completion_tokens": 2, "total_tokens": 7},
            },
        )

    with _isolated_provider():
        manager = InstrumentationManager()
        agents_handle = manager.install(OpenAIAgents())
        client_handle = manager.install(OpenAIClient())
        ctx = _run_context()
        token = set_active_run_context(ctx)
        client = OpenAI(
            api_key="test",
            base_url="https://openai.test/v1",
            http_client=httpx.Client(transport=httpx.MockTransport(handler)),
        )
        try:
            with trace("combined"), generation_span(model="gpt-agent") as generation:
                result = client.chat.completions.create(
                    model="gpt-agent",
                    messages=[{"role": "user", "content": "hello"}],
                )
                generation.span_data.output = [
                    {"role": "assistant", "content": result.choices[0].message.content}
                ]
                generation.span_data.usage = {"total_tokens": 7}
        finally:
            reset_active_run_context(token)
            client_handle.close()
            agents_handle.close()
            manager.close()

    llm_spans = [span for span in ctx.trace.spans if span.kind == "llm"]
    assert len(llm_spans) == 1
    generation_record = next(
        span for span in ctx.trace.spans if span.operation == "openai_agents.generation"
    )
    assert llm_spans[0].parent_span_id == generation_record.span_id
    evidence = UsageExtractor().extract(
        ctx.trace,
        registry=DEFAULT_SEMANTIC_REGISTRY,
        context=ExtractionContext(
            run_id=ctx.run_id,
            benchmark_id=ctx.benchmark_id,
            experiment_id=ctx.experiment_id,
            case_id=ctx.case.id,
            variant_id=ctx.variant.id,
        ),
    )
    total = next(
        observation
        for observation in evidence.observations
        if observation.semantic_type == Semantic.LLM_TOKENS_TOTAL
        and observation.tags.get("abp.summary") is True
    )
    assert total.value == 7


def test_openai_agents_close_abandons_open_native_work_without_breaking_callbacks() -> None:
    with _isolated_provider():
        manager = InstrumentationManager()
        handle = manager.install(OpenAIAgents())
        ctx = _run_context()
        token = set_active_run_context(ctx)
        workflow = trace("interrupted")
        workflow.start(mark_as_current=True)
        agent = agent_span("planner")
        agent.start(mark_as_current=True)
        handle.close()
        agent.finish(reset_current=True)
        workflow.finish(reset_current=True)
        reset_active_run_context(token)
        manager.close()

    interrupted = [
        span
        for span in ctx.trace.spans
        if span.operation in {"openai_agents.workflow", "openai_agents.agent"}
    ]
    assert len(interrupted) == 2
    assert {span.end_reason for span in interrupted} == {EndReason.ABANDONED}
    assert all(span.partial for span in interrupted)


def test_openai_agents_without_an_active_run_preserves_native_tracing() -> None:
    with _isolated_provider():
        existing = _Recorder()
        add_trace_processor(existing)
        manager = InstrumentationManager()
        handle = manager.install(OpenAIAgents())
        try:
            with trace("outside-autobench"), function_span("native-tool"):
                pass
        finally:
            handle.close()
            manager.close()

    assert any(event.startswith("trace.start:") for event in existing.events)
    assert "span.start:function" in existing.events


def test_openai_agents_processor_isolates_callback_failures_and_late_callbacks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = InstrumentationRuntime()
    instrumentor = OpenAIAgents()
    processor = agents_instrumentation._Processor(runtime, instrumentor.info)

    missing_trace = trace("missing")
    missing_span = function_span("missing")
    processor.on_trace_end(missing_trace)
    processor.on_span_end(missing_span)

    ctx = _run_context()
    token = set_active_run_context(ctx)
    workflow = trace("fallback")
    processor.on_trace_start(workflow)
    trace_state = processor._traces[workflow.trace_id]
    trace_state.span.suspend()
    reset_active_run_context(token)

    tool = function_span("fallback-tool", parent=workflow)
    processor.on_span_start(tool)
    tool.span_data.output = "ok"
    processor.on_span_end(tool)
    processor.on_trace_end(workflow)
    assert {span.name for span in ctx.spans} == {
        "openai_agents.workflow",
        "openai_agents.tool",
    }

    callback_ctx = _run_context()
    callback_token = set_active_run_context(callback_ctx)
    callback_trace = trace("callback-errors")
    callback_tool = function_span("callback-tool", parent=callback_trace)
    processor.on_trace_start(callback_trace)
    processor.on_span_start(callback_tool)
    original_finish = Span.finish

    def finish_then_fail(
        self: Span,
        *,
        error: BaseException | None = None,
        reason: EndReason | None = None,
        partial: bool | None = None,
    ) -> None:
        original_finish(self, error=error, reason=reason, partial=partial)
        raise RuntimeError("processor callback failed")

    monkeypatch.setattr(Span, "finish", finish_then_fail)
    processor.on_span_end(callback_tool)
    processor.on_trace_end(callback_trace)
    monkeypatch.setattr(Span, "finish", original_finish)
    assert any(
        diagnostic.code == "openai_agents_processor_error"
        for diagnostic in callback_ctx.trace.diagnostics
    )

    late_ctx = _run_context()
    late_token = set_active_run_context(late_ctx)
    late_ctx.finalize()
    processor.on_trace_start(trace("late-trace"))
    processor.on_span_start(function_span("late-span"))
    reset_active_run_context(late_token)

    abandoned_ctx = _run_context()
    abandoned_token = set_active_run_context(abandoned_ctx)
    abandoned_trace = trace("abandoned-errors")
    abandoned_span = function_span("abandoned-tool", parent=abandoned_trace)
    processor.on_trace_start(abandoned_trace)
    processor.on_span_start(abandoned_span)
    monkeypatch.setattr(Span, "finish", finish_then_fail)
    processor.deactivate()
    monkeypatch.setattr(Span, "finish", original_finish)
    processor.deactivate()
    processor.force_flush()
    processor.shutdown()
    processor.on_trace_start(trace("inactive"))
    processor.on_trace_end(trace("inactive"))
    processor.on_span_start(function_span("inactive"))
    processor.on_span_end(function_span("inactive"))
    reset_active_run_context(abandoned_token)
    reset_active_run_context(callback_token)

    assert any(
        diagnostic.code == "openai_agents_processor_error"
        for diagnostic in abandoned_ctx.trace.diagnostics
    )


def test_openai_agents_compatibility_rejects_unowned_provider_shapes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    instrumentor = OpenAIAgents()
    with monkeypatch.context() as patch:
        patch.setattr(agents_instrumentation, "find_spec", lambda name: None)
        assert instrumentor.check().available is False

    with monkeypatch.context() as patch:
        patch.setattr(agents_instrumentation.Runner, "run", None)
        compatibility = instrumentor.check()
        assert compatibility.supported is False
        assert "run" in compatibility.diagnostics[0]

    with monkeypatch.context() as patch:
        patch.setattr(agents_instrumentation, "get_trace_provider", lambda: None)
        compatibility = instrumentor.check()
        assert compatibility.supported is False
        assert compatibility.private_seam_supported is False
        with pytest.raises(InstrumentationError, match="provider changed"):
            instrumentor.install(InstrumentationRuntime())

    provider = DefaultTraceProvider()
    with monkeypatch.context() as patch:
        patch.setattr(provider, "_multi_processor", None)
        patch.setattr(agents_instrumentation, "get_trace_provider", lambda: provider)
        compatibility = instrumentor.check()
        assert compatibility.supported is False
        assert compatibility.private_seam_supported is False


def test_openai_agents_install_rolls_back_processor_and_runner_patches(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with _isolated_provider() as provider:
        runtime = InstrumentationRuntime()
        original_run = getattr_static(agents_instrumentation.Runner, "run")
        patch_method = runtime.patch_method

        def fail_second_patch(
            info: InstrumentorInfo,
            target: type[Any],
            attribute: str,
            handler: CallHandler,
            *,
            expected_descriptor: Any = None,
        ) -> InstrumentationHandle:
            if attribute == "run_sync":
                raise RuntimeError("runner patch failed")
            return patch_method(
                info,
                target,
                attribute,
                handler,
                expected_descriptor=expected_descriptor,
            )

        monkeypatch.setattr(runtime, "patch_method", fail_second_patch)
        with pytest.raises(RuntimeError, match="runner patch failed"):
            OpenAIAgents().install(runtime)

        assert getattr_static(agents_instrumentation.Runner, "run") is original_run
        processors = provider._multi_processor._processors  # pyright: ignore[reportPrivateUsage]
        assert not any(
            isinstance(processor, agents_instrumentation._Processor) for processor in processors
        )
