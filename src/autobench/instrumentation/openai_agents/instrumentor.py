from __future__ import annotations

from dataclasses import dataclass
from importlib.metadata import version
from importlib.util import find_spec
from threading import Lock
from typing import Any

from agents.tracing import (
    Span as AgentSpan,
)
from agents.tracing import (
    Trace as AgentTrace,
)
from agents.tracing import (
    TracingProcessor,
    add_trace_processor,
    get_trace_provider,
)
from agents.tracing.provider import DefaultTraceProvider, SynchronousMultiTracingProcessor
from agents.tracing.span_data import (
    AgentSpanData,
    CustomSpanData,
    FunctionSpanData,
    GenerationSpanData,
    GuardrailSpanData,
    HandoffSpanData,
    ResponseSpanData,
    SpanData,
    TaskSpanData,
    TurnSpanData,
)

from autobench._version import __version__
from autobench.instrumentation.manager import InstrumentationRuntime
from autobench.instrumentation.models import (
    Compatibility,
    CompatibilityStatus,
    InstrumentationError,
    InstrumentationHandle,
    InstrumentorCapabilities,
    InstrumentorInfo,
)
from autobench.metrics.semantics import Semantic
from autobench.protocol.signals import AbstractionLayer, CaptureMechanism, EndReason
from autobench.runtime.context import RunContext, Span, SpanKind, active_run_context


@dataclass(frozen=True, slots=True)
class _SpanShape:
    operation: str
    kind: str
    input: Any = None
    attributes: dict[str, Any] | None = None


@dataclass(slots=True)
class _TraceState:
    context: RunContext
    span: Span


@dataclass(slots=True)
class _SpanState:
    span: Span


class _Processor(TracingProcessor):
    def __init__(
        self,
        runtime: InstrumentationRuntime,
        info: InstrumentorInfo,
    ) -> None:
        self._runtime = runtime
        self._info = info
        self._scope = runtime.scope(info, target_version=version("openai-agents"))
        self._traces: dict[str, _TraceState] = {}
        self._spans: dict[str, _SpanState] = {}
        self._lock = Lock()
        self._active = True

    def on_trace_start(self, trace: AgentTrace) -> None:
        if not self._active:
            return
        try:
            context = active_run_context()
            if context is None:
                return
            exported = trace.export() or {}
            span = context.span(
                "openai_agents.workflow",
                kind=SpanKind.WORKFLOW,
                input=exported.get("metadata"),
                attributes={
                    Semantic.WORKFLOW_NAME: trace.name,
                    "openai_agents.trace_id": trace.trace_id,
                    "openai_agents.group_id": exported.get("group_id"),
                },
                instrumentation_scope=self._scope,
            )
            span.__enter__()
            with self._lock:
                self._traces[trace.trace_id] = _TraceState(context=context, span=span)
        except Exception as error:
            self._diagnose("trace_start", error)

    def on_trace_end(self, trace: AgentTrace) -> None:
        if not self._active:
            return
        try:
            with self._lock:
                state = self._traces.pop(trace.trace_id, None)
            if state is None:
                return
            state.span.resume()
            state.span.finish()
        except Exception as error:
            self._diagnose("trace_end", error)

    def on_span_start(self, span: AgentSpan[Any]) -> None:
        if not self._active:
            return
        try:
            context = active_run_context()
            if context is None:
                with self._lock:
                    trace_state = self._traces.get(span.trace_id)
                if trace_state is None:
                    return
                context = trace_state.context
            shape = _span_shape(span.span_data)
            attributes = dict(shape.attributes or {})
            attributes.update(
                {
                    "openai_agents.trace_id": span.trace_id,
                    "openai_agents.span_id": span.span_id,
                    "openai_agents.parent_id": span.parent_id,
                    "openai_agents.span_type": span.span_data.type,
                    "abp.logical_operation_id": span.span_id,
                }
            )
            abp_span = context.span(
                shape.operation,
                kind=shape.kind,
                input=shape.input,
                attributes=attributes,
                instrumentation_scope=self._scope,
            )
            abp_span.__enter__()
            with self._lock:
                self._spans[span.span_id] = _SpanState(span=abp_span)
        except Exception as error:
            self._diagnose("span_start", error)

    def on_span_end(self, span: AgentSpan[Any]) -> None:
        if not self._active:
            return
        try:
            with self._lock:
                state = self._spans.pop(span.span_id, None)
            if state is None:
                return
            state.span.resume()
            _complete_span(state.span, span)
        except Exception as error:
            self._diagnose("span_end", error)

    def shutdown(self) -> None:
        self.deactivate()

    def force_flush(self) -> None:
        return None

    def deactivate(self) -> None:
        if not self._active:
            return
        self._active = False
        with self._lock:
            traces = tuple(self._traces.values())
            spans = tuple(self._spans.values())
            self._traces.clear()
            self._spans.clear()
        for state in reversed(spans):
            try:
                state.span.finish(reason=EndReason.ABANDONED, partial=True)
            except Exception as error:
                self._diagnose("span_abandon", error)
        for state in reversed(traces):
            try:
                state.span.finish(reason=EndReason.ABANDONED, partial=True)
            except Exception as error:
                self._diagnose("trace_abandon", error)

    def _diagnose(self, stage: str, error: Exception) -> None:
        self._runtime.diagnose(
            self._info,
            "openai_agents_processor_error",
            f"{stage}: {type(error).__name__}: {error}",
        )


def _span_shape(data: SpanData) -> _SpanShape:
    if isinstance(data, AgentSpanData):
        return _SpanShape(
            operation="openai_agents.agent",
            kind=SpanKind.AGENT,
            attributes={
                Semantic.AGENT_NAME: data.name,
                "agent.handoffs": data.handoffs,
                "agent.tools": data.tools,
                "agent.output_type": data.output_type,
                "agent.metadata": data.metadata,
            },
        )
    if isinstance(data, FunctionSpanData):
        return _SpanShape(
            operation="openai_agents.tool",
            kind=SpanKind.TOOL,
            input=data.input,
            attributes={Semantic.TOOL_NAME: data.name, "tool.mcp": data.mcp_data},
        )
    if isinstance(data, HandoffSpanData):
        return _SpanShape(
            operation="openai_agents.handoff",
            kind="handoff",
            attributes={"handoff.from": data.from_agent, "handoff.to": data.to_agent},
        )
    if isinstance(data, GuardrailSpanData):
        return _SpanShape(
            operation="openai_agents.guardrail",
            kind="validation",
            attributes={"guardrail.name": data.name, "guardrail.triggered": data.triggered},
        )
    if isinstance(data, GenerationSpanData):
        return _SpanShape(
            operation="openai_agents.generation",
            kind="generation",
            input=data.input,
            attributes={
                Semantic.LLM_MODEL_REQUESTED: data.model,
                "model.config": data.model_config,
            },
        )
    if isinstance(data, ResponseSpanData):
        return _SpanShape(
            operation="openai_agents.response",
            kind="generation",
            input=data.input,
        )
    if isinstance(data, TaskSpanData):
        return _SpanShape(
            operation="openai_agents.task",
            kind=SpanKind.WORKFLOW,
            attributes={"task.name": data.name, "task.metadata": data.metadata},
        )
    if isinstance(data, TurnSpanData):
        return _SpanShape(
            operation="openai_agents.turn",
            kind=SpanKind.AGENT,
            attributes={"turn": data.turn, Semantic.AGENT_NAME: data.agent_name},
        )
    if isinstance(data, CustomSpanData):
        return _SpanShape(
            operation="openai_agents.custom",
            kind=SpanKind.CUSTOM,
            input=data.data,
            attributes={"custom.name": data.name},
        )
    return _SpanShape(
        operation="openai_agents.operation",
        kind=SpanKind.CUSTOM,
        input=data.export(),
        attributes={"operation.type": data.type},
    )


def _complete_span(target: Span, source: AgentSpan[Any]) -> None:
    data = source.span_data
    if isinstance(data, FunctionSpanData):
        target.event("tool.result", data.output, semantic_type=Semantic.TOOL_CALL_RESULT)
        target.set_output(data.output)
    elif isinstance(data, GenerationSpanData):
        target.set_output(data.output)
        if data.usage is not None:
            target.set_attribute("openai_agents.usage", data.usage)
    elif isinstance(data, ResponseSpanData):
        target.set_output(data.response)
        if data.response is not None:
            target.set_attribute(Semantic.LLM_MODEL_RESPONSE, data.response.model)
            target.set_attribute("response_id", data.response.id)
        if data.usage is not None:
            target.set_attribute("openai_agents.usage", data.usage)
    elif isinstance(data, GuardrailSpanData):
        target.outcome(not data.triggered, name="guardrail.passed")
    elif isinstance(data, (TaskSpanData, TurnSpanData)) and data.usage is not None:
        target.set_attribute("openai_agents.usage", data.usage)
    if source.error is None:
        target.finish()
        return
    target.event("openai_agents.error", source.error, semantic_type=Semantic.ERROR_EXCEPTION)
    target.finish(error=RuntimeError(str(source.error)))


def _remove_processor(provider: DefaultTraceProvider, processor: _Processor) -> None:
    multi = provider._multi_processor  # pyright: ignore[reportPrivateUsage]
    with multi._lock:  # pyright: ignore[reportPrivateUsage]
        multi._processors = tuple(  # pyright: ignore[reportPrivateUsage]
            current
            for current in multi._processors  # pyright: ignore[reportPrivateUsage]
            if current is not processor
        )


class OpenAIAgents:
    """Install native ABP capture through the OpenAI Agents tracing processor API."""

    def __init__(self) -> None:
        self._info = InstrumentorInfo(
            id="autobench.openai_agents",
            version=__version__,
            target_distribution="openai-agents",
            supported_versions=">=0.19.2,<0.20",
            mechanism=CaptureMechanism.HOOK,
            layer=AbstractionLayer.FRAMEWORK,
            span_kinds=("workflow", "agent", "tool", "handoff", "validation", "generation"),
            semantic_families=("workflow", "agent", "tool", "llm", "result"),
            source_convention="openai-agents",
            source_convention_version="0.19",
            capabilities=InstrumentorCapabilities(
                sync=True,
                async_=True,
                streaming=True,
                native_hooks=True,
            ),
        )

    @property
    def info(self) -> InstrumentorInfo:
        return self._info

    def check(self) -> Compatibility:
        if find_spec("agents") is None:
            return Compatibility(
                status=CompatibilityStatus.UNAVAILABLE,
                diagnostics=(
                    "OpenAI Agents is unavailable; install Autobench with the "
                    "'openai-agents' extra",
                ),
            )
        provider = get_trace_provider()
        if not isinstance(provider, DefaultTraceProvider):
            return Compatibility(
                status=CompatibilityStatus.UNSUPPORTED,
                conflicts=(
                    "the active OpenAI Agents trace provider does not support identity-safe "
                    "processor removal",
                ),
                private_seam_supported=False,
            )
        multi = provider._multi_processor  # pyright: ignore[reportPrivateUsage]
        if not isinstance(multi, SynchronousMultiTracingProcessor):
            return Compatibility(
                status=CompatibilityStatus.UNSUPPORTED,
                diagnostics=("OpenAI Agents processor registry shape is unsupported",),
                private_seam_supported=False,
            )
        return Compatibility(
            target_version=version("openai-agents"),
            private_seam_supported=True,
        )

    def install(self, runtime: InstrumentationRuntime) -> InstrumentationHandle:
        provider = get_trace_provider()
        if not isinstance(provider, DefaultTraceProvider):
            raise InstrumentationError("OpenAI Agents trace provider changed before installation")
        processor = _Processor(runtime, self.info)
        add_trace_processor(processor)

        def close() -> None:
            processor.deactivate()
            _remove_processor(provider, processor)

        return InstrumentationHandle(close, info=self.info)


__all__ = ("OpenAIAgents",)
