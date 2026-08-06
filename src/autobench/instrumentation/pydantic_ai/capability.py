from __future__ import annotations

import asyncio
from collections.abc import AsyncIterable, Mapping, Sequence
from copy import copy
from dataclasses import dataclass
from time import perf_counter
from typing import Any

from pydantic import ValidationError
from pydantic_ai import BinaryContent, DeferredToolRequests, DeferredToolResults
from pydantic_ai import RunContext as AgentRunContext
from pydantic_ai.capabilities import (
    AbstractCapability,
    CapabilityOrdering,
    RawOutput,
    RawToolArgs,
    ValidatedToolArgs,
    WrapModelRequestHandler,
    WrapOutputValidateHandler,
    WrapRunHandler,
    WrapToolExecuteHandler,
    WrapToolValidateHandler,
)
from pydantic_ai.exceptions import (
    ApprovalRequired,
    CallDeferred,
    ModelRetry,
    ToolRetryError,
)
from pydantic_ai.messages import (
    AgentStreamEvent,
    DeferredToolResultsEvent,
    ModelResponse,
    PartDeltaEvent,
    PartEndEvent,
    PartStartEvent,
    ToolCallEvent,
    ToolCallPart,
    ToolResultEvent,
    UserContent,
)
from pydantic_ai.models import ModelRequestContext
from pydantic_ai.output import OutputContext
from pydantic_ai.tools import ToolDefinition
from pydantic_ai.toolsets import AbstractToolset
from pydantic_ai.toolsets.function import FunctionToolset
from pydantic_ai.usage import RequestUsage, RunUsage

from autobench.instrumentation.config import AssetDiscoverySettings
from autobench.instrumentation.manager import InstrumentationRuntime
from autobench.instrumentation.models import InstrumentorInfo
from autobench.instrumentation.pydantic_ai.assets import AssetDiscovery
from autobench.metrics.semantics import Semantic
from autobench.protocol.signals import CaptureLevel, EndReason
from autobench.runtime.context import RunContext, Span, SpanKind, active_run_context
from autobench.tracking import TrackingRegistry, track


@dataclass(slots=True)
class _ModelStreamState:
    span: Span
    started_at: float
    first_chunk_seen: bool = False


class AutobenchCapability(AbstractCapability[Any]):
    """Pydantic AI lifecycle observer that emits native Autobench evidence."""

    id = "autobench"

    def __init__(
        self,
        runtime: InstrumentationRuntime,
        info: InstrumentorInfo,
        *,
        target_version: str,
        assets: Sequence[Any] = (),
        registry: TrackingRegistry = track,
        discovery: AssetDiscoverySettings | None = None,
    ) -> None:
        self._runtime = runtime
        self._info = info
        self._scope = runtime.scope(info, target_version=target_version)
        self._assets = tuple(assets)
        self._registry = registry
        self._asset_discovery = AssetDiscovery(
            runtime,
            info,
            target_version=target_version,
            registry=registry,
            settings=discovery or AssetDiscoverySettings(),
        )
        self._entrypoint_overrides: dict[str, Any] = {}
        self._model_stream: _ModelStreamState | None = None
        self._agent_span: Span | None = None

    def for_entrypoint(self, kwargs: Mapping[str, Any]) -> AutobenchCapability:
        capability = copy(self)
        capability._entrypoint_overrides = {
            key: kwargs[key]
            for key in ("instructions", "output_type", "toolsets", "spec")
            if key in kwargs and kwargs[key] is not None
        }
        capability._model_stream = None
        capability._agent_span = None
        return capability

    @classmethod
    def get_serialization_name(cls) -> str | None:
        return None

    def get_ordering(self) -> CapabilityOrdering:
        return CapabilityOrdering(position="outermost")

    @property
    def has_wrap_run_event_stream(self) -> bool:
        # Explicit streams and user event handlers still invoke the wrapper. Returning
        # false prevents instrumentation alone from changing request() into request_stream().
        return False

    async def for_run(self, ctx: AgentRunContext[Any]) -> AutobenchCapability:
        del ctx
        capability = copy(self)
        capability._model_stream = None
        capability._agent_span = None
        return capability

    async def wrap_run(
        self,
        ctx: AgentRunContext[Any],
        *,
        handler: WrapRunHandler,
    ) -> Any:
        run_context = active_run_context()
        if run_context is None:
            return await handler()
        attributes: dict[str, Any] = {
            Semantic.AGENT_NAME: None if ctx.agent is None else ctx.agent.name,
            Semantic.LLM_MODEL_REQUESTED: ctx.model.model_id,
            Semantic.LLM_PROVIDER_NAME: ctx.model.system,
            "pydantic_ai.run.id": ctx.run_id,
            Semantic.CONVERSATION_ID: ctx.conversation_id,
        }
        with run_context.span(
            "pydantic_ai.agent.run",
            kind=SpanKind.AGENT,
            input=ctx.prompt,
            attributes=attributes,
            instrumentation_scope=self._scope,
        ) as span:
            self._agent_span = span
            try:
                self._capture_multimodal_inputs(run_context, span, ctx.prompt)
                self._attach_run_assets(run_context, span, ctx)
                self._asset_discovery.agent(
                    ctx,
                    span_id=span.id,
                    overrides=self._entrypoint_overrides,
                )
                result = await handler()
                span.set_output(result.output)
                for name, value in _usage_values(
                    ctx.usage,
                    requests=ctx.usage.requests,
                ).items():
                    span.set_usage(name, value)
                return result
            finally:
                self._agent_span = None

    async def wrap_model_request(
        self,
        ctx: AgentRunContext[Any],
        *,
        request_context: ModelRequestContext,
        handler: WrapModelRequestHandler,
    ) -> ModelResponse:
        run_context = active_run_context()
        if run_context is None:
            return await handler(request_context)
        requested_model = request_context.model_id or request_context.model.model_id
        attributes: dict[str, Any] = {
            Semantic.LLM_MODEL_REQUESTED: requested_model,
            Semantic.LLM_PROVIDER_NAME: request_context.model.system,
            "streaming": request_context.streaming,
            "abp.logical_operation_id": f"{ctx.run_id or 'run'}:{ctx.run_step}",
            "pydantic_ai.run.id": ctx.run_id,
            Semantic.CONVERSATION_ID: ctx.conversation_id,
            "message_input_count": len(request_context.messages),
            "usage_authority": "framework",
        }
        if request_context.model_settings is not None:
            temperature = request_context.model_settings.get("temperature")
            if temperature is not None:
                attributes["temperature"] = temperature
        with run_context.span(
            "pydantic_ai.model.request",
            kind=SpanKind.LLM,
            input=request_context.messages,
            attributes=attributes,
            instrumentation_scope=self._scope,
        ) as span:
            self._asset_discovery.request(ctx, request_context, span_id=span.id)
            span.event(
                "messages.input",
                request_context.messages,
                semantic_type=Semantic.MESSAGE_INPUT,
            )
            stream_state = _ModelStreamState(
                span=span,
                started_at=perf_counter(),
            )
            self._model_stream = stream_state
            try:
                response = await handler(request_context)
            finally:
                self._model_stream = None
            span.set_attribute(Semantic.LLM_MODEL_RESPONSE, response.model_name)
            span.set_attribute(Semantic.LLM_PROVIDER_NAME, response.provider_name)
            span.set_attribute("message_output_count", 1)
            if response.provider_response_id is not None:
                span.set_attribute("response_id", response.provider_response_id)
            if response.finish_reason is not None:
                span.set_attribute("finish_reason", response.finish_reason)
            for name, value in _usage_values(response.usage, requests=1).items():
                span.set_usage(name, value)
            span.event(
                "messages.output",
                response.parts,
                semantic_type=Semantic.MESSAGE_OUTPUT,
            )
            span.set_output(response.parts)
            return response

    async def wrap_tool_validate(
        self,
        ctx: AgentRunContext[Any],
        *,
        call: ToolCallPart,
        tool_def: ToolDefinition,
        args: RawToolArgs,
        handler: WrapToolValidateHandler,
    ) -> ValidatedToolArgs:
        run_context = active_run_context()
        if run_context is None:
            return await handler(args)
        span = run_context.span(
            "pydantic_ai.tool.validate",
            kind="validation",
            input=args,
            attributes=_tool_attributes(ctx, call, tool_def),
            instrumentation_scope=self._scope,
        )
        span.__enter__()
        span.event("tool.arguments", args, semantic_type=Semantic.TOOL_CALL_ARGUMENTS)
        try:
            validated = await handler(args)
        except ApprovalRequired:
            span.event(
                "approval_requested",
                call,
                semantic_type=Semantic.APPROVAL_REQUESTED,
            )
            span.finish(reason=EndReason.DEFERRED)
            raise
        except CallDeferred:
            span.event("tool.deferred", semantic_type=Semantic.OPERATION_DEFERRED)
            span.finish(reason=EndReason.DEFERRED)
            raise
        except BaseException as error:
            if isinstance(error, (ValidationError, ModelRetry)):
                span.event(
                    "validation_failure",
                    str(error),
                    semantic_type=Semantic.VALIDATION_FAILURE,
                )
                span.event("retry", semantic_type=Semantic.OPERATION_RETRY)
            span.finish(error=error)
            raise
        span.set_output(validated)
        span.finish()
        return validated

    async def wrap_tool_execute(
        self,
        ctx: AgentRunContext[Any],
        *,
        call: ToolCallPart,
        tool_def: ToolDefinition,
        args: ValidatedToolArgs,
        handler: WrapToolExecuteHandler,
    ) -> Any:
        run_context = active_run_context()
        if run_context is None:
            return await handler(args)
        span = run_context.span(
            "pydantic_ai.tool.execute",
            kind=SpanKind.TOOL,
            input=args,
            attributes=_tool_attributes(ctx, call, tool_def),
            instrumentation_scope=self._scope,
        )
        span.__enter__()
        span.event("tool.arguments", args, semantic_type=Semantic.TOOL_CALL_ARGUMENTS)
        try:
            result = await handler(args)
        except ApprovalRequired:
            span.event(
                "approval_requested",
                call,
                semantic_type=Semantic.APPROVAL_REQUESTED,
            )
            span.finish(reason=EndReason.DEFERRED)
            raise
        except CallDeferred:
            span.event("tool.deferred", semantic_type=Semantic.OPERATION_DEFERRED)
            span.finish(reason=EndReason.DEFERRED)
            raise
        except BaseException as error:
            if isinstance(error, (ModelRetry, ToolRetryError)):
                span.event("retry", semantic_type=Semantic.OPERATION_RETRY)
            span.finish(error=error)
            raise
        span.event("tool.result", result, semantic_type=Semantic.TOOL_CALL_RESULT)
        span.set_output(result)
        span.finish()
        return result

    async def wrap_output_validate(
        self,
        ctx: AgentRunContext[Any],
        *,
        output_context: OutputContext,
        output: RawOutput,
        handler: WrapOutputValidateHandler,
    ) -> Any:
        if ctx.partial_output:
            return await handler(output)
        run_context = active_run_context()
        if run_context is None:
            return await handler(output)
        span = run_context.span(
            "pydantic_ai.output.validate",
            kind="validation",
            input=output,
            attributes={
                "output_mode": output_context.mode,
                "function_name": output_context.function_name,
                "retry": ctx.retry,
                "max_retries": ctx.max_retries,
            },
            instrumentation_scope=self._scope,
        )
        span.__enter__()
        if output_context.output_type is not None:
            self._attach_if_tracked(run_context, span, output_context.output_type)
            self._asset_discovery.output(ctx, output_context, span_id=span.id)
        try:
            validated = await handler(output)
        except BaseException as error:
            if isinstance(error, (ValidationError, ModelRetry)):
                span.event(
                    "validation_failure",
                    str(error),
                    semantic_type=Semantic.VALIDATION_FAILURE,
                )
                span.event("retry", semantic_type=Semantic.OPERATION_RETRY)
            span.finish(error=error)
            raise
        span.set_output(validated)
        span.finish()
        return validated

    async def handle_deferred_tool_calls(
        self,
        ctx: AgentRunContext[Any],
        *,
        requests: DeferredToolRequests,
    ) -> DeferredToolResults | None:
        del ctx
        if self._agent_span is None:
            return None
        for approval in requests.approvals:
            self._agent_span.event(
                "approval_requested",
                approval,
                semantic_type=Semantic.APPROVAL_REQUESTED,
            )
        for call in requests.calls:
            self._agent_span.event(
                "tool.deferred",
                call,
                semantic_type=Semantic.OPERATION_DEFERRED,
            )
        return None

    async def wrap_run_event_stream(
        self,
        ctx: AgentRunContext[Any],
        *,
        stream: AsyncIterable[AgentStreamEvent],
    ) -> AsyncIterable[AgentStreamEvent]:
        state = self._model_stream
        if state is None and self._agent_span is not None:
            state = _ModelStreamState(
                span=self._agent_span,
                started_at=perf_counter(),
                first_chunk_seen=True,
            )
        try:
            async for event in stream:
                if state is not None:
                    self._record_stream_event(state, event)
                yield event
        except asyncio.CancelledError:
            if state is not None:
                state.span.event("stream.partial", semantic_type=Semantic.STREAM_PARTIAL)
            raise
        except BaseException:
            if state is not None:
                state.span.event("stream.failed", semantic_type=Semantic.STREAM_FAILED)
            raise
        else:
            if state is not None:
                state.span.event("stream.completed", semantic_type=Semantic.STREAM_COMPLETED)

    def _record_stream_event(
        self,
        state: _ModelStreamState,
        event: AgentStreamEvent,
    ) -> None:
        if not state.first_chunk_seen and isinstance(
            event,
            (PartStartEvent, PartDeltaEvent, PartEndEvent),
        ):
            state.first_chunk_seen = True
            state.span.metric(
                "time.first_chunk",
                perf_counter() - state.started_at,
                semantic_type=Semantic.TIME_FIRST_CHUNK,
                unit="s",
            )
            state.span.event("stream.first_chunk", semantic_type=Semantic.STREAM_FIRST_CHUNK)
        if isinstance(event, ToolCallEvent):
            state.span.event(
                "tool.requested",
                event.part,
                semantic_type=Semantic.TOOL_CALL_REQUESTED,
            )
        elif isinstance(event, ToolResultEvent):
            state.span.event(
                "tool.result",
                event.part,
                semantic_type=Semantic.TOOL_CALL_RESULT,
            )
        elif isinstance(event, DeferredToolResultsEvent):
            state.span.event(
                "tool.deferred.resolved",
                event.results,
                semantic_type=Semantic.OPERATION_DEFERRED_RESOLVED,
            )

    def _capture_multimodal_inputs(
        self,
        run_context: RunContext,
        span: Span,
        prompt: str | Sequence[UserContent] | None,
    ) -> None:
        if prompt is None or isinstance(prompt, str):
            return
        policy = run_context.capture_policy
        for index, part in enumerate(prompt):
            if not isinstance(part, BinaryContent):
                continue
            metadata = {
                "identifier": part.identifier,
                "media_type": part.media_type,
                "size_bytes": len(part.data),
            }
            span.event(
                "input.binary",
                metadata,
                semantic_type=Semantic.MESSAGE_INPUT,
                tags={"content_kind": "binary"},
            )
            if policy.level_for(Semantic.MESSAGE_INPUT, part.data) is CaptureLevel.FULL:
                span.artifact(
                    f"input.binary.{index}",
                    part.data,
                    media_type=part.media_type,
                    tags=metadata,
                )

    def _attach_run_assets(
        self,
        run_context: RunContext,
        span: Span,
        ctx: AgentRunContext[Any],
    ) -> None:
        for asset in self._assets:
            self._attach_if_tracked(run_context, span, asset)
        if ctx.agent is None:
            return
        self._attach_if_tracked(run_context, span, ctx.agent.output_type)

        def visit(toolset: AbstractToolset[Any]) -> None:
            if isinstance(toolset, FunctionToolset):
                for tool in toolset.tools.values():
                    self._attach_if_tracked(run_context, span, tool.function)

        for toolset in ctx.agent.toolsets:
            toolset.apply(visit)

    def _attach_if_tracked(
        self,
        run_context: RunContext,
        span: Span,
        target: Any,
    ) -> None:
        if isinstance(target, Sequence) and not isinstance(target, (str, bytes, bytearray)):
            for item in target:
                self._attach_if_tracked(run_context, span, item)
            return
        try:
            run_context.attach_tracked_asset(
                target,
                registry=self._registry,
                span_id=span.id,
            )
        except KeyError:
            return


def _tool_attributes(
    ctx: AgentRunContext[Any],
    call: ToolCallPart,
    tool_def: ToolDefinition,
) -> dict[str, Any]:
    return {
        Semantic.TOOL_NAME: tool_def.name,
        Semantic.TOOL_TYPE: tool_def.kind,
        Semantic.TOOL_CALL_ID: call.tool_call_id,
        "retry": ctx.retry,
        "max_retries": ctx.max_retries,
        "toolset_id": tool_def.toolset_id,
    }


def _usage_values(
    usage: RequestUsage | RunUsage,
    *,
    requests: int,
) -> dict[str, int]:
    values = {
        "input_tokens": usage.input_tokens,
        "output_tokens": usage.output_tokens,
        "total_tokens": usage.total_tokens,
        "cache_read_tokens": usage.cache_read_tokens,
        "cache_write_tokens": usage.cache_write_tokens,
        "requests": requests,
    }
    for key in ("reasoning_tokens", "reasoning_output_tokens"):
        reasoning_tokens = usage.details.get(key)
        if reasoning_tokens is not None:
            values["reasoning_tokens"] = reasoning_tokens
            break
    return values


__all__ = ("AutobenchCapability",)
