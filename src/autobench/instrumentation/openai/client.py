from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from importlib.metadata import version
from importlib.util import find_spec
from threading import Lock
from time import perf_counter
from typing import Any, Literal, Protocol, runtime_checkable
from weakref import WeakKeyDictionary

from openai import APIResponse, AsyncAPIResponse
from openai._legacy_response import LegacyAPIResponse
from openai.lib.streaming.chat import AsyncChatCompletionStream, ChatCompletionStream
from openai.lib.streaming.responses import AsyncResponseStream, ResponseStream
from openai.resources.chat.completions import AsyncCompletions, Completions
from openai.resources.embeddings import AsyncEmbeddings, Embeddings
from openai.resources.responses import AsyncResponses, Responses
from openai.types.chat import ChatCompletion, ChatCompletionChunk
from openai.types.completion_usage import CompletionUsage
from openai.types.create_embedding_response import CreateEmbeddingResponse, Usage
from openai.types.responses import (
    Response,
    ResponseCompletedEvent,
    ResponseCreatedEvent,
    ResponseFailedEvent,
    ResponseFunctionCallArgumentsDeltaEvent,
    ResponseFunctionCallArgumentsDoneEvent,
    ResponseFunctionToolCall,
    ResponseIncompleteEvent,
    ResponseInProgressEvent,
    ResponseOutputItemAddedEvent,
    ResponseOutputItemDoneEvent,
    ResponseQueuedEvent,
    ResponseTextDeltaEvent,
)
from openai.types.responses.response_usage import ResponseUsage

from autobench._version import __version__
from autobench.instrumentation.manager import InstrumentationRuntime
from autobench.instrumentation.models import (
    Compatibility,
    CompatibilityStatus,
    InstrumentationHandle,
    InstrumentCall,
    InstrumentorCapabilities,
    InstrumentorInfo,
)
from autobench.instrumentation.patching import CallLifecycle
from autobench.metrics.semantics import Semantic
from autobench.protocol.signals import AbstractionLayer, CaptureMechanism, EndReason
from autobench.runtime.context import Span, SpanKind, active_run_context

RawAction = Literal["parse", "close"]
RawResponse = APIResponse[Any] | AsyncAPIResponse[Any] | LegacyAPIResponse[Any]


@runtime_checkable
class Cancelable(Protocol):
    def cancel(self) -> None: ...


@dataclass(frozen=True, slots=True)
class _Endpoint:
    name: str
    input_keys: tuple[str, ...]


@dataclass(slots=True)
class _ToolCallState:
    id: str | None = None
    name: str | None = None
    arguments: str = ""

    def payload(self) -> dict[str, str]:
        value = {"arguments": self.arguments}
        if self.id is not None:
            value["id"] = self.id
        if self.name is not None:
            value["name"] = self.name
        return value


@dataclass(slots=True)
class _StreamState:
    started_at: float
    chunk_count: int = 0
    first_chunk_seen: bool = False
    text: list[str] = field(default_factory=list)
    finish_reasons: set[str] = field(default_factory=set)
    tool_calls: dict[str, _ToolCallState] = field(default_factory=dict)


class _RawRegistry:
    def __init__(self) -> None:
        self._calls: WeakKeyDictionary[RawResponse, _OpenAICall] = WeakKeyDictionary()
        self._lock = Lock()

    def register(self, response: RawResponse, call: _OpenAICall) -> None:
        with self._lock:
            self._calls[response] = call

    def take(self, response: RawResponse) -> _OpenAICall | None:
        with self._lock:
            return self._calls.pop(response, None)


class _OpenAICall:
    def __init__(
        self,
        handler: _EndpointHandler,
        call: InstrumentCall,
        span: Span,
    ) -> None:
        self._handler = handler
        self._call = call
        self._span = span
        self._stream = _StreamState(started_at=perf_counter())
        self._finished = False

    def resume(self) -> None:
        self._span.resume()

    def suspend(self) -> None:
        self._span.suspend()

    def observe(self, item: Any) -> None:
        if not self._stream.first_chunk_seen:
            self._stream.first_chunk_seen = True
            self._span.metric(
                "time.first_chunk",
                perf_counter() - self._stream.started_at,
                semantic_type=Semantic.TIME_FIRST_CHUNK,
                unit="s",
            )
            self._span.event("stream.first_chunk", semantic_type=Semantic.STREAM_FIRST_CHUNK)
        self._stream.chunk_count += 1
        if isinstance(item, ChatCompletionChunk):
            self._capture_chat_chunk(item)
        else:
            self._capture_response_event(item)

    def finish(
        self,
        *,
        result: Any = None,
        error: BaseException | None = None,
        reason: EndReason = EndReason.COMPLETED,
        partial: bool = False,
    ) -> None:
        if self._finished:
            return
        if isinstance(result, (APIResponse, AsyncAPIResponse, LegacyAPIResponse)):
            self._handler.raw.register(result, self)
            self.suspend()
            return
        if isinstance(error, asyncio.CancelledError):
            reason = EndReason.CANCELLED
            partial = True
        self.resume()
        try:
            if isinstance(result, ChatCompletion):
                self._capture_chat_response(result)
            elif isinstance(result, Response):
                self._capture_response(result)
            elif isinstance(result, CreateEmbeddingResponse):
                self._capture_embeddings(result)
            elif self._stream.chunk_count:
                self._finish_stream_output()
            if self._stream.chunk_count:
                self._span.set_attribute("stream.chunk_count", self._stream.chunk_count)
                event = "stream.completed" if error is None and not partial else "stream.partial"
                semantic = (
                    Semantic.STREAM_COMPLETED
                    if event == "stream.completed"
                    else Semantic.STREAM_PARTIAL
                )
                self._span.event(event, semantic_type=semantic)
            if error is not None and reason is not EndReason.CANCELLED:
                self._span.event("stream.failed", semantic_type=Semantic.STREAM_FAILED)
            self._span.finish(error=error, reason=reason, partial=partial)
        finally:
            self._finished = True
            self._handler.discard(self)

    def finish_raw(
        self,
        *,
        result: Any = None,
        error: BaseException | None = None,
        closed: bool = False,
    ) -> None:
        self.finish(
            result=result,
            error=error,
            reason=EndReason.CANCELLED if closed else EndReason.COMPLETED,
            partial=closed,
        )

    def _capture_chat_response(self, response: ChatCompletion) -> None:
        self._span.set_attribute("response_id", response.id)
        self._span.set_attribute(Semantic.LLM_MODEL_RESPONSE, response.model)
        if response.service_tier is not None:
            self._span.set_attribute("service_tier", response.service_tier)
        finish_reasons = {
            choice.finish_reason for choice in response.choices if choice.finish_reason is not None
        }
        if finish_reasons:
            self._span.set_attribute("finish_reason", ",".join(sorted(finish_reasons)))
        tool_calls = [
            tool_call
            for choice in response.choices
            for tool_call in (choice.message.tool_calls or ())
        ]
        if tool_calls:
            self._span.set_attribute("tool_call_count", len(tool_calls))
            for tool_call in tool_calls:
                self._span.event(
                    "tool.requested",
                    tool_call,
                    semantic_type=Semantic.TOOL_CALL_REQUESTED,
                )
        if response.usage is not None:
            self._set_chat_usage(response.usage)
        self._span.event(
            "messages.output",
            [choice.message for choice in response.choices],
            semantic_type=Semantic.MESSAGE_OUTPUT,
        )
        self._span.set_output(response)

    def _capture_chat_chunk(self, chunk: ChatCompletionChunk) -> None:
        self._span.set_attribute("response_id", chunk.id)
        self._span.set_attribute(Semantic.LLM_MODEL_RESPONSE, chunk.model)
        if chunk.service_tier is not None:
            self._span.set_attribute("service_tier", chunk.service_tier)
        if chunk.usage is not None:
            self._set_chat_usage(chunk.usage)
        for choice in chunk.choices:
            if choice.finish_reason is not None:
                self._stream.finish_reasons.add(choice.finish_reason)
            if choice.delta.content is not None:
                self._stream.text.append(choice.delta.content)
            for tool_call in choice.delta.tool_calls or ():
                key = f"{choice.index}:{tool_call.index}"
                captured = self._stream.tool_calls.setdefault(key, _ToolCallState())
                if tool_call.id is not None:
                    captured.id = tool_call.id
                if tool_call.function is not None:
                    if tool_call.function.name is not None:
                        captured.name = tool_call.function.name
                    if tool_call.function.arguments is not None:
                        captured.arguments += tool_call.function.arguments

    def _capture_response_event(self, event: Any) -> None:
        if isinstance(event, (ResponseCreatedEvent, ResponseInProgressEvent, ResponseQueuedEvent)):
            self._capture_response_metadata(event.response)
        elif isinstance(
            event,
            (
                ResponseCompletedEvent,
                ResponseFailedEvent,
                ResponseIncompleteEvent,
            ),
        ):
            self._capture_response_metadata(event.response)
            if event.response.usage is not None:
                self._set_response_usage(event.response.usage)
        elif isinstance(event, ResponseTextDeltaEvent):
            self._stream.text.append(event.delta)
        elif isinstance(event, ResponseFunctionCallArgumentsDeltaEvent):
            captured = self._stream.tool_calls.setdefault(event.item_id, _ToolCallState())
            captured.arguments += event.delta
        elif isinstance(event, ResponseFunctionCallArgumentsDoneEvent):
            self._stream.tool_calls[event.item_id] = _ToolCallState(
                name=event.name,
                arguments=event.arguments,
            )
        elif isinstance(
            event,
            (ResponseOutputItemAddedEvent, ResponseOutputItemDoneEvent),
        ) and isinstance(event.item, ResponseFunctionToolCall):
            key = event.item.id or event.item.call_id
            self._stream.tool_calls[key] = _ToolCallState(
                id=event.item.id,
                name=event.item.name,
                arguments=event.item.arguments,
            )

    def _capture_response(self, response: Response) -> None:
        self._capture_response_metadata(response)
        if response.usage is not None:
            self._set_response_usage(response.usage)
        tool_calls = [
            item for item in response.output if isinstance(item, ResponseFunctionToolCall)
        ]
        if tool_calls:
            self._span.set_attribute("tool_call_count", len(tool_calls))
            for tool_call in tool_calls:
                self._span.event(
                    "tool.requested",
                    tool_call,
                    semantic_type=Semantic.TOOL_CALL_REQUESTED,
                )
        self._span.event("messages.output", response.output, semantic_type=Semantic.MESSAGE_OUTPUT)
        self._span.set_output(response)

    def _capture_response_metadata(self, response: Response) -> None:
        self._span.set_attribute("response_id", response.id)
        self._span.set_attribute(Semantic.LLM_MODEL_RESPONSE, response.model)
        self._span.set_attribute("response.status", response.status)
        if response.service_tier is not None:
            self._span.set_attribute("service_tier", response.service_tier)

    def _capture_embeddings(self, response: CreateEmbeddingResponse) -> None:
        self._span.set_attribute(Semantic.LLM_MODEL_RESPONSE, response.model)
        self._span.set_attribute("embedding.count", len(response.data))
        self._set_embedding_usage(response.usage)
        self._span.set_output({"embedding_count": len(response.data)})

    def _finish_stream_output(self) -> None:
        if self._stream.finish_reasons:
            self._span.set_attribute("finish_reason", ",".join(sorted(self._stream.finish_reasons)))
        tool_calls = [tool_call.payload() for tool_call in self._stream.tool_calls.values()]
        if tool_calls:
            self._span.set_attribute("tool_call_count", len(tool_calls))
            for tool_call in tool_calls:
                self._span.event(
                    "tool.requested",
                    tool_call,
                    semantic_type=Semantic.TOOL_CALL_REQUESTED,
                )
        output = {"text": "".join(self._stream.text), "tool_calls": tool_calls}
        self._span.event("messages.output", output, semantic_type=Semantic.MESSAGE_OUTPUT)
        self._span.set_output(output)

    def _set_chat_usage(self, usage: CompletionUsage) -> None:
        self._set_usage(usage.prompt_tokens, usage.completion_tokens, usage.total_tokens)
        if usage.prompt_tokens_details is not None:
            if usage.prompt_tokens_details.cached_tokens is not None:
                self._span.set_usage(
                    "cached_input_tokens", usage.prompt_tokens_details.cached_tokens
                )
            if usage.prompt_tokens_details.cache_write_tokens is not None:
                self._span.set_usage(
                    "cache_write_tokens", usage.prompt_tokens_details.cache_write_tokens
                )
        if (
            usage.completion_tokens_details is not None
            and usage.completion_tokens_details.reasoning_tokens is not None
        ):
            self._span.set_usage(
                "reasoning_tokens", usage.completion_tokens_details.reasoning_tokens
            )

    def _set_response_usage(self, usage: ResponseUsage) -> None:
        self._set_usage(usage.input_tokens, usage.output_tokens, usage.total_tokens)
        if usage.input_tokens_details.cached_tokens is not None:
            self._span.set_usage("cached_input_tokens", usage.input_tokens_details.cached_tokens)
        if usage.input_tokens_details.cache_write_tokens is not None:
            self._span.set_usage(
                "cache_write_tokens", usage.input_tokens_details.cache_write_tokens
            )
        if usage.output_tokens_details.reasoning_tokens is not None:
            self._span.set_usage("reasoning_tokens", usage.output_tokens_details.reasoning_tokens)

    def _set_embedding_usage(self, usage: Usage) -> None:
        self._span.set_usage("input_tokens", usage.prompt_tokens)
        self._span.set_usage("total_tokens", usage.total_tokens)

    def _set_usage(self, input_tokens: int, output_tokens: int, total_tokens: int) -> None:
        self._span.set_usage("input_tokens", input_tokens)
        self._span.set_usage("output_tokens", output_tokens)
        self._span.set_usage("total_tokens", total_tokens)


class _EndpointHandler:
    def __init__(
        self,
        endpoint: _Endpoint,
        runtime: InstrumentationRuntime,
        info: InstrumentorInfo,
        raw: _RawRegistry,
    ) -> None:
        self.endpoint = endpoint
        self.runtime = runtime
        self.info = info
        self.raw = raw
        self._scope = runtime.scope(info, target_version=version("openai"))
        self._active: set[_OpenAICall] = set()

    @property
    def suppression_keys(self) -> tuple[str, ...]:
        return self.info.id, "openai"

    def begin(self, call: InstrumentCall) -> CallLifecycle | None:
        run_context = active_run_context()
        if run_context is None:
            return None
        attributes: dict[str, Any] = {
            Semantic.LLM_PROVIDER_NAME: "openai",
            "usage_authority": "provider",
            "streaming": call.kwargs.get("stream") is True,
        }
        requested_model = call.kwargs.get("model")
        if isinstance(requested_model, str):
            attributes[Semantic.LLM_MODEL_REQUESTED] = requested_model
        for key in ("service_tier", "temperature"):
            value = call.kwargs.get(key)
            if isinstance(value, str | int | float):
                attributes[key] = value
        input_payload = {
            key: call.kwargs[key] for key in self.endpoint.input_keys if key in call.kwargs
        }
        span = run_context.span(
            self.endpoint.name,
            kind=SpanKind.LLM,
            input=input_payload,
            attributes=attributes,
            instrumentation_scope=self._scope,
        )
        span.__enter__()
        span.set_attribute("abp.logical_operation_id", span.id)
        if input_payload:
            span.event("messages.input", input_payload, semantic_type=Semantic.MESSAGE_INPUT)
        active = _OpenAICall(self, call, span)
        self._active.add(active)
        return active

    def diagnose(self, stage: str, error: Exception) -> None:
        self.runtime.diagnose(
            self.info,
            "openai_instrumentation_error",
            f"{stage}: {type(error).__name__}: {error}",
        )

    def discard(self, call: _OpenAICall) -> None:
        self._active.discard(call)

    def close(self) -> None:
        for call in tuple(self._active):
            try:
                call.finish(reason=EndReason.ABANDONED, partial=True)
            except Exception as error:
                self.diagnose("close", error)


class _RawLifecycle:
    def __init__(self, call: _OpenAICall, action: RawAction) -> None:
        self._call = call
        self._action: RawAction = action

    def resume(self) -> None:
        self._call.resume()

    def suspend(self) -> None:
        self._call.suspend()

    def observe(self, item: Any) -> None:
        del item

    def finish(
        self,
        *,
        result: Any = None,
        error: BaseException | None = None,
        reason: EndReason = EndReason.COMPLETED,
        partial: bool = False,
    ) -> None:
        del reason, partial
        self._call.finish_raw(result=result, error=error, closed=self._action == "close")


class _RawHandler:
    def __init__(
        self,
        action: RawAction,
        registry: _RawRegistry,
        runtime: InstrumentationRuntime,
        info: InstrumentorInfo,
    ) -> None:
        self._action: RawAction = action
        self._registry = registry
        self._runtime = runtime
        self._info = info

    @property
    def suppression_keys(self) -> tuple[str, ...]:
        return self._info.id, "openai"

    def begin(self, call: InstrumentCall) -> CallLifecycle | None:
        response = call.instance
        if not isinstance(response, (APIResponse, AsyncAPIResponse, LegacyAPIResponse)):
            return None
        parent = self._registry.take(response)
        return None if parent is None else _RawLifecycle(parent, self._action)

    def diagnose(self, stage: str, error: Exception) -> None:
        self._runtime.diagnose(
            self._info,
            "openai_raw_response_error",
            f"{stage}: {type(error).__name__}: {error}",
        )

    def close(self) -> None:
        return None


class _HelperStreamHandler:
    def __init__(self, runtime: InstrumentationRuntime, info: InstrumentorInfo) -> None:
        self._runtime = runtime
        self._info = info

    @property
    def suppression_keys(self) -> tuple[str, ...]:
        return self._info.id, "openai"

    def begin(self, call: InstrumentCall) -> None:
        stream = call.instance
        if isinstance(
            stream,
            (
                ChatCompletionStream,
                AsyncChatCompletionStream,
                ResponseStream,
                AsyncResponseStream,
            ),
        ):
            raw_stream = stream._raw_stream  # pyright: ignore[reportPrivateUsage]
        else:
            return None
        if isinstance(raw_stream, Cancelable):
            raw_stream.cancel()
        return None

    def diagnose(self, stage: str, error: Exception) -> None:
        self._runtime.diagnose(
            self._info,
            "openai_helper_stream_error",
            f"{stage}: {type(error).__name__}: {error}",
        )

    def close(self) -> None:
        return None


class OpenAIClient:
    """Install provider-level ABP capture for the official OpenAI Python SDK."""

    def __init__(self) -> None:
        self._info = InstrumentorInfo(
            id="autobench.openai",
            version=__version__,
            target_distribution="openai",
            supported_versions=">=2.52,<2.53",
            mechanism=CaptureMechanism.PATCH,
            layer=AbstractionLayer.CLIENT,
            span_kinds=("llm",),
            semantic_families=("llm", "message", "tool", "stream"),
            source_convention="openai-python",
            source_convention_version="2.52",
            capabilities=InstrumentorCapabilities(sync=True, async_=True, streaming=True),
        )

    @property
    def info(self) -> InstrumentorInfo:
        return self._info

    def check(self) -> Compatibility:
        if find_spec("openai") is None:
            return Compatibility(
                status=CompatibilityStatus.UNAVAILABLE,
                diagnostics=("OpenAI is unavailable; install Autobench with the 'openai' extra",),
            )
        required: tuple[tuple[type[Any], str], ...] = (
            (Completions, "create"),
            (Completions, "parse"),
            (AsyncCompletions, "create"),
            (AsyncCompletions, "parse"),
            (Responses, "create"),
            (Responses, "parse"),
            (AsyncResponses, "create"),
            (AsyncResponses, "parse"),
            (Embeddings, "create"),
            (AsyncEmbeddings, "create"),
            (APIResponse, "parse"),
            (APIResponse, "close"),
            (AsyncAPIResponse, "parse"),
            (AsyncAPIResponse, "close"),
            (LegacyAPIResponse, "parse"),
            (ChatCompletionStream, "close"),
            (AsyncChatCompletionStream, "close"),
            (ResponseStream, "close"),
            (AsyncResponseStream, "close"),
        )
        missing = tuple(
            f"{target.__name__}.{name}" for target, name in required if name not in dir(target)
        )
        if missing:
            return Compatibility(
                status=CompatibilityStatus.UNSUPPORTED,
                diagnostics=(f"OpenAI SDK lacks required lifecycle seams: {', '.join(missing)}",),
                private_seam_supported=False,
            )
        return Compatibility(target_version=version("openai"), private_seam_supported=True)

    def install(self, runtime: InstrumentationRuntime) -> InstrumentationHandle:
        raw = _RawRegistry()
        endpoints = {
            "chat": _Endpoint(
                name="openai.chat.completions",
                input_keys=("messages", "tools", "response_format"),
            ),
            "responses": _Endpoint(
                name="openai.responses",
                input_keys=("input", "instructions", "tools", "text"),
            ),
            "embeddings": _Endpoint(
                name="openai.embeddings",
                input_keys=("input",),
            ),
        }
        handles: list[InstrumentationHandle] = []

        def patch(
            target: type[Any],
            method: str,
            handler: _EndpointHandler | _RawHandler | _HelperStreamHandler,
        ) -> None:
            handles.append(runtime.patch_method(self.info, target, method, handler))

        try:
            for target in (Completions, AsyncCompletions):
                patch(
                    target, "create", _EndpointHandler(endpoints["chat"], runtime, self.info, raw)
                )
                patch(target, "parse", _EndpointHandler(endpoints["chat"], runtime, self.info, raw))
            for target in (Responses, AsyncResponses):
                patch(
                    target,
                    "create",
                    _EndpointHandler(endpoints["responses"], runtime, self.info, raw),
                )
                patch(
                    target,
                    "parse",
                    _EndpointHandler(endpoints["responses"], runtime, self.info, raw),
                )
            for target in (Embeddings, AsyncEmbeddings):
                patch(
                    target,
                    "create",
                    _EndpointHandler(endpoints["embeddings"], runtime, self.info, raw),
                )
            patch(APIResponse, "parse", _RawHandler("parse", raw, runtime, self.info))
            patch(APIResponse, "close", _RawHandler("close", raw, runtime, self.info))
            patch(AsyncAPIResponse, "parse", _RawHandler("parse", raw, runtime, self.info))
            patch(AsyncAPIResponse, "close", _RawHandler("close", raw, runtime, self.info))
            patch(LegacyAPIResponse, "parse", _RawHandler("parse", raw, runtime, self.info))
            for target in (
                ChatCompletionStream,
                AsyncChatCompletionStream,
                ResponseStream,
                AsyncResponseStream,
            ):
                patch(target, "close", _HelperStreamHandler(runtime, self.info))
        except BaseException:
            for handle in reversed(handles):
                handle.close()
            raise

        def close() -> None:
            for handle in reversed(handles):
                handle.close()

        return InstrumentationHandle(close, info=self.info)


__all__ = ("OpenAIClient",)
