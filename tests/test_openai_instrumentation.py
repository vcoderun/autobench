from __future__ import annotations

import asyncio
import json
from collections.abc import Iterator
from contextlib import contextmanager
from inspect import getattr_static
from typing import Any

import httpx
import pytest
from openai import AsyncOpenAI, InternalServerError, OpenAI
from pydantic import BaseModel

from autobench import Case, InstrumentationManager, RunContext, Variant
from autobench.instrumentation.manager import InstrumentationRuntime
from autobench.instrumentation.models import InstrumentCall
from autobench.instrumentation.openai import OpenAIClient
from autobench.instrumentation.openai import client as openai_instrumentation
from autobench.protocol.signals import EndReason
from autobench.protocol.traces import SpanRecord
from autobench.runtime.context import Span
from autobench.runtime.context import SpanRecord as RuntimeSpanRecord
from autobench.runtime.instrumentation import reset_active_run_context, set_active_run_context


class _ParsedAnswer(BaseModel):
    answer: str


def _chat_response() -> dict[str, Any]:
    return {
        "id": "chatcmpl-test",
        "object": "chat.completion",
        "created": 1,
        "model": "gpt-test-2026",
        "service_tier": "default",
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "done",
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "type": "function",
                            "function": {"name": "lookup", "arguments": '{"id":1}'},
                        }
                    ],
                },
                "finish_reason": "tool_calls",
            }
        ],
        "usage": {
            "prompt_tokens": 10,
            "completion_tokens": 4,
            "total_tokens": 14,
            "prompt_tokens_details": {"cached_tokens": 3, "cache_write_tokens": 2},
            "completion_tokens_details": {"reasoning_tokens": 1},
        },
    }


def _response() -> dict[str, Any]:
    return {
        "id": "resp_test",
        "object": "response",
        "created_at": 1,
        "status": "completed",
        "error": None,
        "incomplete_details": None,
        "instructions": None,
        "max_output_tokens": None,
        "model": "gpt-test-2026",
        "output": [
            {
                "type": "function_call",
                "id": "fc_1",
                "call_id": "call_1",
                "name": "lookup",
                "arguments": '{"id":1}',
                "status": "completed",
            }
        ],
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
            "input_tokens": 12,
            "input_tokens_details": {"cached_tokens": 4, "cache_write_tokens": 2},
            "output_tokens": 6,
            "output_tokens_details": {"reasoning_tokens": 2},
            "total_tokens": 18,
        },
        "service_tier": "default",
    }


def _embedding_response() -> dict[str, Any]:
    return {
        "object": "list",
        "model": "embedding-test-2026",
        "data": [
            {"object": "embedding", "index": 0, "embedding": [0.1, 0.2]},
            {"object": "embedding", "index": 1, "embedding": [0.3, 0.4]},
        ],
        "usage": {"prompt_tokens": 5, "total_tokens": 5},
    }


def _sync_transport(*, fail: bool = False) -> httpx.MockTransport:
    def handler(request: httpx.Request) -> httpx.Response:
        if fail:
            return httpx.Response(500, json={"error": {"message": "provider failed"}})
        path = request.url.path
        if path.endswith("/chat/completions"):
            payload = json.loads(request.content)
            if payload.get("stream") is True:
                chunks = (
                    {
                        "id": "chatcmpl-stream",
                        "object": "chat.completion.chunk",
                        "created": 1,
                        "model": "gpt-test-2026",
                        "choices": [
                            {"index": 0, "delta": {"content": "hello "}, "finish_reason": None}
                        ],
                    },
                    {
                        "id": "chatcmpl-stream",
                        "object": "chat.completion.chunk",
                        "created": 1,
                        "model": "gpt-test-2026",
                        "choices": [
                            {
                                "index": 0,
                                "delta": {
                                    "tool_calls": [
                                        {
                                            "index": 0,
                                            "id": "call_stream",
                                            "type": "function",
                                            "function": {
                                                "name": "lookup",
                                                "arguments": '{"id":',
                                            },
                                        }
                                    ]
                                },
                                "finish_reason": None,
                            }
                        ],
                    },
                    {
                        "id": "chatcmpl-stream",
                        "object": "chat.completion.chunk",
                        "created": 1,
                        "model": "gpt-test-2026",
                        "service_tier": "default",
                        "choices": [
                            {
                                "index": 0,
                                "delta": {
                                    "content": "world",
                                    "tool_calls": [
                                        {
                                            "index": 0,
                                            "function": {"arguments": "1}"},
                                        }
                                    ],
                                },
                                "finish_reason": "tool_calls",
                            }
                        ],
                        "usage": {
                            "prompt_tokens": 8,
                            "completion_tokens": 3,
                            "total_tokens": 11,
                        },
                    },
                )
                body = "".join(f"data: {json.dumps(chunk)}\n\n" for chunk in chunks)
                return httpx.Response(
                    200,
                    text=f"{body}data: [DONE]\n\n",
                    headers={"content-type": "text/event-stream"},
                )
            response = _chat_response()
            if "response_format" in payload:
                response["choices"][0]["message"] = {
                    "role": "assistant",
                    "content": '{"answer":"done"}',
                }
            return httpx.Response(200, json=response)
        if path.endswith("/responses"):
            payload = json.loads(request.content)
            if payload.get("stream") is True:
                message = {
                    "type": "message",
                    "id": "msg_1",
                    "status": "completed",
                    "role": "assistant",
                    "content": [
                        {
                            "type": "output_text",
                            "text": "done",
                            "annotations": [],
                            "logprobs": [],
                        }
                    ],
                }
                function_call = {
                    "type": "function_call",
                    "id": "fc_stream",
                    "call_id": "call_stream",
                    "name": "lookup",
                    "arguments": '{"id":1}',
                    "status": "completed",
                }
                created_response = _response()
                created_response["status"] = "in_progress"
                created_response["output"] = []
                created_response["usage"] = None
                completed_response = _response()
                completed_response["output"] = [message, function_call]
                events = (
                    {
                        "type": "response.created",
                        "sequence_number": 0,
                        "response": created_response,
                    },
                    {
                        "type": "response.output_item.added",
                        "sequence_number": 1,
                        "output_index": 0,
                        "item": {
                            "type": "message",
                            "id": "msg_1",
                            "status": "in_progress",
                            "role": "assistant",
                            "content": [],
                        },
                    },
                    {
                        "type": "response.content_part.added",
                        "sequence_number": 2,
                        "item_id": "msg_1",
                        "output_index": 0,
                        "content_index": 0,
                        "part": {
                            "type": "output_text",
                            "text": "",
                            "annotations": [],
                            "logprobs": [],
                        },
                    },
                    {
                        "type": "response.output_text.delta",
                        "sequence_number": 3,
                        "item_id": "msg_1",
                        "output_index": 0,
                        "content_index": 0,
                        "delta": "done",
                        "logprobs": [],
                    },
                    {
                        "type": "response.output_text.done",
                        "sequence_number": 4,
                        "item_id": "msg_1",
                        "output_index": 0,
                        "content_index": 0,
                        "text": "done",
                        "logprobs": [],
                    },
                    {
                        "type": "response.content_part.done",
                        "sequence_number": 5,
                        "item_id": "msg_1",
                        "output_index": 0,
                        "content_index": 0,
                        "part": message["content"][0],
                    },
                    {
                        "type": "response.output_item.done",
                        "sequence_number": 6,
                        "output_index": 0,
                        "item": message,
                    },
                    {
                        "type": "response.output_item.added",
                        "sequence_number": 7,
                        "output_index": 1,
                        "item": function_call,
                    },
                    {
                        "type": "response.function_call_arguments.delta",
                        "sequence_number": 8,
                        "item_id": "fc_stream",
                        "output_index": 1,
                        "delta": '{"id":',
                    },
                    {
                        "type": "response.function_call_arguments.done",
                        "sequence_number": 9,
                        "item_id": "fc_stream",
                        "output_index": 1,
                        "name": "lookup",
                        "arguments": '{"id":1}',
                    },
                    {
                        "type": "response.output_item.done",
                        "sequence_number": 10,
                        "output_index": 1,
                        "item": function_call,
                    },
                    {
                        "type": "response.completed",
                        "sequence_number": 11,
                        "response": completed_response,
                    },
                )
                body = "".join(f"data: {json.dumps(event)}\n\n" for event in events)
                return httpx.Response(
                    200,
                    text=body,
                    headers={"content-type": "text/event-stream"},
                )
            return httpx.Response(200, json=_response())
        if path.endswith("/embeddings"):
            return httpx.Response(200, json=_embedding_response())
        raise AssertionError(f"unexpected path: {path}")

    return httpx.MockTransport(handler)


def _async_transport() -> httpx.MockTransport:
    transport = _sync_transport()

    async def handler(request: httpx.Request) -> httpx.Response:
        return transport.handle_request(request)

    return httpx.MockTransport(handler)


def _sparse_transport() -> httpx.MockTransport:
    def handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content)
        if request.url.path.endswith("/chat/completions"):
            usage = None
            if payload["messages"][0]["content"] == "details":
                usage = {
                    "prompt_tokens": 1,
                    "completion_tokens": 1,
                    "total_tokens": 2,
                    "prompt_tokens_details": {
                        "cached_tokens": None,
                        "cache_write_tokens": None,
                    },
                    "completion_tokens_details": {"reasoning_tokens": None},
                }
            return httpx.Response(
                200,
                json={
                    "id": "chatcmpl-sparse",
                    "object": "chat.completion",
                    "created": 1,
                    "model": "gpt-sparse",
                    "choices": [],
                    "usage": usage,
                },
            )
        response = _response()
        response["service_tier"] = None
        response["output"] = []
        if "input" in payload:
            response["usage"] = {
                "input_tokens": 1,
                "input_tokens_details": {
                    "cached_tokens": None,
                    "cache_write_tokens": None,
                },
                "output_tokens": 1,
                "output_tokens_details": {"reasoning_tokens": None},
                "total_tokens": 2,
            }
        else:
            response["usage"] = None
        return httpx.Response(200, json=response)

    return httpx.MockTransport(handler)


def _run_context() -> RunContext:
    return RunContext(
        benchmark_id="openai",
        case=Case(id="case", input={"prompt": "hello"}),
        variant=Variant(id="variant"),
    )


@contextmanager
def _instrument(ctx: RunContext) -> Iterator[None]:
    manager = InstrumentationManager()
    handle = manager.install(OpenAIClient())
    token = set_active_run_context(ctx)
    try:
        yield
    finally:
        reset_active_run_context(token)
        handle.close()
        manager.close()


def _operation_spans(ctx: RunContext, operation: str) -> list[SpanRecord]:
    return [span for span in ctx.trace.spans if span.operation == operation]


def _runtime_spans(ctx: RunContext, operation: str) -> list[RuntimeSpanRecord]:
    return [span for span in ctx.spans if span.name == operation]


def test_openai_client_check_and_sync_chat_capture_provider_evidence() -> None:
    instrumentor = OpenAIClient()
    assert instrumentor.check().installable is True
    assert instrumentor.info.capabilities.streaming is True
    ctx = _run_context()
    client = OpenAI(
        api_key="test",
        base_url="https://openai.test/v1",
        http_client=httpx.Client(transport=_sync_transport()),
    )

    with _instrument(ctx):
        result = client.chat.completions.create(
            model="gpt-test",
            messages=[{"role": "user", "content": "hello"}],
            tools=[
                {
                    "type": "function",
                    "function": {
                        "name": "lookup",
                        "description": "Lookup a record",
                        "parameters": {"type": "object", "properties": {}},
                    },
                }
            ],
            temperature=0.2,
            service_tier="default",
        )

    assert result.id == "chatcmpl-test"
    span = _operation_spans(ctx, "openai.chat.completions")[0]
    assert span.attributes["llm.model.requested"] == "gpt-test"
    assert span.attributes["llm.model.response"] == "gpt-test-2026"
    assert span.attributes["tool_call_count"] == 1
    assert span.usage == {
        "input_tokens": 10,
        "output_tokens": 4,
        "total_tokens": 14,
        "cached_input_tokens": 3,
        "cache_write_tokens": 2,
        "reasoning_tokens": 1,
    }

    parse_ctx = _run_context()
    with _instrument(parse_ctx):
        parsed = client.chat.completions.parse(
            model="gpt-test",
            messages=[{"role": "user", "content": "return JSON"}],
            response_format=_ParsedAnswer,
        )
    assert parsed.choices[0].message.parsed == _ParsedAnswer(answer="done")


async def test_openai_client_async_chat_preserves_decorated_coroutine_behavior() -> None:
    ctx = _run_context()
    client = AsyncOpenAI(
        api_key="test",
        base_url="https://openai.test/v1",
        http_client=httpx.AsyncClient(transport=_async_transport()),
    )
    with _instrument(ctx):
        result = await client.chat.completions.create(
            model="gpt-test",
            messages=[{"role": "user", "content": "hello"}],
        )
    await client.close()

    assert result.id == "chatcmpl-test"
    assert _operation_spans(ctx, "openai.chat.completions")[0].end_reason is EndReason.COMPLETED


async def test_openai_async_chat_stream_and_raw_response_lifecycles() -> None:
    ctx = _run_context()
    client = AsyncOpenAI(
        api_key="test",
        base_url="https://openai.test/v1",
        http_client=httpx.AsyncClient(transport=_async_transport()),
    )
    with _instrument(ctx):
        stream = await client.chat.completions.create(
            model="gpt-test",
            messages=[{"role": "user", "content": "hello"}],
            stream=True,
            stream_options={"include_usage": True},
        )
        chunks = [chunk async for chunk in stream]

        raw = await client.chat.completions.with_raw_response.create(
            model="gpt-test",
            messages=[{"role": "user", "content": "hello"}],
        )
        parsed_legacy = raw.parse()

        async with client.chat.completions.with_streaming_response.create(
            model="gpt-test",
            messages=[{"role": "user", "content": "hello"}],
        ) as streaming_raw:
            parsed_streaming = await streaming_raw.parse()

        async with client.chat.completions.with_streaming_response.create(
            model="gpt-test",
            messages=[{"role": "user", "content": "hello"}],
        ):
            pass
    await client.close()

    assert len(chunks) == 3
    assert isinstance(parsed_legacy, openai_instrumentation.ChatCompletion)
    assert isinstance(parsed_streaming, openai_instrumentation.ChatCompletion)
    assert parsed_legacy.id == "chatcmpl-test"
    assert parsed_streaming.id == "chatcmpl-test"
    spans = _operation_spans(ctx, "openai.chat.completions")
    assert [span.end_reason for span in spans] == [
        EndReason.COMPLETED,
        EndReason.COMPLETED,
        EndReason.COMPLETED,
        EndReason.CANCELLED,
    ]


def test_openai_chat_stream_accumulates_output_tool_arguments_and_usage() -> None:
    ctx = _run_context()
    client = OpenAI(
        api_key="test",
        base_url="https://openai.test/v1",
        http_client=httpx.Client(transport=_sync_transport()),
    )

    with _instrument(ctx):
        stream = client.chat.completions.create(
            model="gpt-test",
            messages=[{"role": "user", "content": "hello"}],
            stream=True,
            stream_options={"include_usage": True},
        )
        chunks = list(stream)

    assert len(chunks) == 3
    span = _operation_spans(ctx, "openai.chat.completions")[0]
    assert span.attributes["stream.chunk_count"] == 3
    assert span.attributes["finish_reason"] == "tool_calls"
    assert _runtime_spans(ctx, "openai.chat.completions")[0].output == {
        "text": "hello world",
        "tool_calls": [{"id": "call_stream", "name": "lookup", "arguments": '{"id":1}'}],
    }
    assert span.usage["total_tokens"] == 11
    assert span.end_reason is EndReason.COMPLETED


def test_openai_chat_stream_manager_and_early_close_finalize_once() -> None:
    ctx = _run_context()
    client = OpenAI(
        api_key="test",
        base_url="https://openai.test/v1",
        http_client=httpx.Client(transport=_sync_transport()),
    )

    with (
        _instrument(ctx),
        client.chat.completions.stream(
            model="gpt-test",
            messages=[{"role": "user", "content": "hello"}],
        ) as stream,
    ):
        first = next(stream)
        assert first.type == "chunk"

    span = _operation_spans(ctx, "openai.chat.completions")[0]
    assert span.attributes["stream.chunk_count"] == 1
    assert span.end_reason is EndReason.CANCELLED
    assert span.partial is True


def test_openai_responses_and_embeddings_sync_capture_distinct_shapes() -> None:
    ctx = _run_context()
    client = OpenAI(
        api_key="test",
        base_url="https://openai.test/v1",
        http_client=httpx.Client(transport=_sync_transport()),
    )

    with _instrument(ctx):
        response = client.responses.create(model="gpt-test", input="hello")
        embedding = client.embeddings.create(
            model="embedding-test",
            input=["hello", "world"],
        )

    assert response.id == "resp_test"
    assert len(embedding.data) == 2
    response_span = _operation_spans(ctx, "openai.responses")[0]
    assert response_span.attributes["response.status"] == "completed"
    assert response_span.attributes["tool_call_count"] == 1
    assert response_span.usage["cached_input_tokens"] == 4
    assert response_span.usage["cache_write_tokens"] == 2
    assert response_span.usage["reasoning_tokens"] == 2
    embedding_span = _operation_spans(ctx, "openai.embeddings")[0]
    assert embedding_span.attributes["embedding.count"] == 2
    assert _runtime_spans(ctx, "openai.embeddings")[0].output == {"embedding_count": 2}
    assert embedding_span.usage == {"input_tokens": 5, "total_tokens": 5}


def test_openai_responses_stream_manager_captures_text_tools_and_terminal_usage() -> None:
    ctx = _run_context()
    client = OpenAI(
        api_key="test",
        base_url="https://openai.test/v1",
        http_client=httpx.Client(transport=_sync_transport()),
    )

    with _instrument(ctx), client.responses.stream(model="gpt-test", input="hello") as stream:
        events = list(stream)

    assert len(events) == 12
    span = _operation_spans(ctx, "openai.responses")[0]
    assert span.attributes["stream.chunk_count"] == 12
    assert span.usage["total_tokens"] == 18
    assert _runtime_spans(ctx, "openai.responses")[0].output == {
        "text": "done",
        "tool_calls": [
            {
                "id": "fc_stream",
                "name": "lookup",
                "arguments": '{"id":1}',
            }
        ],
    }


async def test_openai_responses_and_embeddings_async_capture() -> None:
    ctx = _run_context()
    client = AsyncOpenAI(
        api_key="test",
        base_url="https://openai.test/v1",
        http_client=httpx.AsyncClient(transport=_async_transport()),
    )

    with _instrument(ctx):
        response = await client.responses.create(model="gpt-test", input="hello")
        embedding = await client.embeddings.create(model="embedding-test", input="hello")
    await client.close()

    assert response.id == "resp_test"
    assert len(embedding.data) == 2
    assert len(_operation_spans(ctx, "openai.responses")) == 1
    assert len(_operation_spans(ctx, "openai.embeddings")) == 1


async def test_openai_async_responses_stream_manager_closes_partial_work() -> None:
    ctx = _run_context()
    client = AsyncOpenAI(
        api_key="test",
        base_url="https://openai.test/v1",
        http_client=httpx.AsyncClient(transport=_async_transport()),
    )

    with _instrument(ctx):
        async with client.responses.stream(model="gpt-test", input="hello") as stream:
            first = await anext(stream)
            assert first.type == "response.created"
    await client.close()

    span = _operation_spans(ctx, "openai.responses")[0]
    assert span.end_reason is EndReason.CANCELLED
    assert span.partial is True


def test_openai_raw_response_parse_and_unparsed_close_have_correct_lifecycle() -> None:
    ctx = _run_context()
    client = OpenAI(
        api_key="test",
        base_url="https://openai.test/v1",
        http_client=httpx.Client(transport=_sync_transport()),
    )

    with _instrument(ctx):
        raw = client.chat.completions.with_raw_response.create(
            model="gpt-test",
            messages=[{"role": "user", "content": "hello"}],
        )
        parsed = raw.parse()
        with client.chat.completions.with_streaming_response.create(
            model="gpt-test",
            messages=[{"role": "user", "content": "hello"}],
        ):
            pass

    assert isinstance(parsed, openai_instrumentation.ChatCompletion)
    assert parsed.id == "chatcmpl-test"
    spans = _operation_spans(ctx, "openai.chat.completions")
    assert spans[0].end_reason is EndReason.COMPLETED
    assert spans[0].usage["total_tokens"] == 14
    assert spans[1].end_reason is EndReason.CANCELLED
    assert spans[1].partial is True


def test_openai_provider_error_is_unchanged_and_recorded() -> None:
    ctx = _run_context()
    client = OpenAI(
        api_key="test",
        base_url="https://openai.test/v1",
        max_retries=0,
        http_client=httpx.Client(transport=_sync_transport(fail=True)),
    )

    with pytest.raises(InternalServerError) as caught, _instrument(ctx):
        client.chat.completions.create(
            model="gpt-test",
            messages=[{"role": "user", "content": "hello"}],
        )

    assert caught.value.status_code == 500
    span = _operation_spans(ctx, "openai.chat.completions")[0]
    assert span.end_reason is EndReason.FAILED
    assert span.errors


def test_openai_sparse_provider_payloads_remain_valid_evidence() -> None:
    ctx = _run_context()
    client = OpenAI(
        api_key="test",
        base_url="https://openai.test/v1",
        http_client=httpx.Client(transport=_sparse_transport()),
    )

    with _instrument(ctx):
        without_usage = client.chat.completions.create(
            model="gpt-sparse",
            messages=[{"role": "user", "content": "none"}],
        )
        with_empty_details = client.chat.completions.create(
            model="gpt-sparse",
            messages=[{"role": "user", "content": "details"}],
        )
        response_without_input = client.responses.create()
        response_with_details = client.responses.create(input="details")

    assert without_usage.usage is None
    assert with_empty_details.usage is not None
    assert response_without_input.usage is None
    assert response_with_details.usage is not None
    spans = _runtime_spans(ctx, "openai.chat.completions")
    assert spans[0].usage == {}
    assert spans[1].usage == {"input_tokens": 1, "output_tokens": 1, "total_tokens": 2}
    response_spans = _runtime_spans(ctx, "openai.responses")
    assert response_spans[0].usage == {}
    assert response_spans[1].usage == {
        "input_tokens": 1,
        "output_tokens": 1,
        "total_tokens": 2,
    }


def test_openai_callback_failures_do_not_change_application_results(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ctx = _run_context()
    runtime = InstrumentationRuntime()
    instrumentor = OpenAIClient()
    raw = openai_instrumentation._RawRegistry()
    handler = openai_instrumentation._EndpointHandler(
        openai_instrumentation._Endpoint(name="test.openai", input_keys=()),
        runtime,
        instrumentor.info,
        raw,
    )
    token = set_active_run_context(ctx)
    try:
        lifecycle = handler.begin(InstrumentCall(instance=None, args=(), kwargs={}))
        assert isinstance(lifecycle, openai_instrumentation._OpenAICall)
        lifecycle.observe(
            openai_instrumentation.ResponseTextDeltaEvent.model_validate(
                {
                    "type": "response.output_text.delta",
                    "sequence_number": 1,
                    "item_id": "msg",
                    "output_index": 0,
                    "content_index": 0,
                    "delta": "partial",
                    "logprobs": [],
                }
            )
        )
        lifecycle.observe(
            openai_instrumentation.ChatCompletionChunk.model_validate(
                {
                    "id": "chatcmpl-partial-tool",
                    "object": "chat.completion.chunk",
                    "created": 1,
                    "model": "gpt-test-2026",
                    "choices": [
                        {
                            "index": 0,
                            "delta": {
                                "tool_calls": [
                                    {"index": 0, "id": "call_without_function"},
                                    {"index": 1, "function": {}},
                                ]
                            },
                            "finish_reason": None,
                        }
                    ],
                }
            )
        )
        completed_without_usage = _response()
        completed_without_usage["usage"] = None
        lifecycle.observe(
            openai_instrumentation.ResponseCompletedEvent.model_validate(
                {
                    "type": "response.completed",
                    "sequence_number": 2,
                    "response": completed_without_usage,
                }
            )
        )
        lifecycle.finish(error=asyncio.CancelledError())
        lifecycle.finish()

        raw_parent = handler.begin(InstrumentCall(instance=None, args=(), kwargs={}))
        assert isinstance(raw_parent, openai_instrumentation._OpenAICall)
        raw_lifecycle = openai_instrumentation._RawLifecycle(raw_parent, "parse")
        raw_lifecycle.suspend()
        raw_lifecycle.resume()
        raw_lifecycle.observe("ignored")
        raw_lifecycle.finish(
            result=openai_instrumentation.ChatCompletion.model_validate(_chat_response())
        )

        failing = handler.begin(InstrumentCall(instance=None, args=(), kwargs={}))
        assert isinstance(failing, openai_instrumentation._OpenAICall)
        original_finish = Span.finish

        def fail_finish(
            self: Span,
            *,
            error: BaseException | None = None,
            reason: EndReason | None = None,
            partial: bool | None = None,
        ) -> None:
            del self, error, reason, partial
            raise RuntimeError("capture failed")

        monkeypatch.setattr(Span, "finish", fail_finish)
        handler.close()
        monkeypatch.setattr(Span, "finish", original_finish)

        raw_handler = openai_instrumentation._RawHandler("parse", raw, runtime, instrumentor.info)
        assert raw_handler.begin(InstrumentCall(instance=None, args=(), kwargs={})) is None
        raw_handler.diagnose("test", RuntimeError("raw callback"))
        raw_handler.close()

        helper = openai_instrumentation._HelperStreamHandler(runtime, instrumentor.info)
        assert helper.begin(InstrumentCall(instance=None, args=(), kwargs={})) is None
        helper.diagnose("test", RuntimeError("helper callback"))
        helper.close()
        handler.diagnose("test", RuntimeError("endpoint callback"))

        assert openai_instrumentation._ToolCallState(arguments="{}").payload() == {
            "arguments": "{}"
        }
        assert openai_instrumentation._ToolCallState(
            id="call",
            name="lookup",
            arguments="{}",
        ).payload() == {"id": "call", "name": "lookup", "arguments": "{}"}
    finally:
        reset_active_run_context(token)
        ctx.finalize()


def test_openai_compatibility_and_install_rollback_are_non_destructive(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    instrumentor = OpenAIClient()
    with monkeypatch.context() as patch:
        patch.setattr(openai_instrumentation, "find_spec", lambda name: None)
        assert instrumentor.check().available is False

    with monkeypatch.context() as patch:
        patch.delattr(openai_instrumentation.LegacyAPIResponse, "parse")
        compatibility = instrumentor.check()
        assert compatibility.supported is False
        assert compatibility.private_seam_supported is False

    original_create = getattr_static(openai_instrumentation.Completions, "create")
    with monkeypatch.context() as patch:
        patch.delattr(openai_instrumentation.AsyncResponseStream, "close")
        with pytest.raises(AttributeError):
            instrumentor.install(InstrumentationRuntime())
    assert getattr_static(openai_instrumentation.Completions, "create") is original_create


def test_openai_calls_outside_an_active_run_are_untouched() -> None:
    manager = InstrumentationManager()
    handle = manager.install(OpenAIClient())
    client = OpenAI(
        api_key="test",
        base_url="https://openai.test/v1",
        http_client=httpx.Client(transport=_sync_transport()),
    )
    try:
        result = client.chat.completions.create(
            model="gpt-test",
            messages=[{"role": "user", "content": "hello"}],
        )
        with client.chat.completions.stream(
            model="gpt-test",
            messages=[{"role": "user", "content": "hello"}],
        ) as stream:
            next(stream)
    finally:
        handle.close()
        manager.close()

    assert result.id == "chatcmpl-test"
