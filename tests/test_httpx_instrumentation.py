from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Iterator
from contextlib import contextmanager
from inspect import getattr_static
from typing import Any, cast

import httpx
import pytest
from openai import AsyncOpenAI
from pydantic import ValidationError
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider

from autobench import Case, InstrumentationManager, RunContext, Variant
from autobench.instrumentation.httpx import HTTPX, HTTPXCapture
from autobench.instrumentation.httpx import client as httpx_instrumentation
from autobench.instrumentation.manager import InstrumentationRuntime
from autobench.instrumentation.models import Instrumentor
from autobench.instrumentation.openai import OpenAIClient
from autobench.instrumentation.pydantic_ai import PydanticAI
from autobench.protocol.capture import CapturePolicy
from autobench.protocol.context import suppress_instrumentation
from autobench.protocol.signals import EndReason
from autobench.protocol.traces import SpanRecord
from autobench.runtime.instrumentation import reset_active_run_context, set_active_run_context


class SyncChunks(httpx.SyncByteStream):
    def __init__(
        self,
        chunks: tuple[bytes, ...],
        *,
        fail: bool = False,
        close_error: bool = False,
    ) -> None:
        self.chunks = chunks
        self.fail = fail
        self.close_error = close_error
        self.closed = False

    def __iter__(self) -> Iterator[bytes]:
        yield from self.chunks
        if self.fail:
            raise httpx.ReadError("stream failed")

    def close(self) -> None:
        self.closed = True
        if self.close_error:
            raise httpx.ReadError("sync close failed")


class AsyncChunks(httpx.AsyncByteStream):
    def __init__(
        self,
        chunks: tuple[bytes, ...],
        *,
        fail: bool = False,
        close_error: bool = False,
    ) -> None:
        self.chunks = chunks
        self.fail = fail
        self.close_error = close_error
        self.closed = False

    async def __aiter__(self) -> AsyncIterator[bytes]:
        for chunk in self.chunks:
            await asyncio.sleep(0)
            yield chunk
        if self.fail:
            raise httpx.ReadError("async stream failed")

    async def aclose(self) -> None:
        self.closed = True
        if self.close_error:
            raise httpx.ReadError("close failed")


class InvalidSyncChunks(httpx.SyncByteStream):
    def __iter__(self) -> Iterator[bytes]:
        yield cast(bytes, "invalid")


class CustomTransport(httpx.BaseTransport):
    def handle_request(self, request: httpx.Request) -> httpx.Response:
        return httpx.Response(204, request=request, extensions={"http_version": b"HTTP/2"})


def run_context(*, full: bool = False) -> RunContext:
    return RunContext(
        benchmark_id="httpx",
        case=Case(id="case", input={"url": "https://service.test"}),
        variant=Variant(id="variant"),
        capture_policy=CapturePolicy.full() if full else None,
    )


@contextmanager
def instrument(ctx: RunContext, *instrumentors: Instrumentor) -> Iterator[InstrumentationManager]:
    manager = InstrumentationManager()
    for instrumentor in instrumentors:
        manager.install(instrumentor)
    token = set_active_run_context(ctx)
    try:
        yield manager
    finally:
        reset_active_run_context(token)
        manager.close()


def http_spans(ctx: RunContext) -> list[SpanRecord]:
    return [span for span in ctx.trace.spans if span.kind == "http"]


def test_httpx_sync_redirect_hooks_and_opt_in_capture_are_preserved() -> None:
    hook_paths: list[str] = []

    def transport(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/redirect":
            return httpx.Response(307, headers={"location": "/final"}, request=request)
        return httpx.Response(
            200,
            json={"ok": True, "token": "response-secret"},
            headers={
                "x-request-id": "req-1",
                "set-cookie": "session=secret",
                "content-type": "application/json",
            },
            request=request,
            extensions={"http_version": b"HTTP/1.1"},
        )

    capture = HTTPXCapture(
        path="full",
        request_headers=("authorization", "x-trace", "missing"),
        response_headers=("x-request-id", "set-cookie"),
        request_body=True,
        response_body=True,
    )
    ctx = run_context(full=True)
    client = httpx.Client(
        transport=httpx.MockTransport(transport),
        follow_redirects=True,
        event_hooks={"response": [lambda response: hook_paths.append(response.request.url.path)]},
    )

    with instrument(ctx, HTTPX(capture=capture)):
        response = client.post(
            "https://service.test/redirect?secret=query",
            headers={"authorization": "Bearer secret", "x-trace": "trace-1"},
            json={"name": "Ada", "password": "request-secret", "nested": [{"api_key": "x"}]},
        )
    client.close()

    assert response.json() == {"ok": True, "token": "response-secret"}
    assert hook_paths == ["/redirect", "/final"]
    spans = http_spans(ctx)
    assert len(spans) == 2
    assert spans[0].attributes["http.request.path"] == "/redirect"
    assert "query" not in str(spans[0].attributes)
    assert spans[0].attributes["http.request.headers"] == {
        "authorization": "[REDACTED]",
        "x-trace": "trace-1",
    }
    assert spans[1].attributes["http.response.headers"] == {
        "x-request-id": "req-1",
        "set-cookie": "[REDACTED]",
    }
    assert spans[1].attributes["network.protocol.version"] == "HTTP/1.1"
    response_size = spans[1].attributes["http.response.body.size"]
    assert isinstance(response_size, int)
    assert response_size > 0
    body_artifacts = [artifact for artifact in ctx.artifacts if ".body" in artifact.name]
    assert body_artifacts
    assert all("secret" not in str(artifact.value) for artifact in body_artifacts)
    assert ctx.asset_versions == []
    assert ctx.asset_uses == []


def test_httpx_default_capture_hashes_path_and_omits_headers_and_bodies() -> None:
    ctx = run_context()
    client = httpx.Client(
        transport=httpx.MockTransport(lambda request: httpx.Response(200, text="ok"))
    )
    with instrument(ctx, HTTPX()):
        response = client.get(
            "https://user:pass@service.test:8443/private/path?api_key=secret",
            headers={"authorization": "secret"},
        )
    client.close()

    assert response.text == "ok"
    span = http_spans(ctx)[0]
    assert span.attributes["http.request.method"] == "GET"
    assert span.attributes["http.request.scheme"] == "https"
    assert span.attributes["http.request.host"] == "service.test"
    assert span.attributes["http.request.port"] == 8443
    assert "http.request.path_hash" in span.attributes
    assert "http.request.path" not in span.attributes
    assert "http.request.headers" not in span.attributes
    assert not ctx.artifacts


def test_httpx_call_finalization_is_idempotent() -> None:
    ctx = run_context()
    instrumentor = HTTPX()
    handler = httpx_instrumentation._HTTPXHandler(  # pyright: ignore[reportPrivateUsage]
        InstrumentationRuntime(),
        instrumentor.info,
        instrumentor.capture,
    )
    span = ctx.span("httpx.idempotence", kind="http")
    span.__enter__()
    call = httpx_instrumentation._HTTPXCall(  # pyright: ignore[reportPrivateUsage]
        handler,
        span,
        instrumentor.capture,
        httpx.Request("GET", "https://service.test"),
    )

    call.finish()
    signal_count = len(ctx.trace.signals)
    call.finish()

    assert len(ctx.trace.signals) == signal_count
    assert next(item for item in ctx.trace.spans if item.operation == "httpx.idempotence")


def test_httpx_sync_stream_completion_early_close_failure_and_transport_error() -> None:
    streams: list[SyncChunks] = []

    def transport(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/transport-error":
            raise httpx.ConnectError("connect failed", request=request)
        stream = SyncChunks((b"one", b"two"), fail=request.url.path == "/stream-error")
        streams.append(stream)
        return httpx.Response(200, stream=stream, request=request)

    ctx = run_context()
    client = httpx.Client(transport=httpx.MockTransport(transport))
    with instrument(ctx, HTTPX()):
        assert client.get("https://service.test/complete").content == b"onetwo"
        with client.stream("GET", "https://service.test/partial") as response:
            assert next(response.iter_raw()) == b"one"
        with pytest.raises(httpx.ReadError, match="stream failed"):
            client.get("https://service.test/stream-error")
        with pytest.raises(httpx.ConnectError, match="connect failed") as captured:
            client.get("https://service.test/transport-error")
    client.close()

    assert captured.value.request.url.path == "/transport-error"
    spans = http_spans(ctx)
    assert [span.end_reason for span in spans] == [
        EndReason.COMPLETED,
        EndReason.CANCELLED,
        EndReason.FAILED,
        EndReason.FAILED,
    ]
    assert spans[1].partial is True
    assert spans[2].attributes["error.type"] == "ReadError"
    assert spans[3].attributes["error.type"] == "ConnectError"
    assert all(stream.closed for stream in streams)


def test_httpx_sync_stream_close_failure_and_invalid_chunk_preserve_errors() -> None:
    streams: dict[str, SyncChunks] = {}

    def transport(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/invalid":
            return httpx.Response(200, stream=InvalidSyncChunks(), request=request)
        stream = SyncChunks((b"one",), close_error=True)
        streams[request.url.path] = stream
        return httpx.Response(200, stream=stream, request=request)

    ctx = run_context()
    client = httpx.Client(transport=httpx.MockTransport(transport))
    with instrument(ctx, HTTPX()):
        response = client.send(
            client.build_request("GET", "https://service.test/close-error"), stream=True
        )
        with pytest.raises(httpx.ReadError, match="sync close failed"):
            response.close()
        with pytest.raises(TypeError, match="must yield bytes"):
            client.get("https://service.test/invalid")
    client.close()

    spans = http_spans(ctx)
    assert [span.end_reason for span in spans] == [EndReason.FAILED, EndReason.FAILED]
    assert streams["/close-error"].closed is True


async def test_httpx_async_streams_concurrency_cancellation_and_close_error() -> None:
    streams: dict[str, AsyncChunks] = {}

    def transport(request: httpx.Request) -> httpx.Response:
        path = request.url.path
        stream = AsyncChunks(
            (path.encode(), b"-done"),
            fail=path == "/failed",
            close_error=path == "/close-error",
        )
        streams[path] = stream
        return httpx.Response(200, stream=stream, request=request)

    ctx = run_context()
    client = httpx.AsyncClient(transport=httpx.MockTransport(transport))
    with instrument(ctx, HTTPX()):
        responses = await asyncio.gather(
            client.get("https://service.test/a"),
            client.get("https://service.test/b"),
        )
        assert [response.content for response in responses] == [b"/a-done", b"/b-done"]
        response = await client.send(
            client.build_request("GET", "https://service.test/partial"), stream=True
        )
        iterator = response.aiter_raw()
        assert await anext(iterator) == b"/partial"
        await response.aclose()
        with pytest.raises(httpx.ReadError, match="async stream failed"):
            await client.get("https://service.test/failed")
        close_response = await client.send(
            client.build_request("GET", "https://service.test/close-error"), stream=True
        )
        with pytest.raises(httpx.ReadError, match="close failed"):
            await close_response.aclose()
    await client.aclose()

    spans = http_spans(ctx)
    reasons = [span.end_reason for span in spans]
    assert reasons.count(EndReason.COMPLETED) == 2
    assert reasons.count(EndReason.CANCELLED) == 1
    assert reasons.count(EndReason.FAILED) == 2
    concurrent = [span for span in spans if span.attributes["http.request.path_hash"]]
    assert len({span.parent_span_id for span in concurrent[:2]}) == 1


async def test_httpx_async_transport_cancellation_marks_partial_span() -> None:
    async def transport(request: httpx.Request) -> httpx.Response:
        raise asyncio.CancelledError

    ctx = run_context()
    client = httpx.AsyncClient(transport=httpx.MockTransport(transport))
    with instrument(ctx, HTTPX()), pytest.raises(asyncio.CancelledError):
        await client.get("https://service.test/cancelled")
    await client.aclose()

    span = http_spans(ctx)[0]
    assert span.end_reason is EndReason.CANCELLED
    assert span.partial is True


def test_httpx_custom_transport_restoration_suppression_and_unread_request_body() -> None:
    original = getattr_static(CustomTransport, "handle_request")
    capture = HTTPXCapture(path="omit", request_body=True)
    instrumentor = HTTPX(capture=capture, transports=(CustomTransport,))
    assert instrumentor.check().installable is True
    ctx = run_context()
    client = httpx.Client(transport=CustomTransport())
    with instrument(ctx, instrumentor):
        with suppress_instrumentation("httpx"):
            assert client.get("https://service.test/suppressed").status_code == 204
        request = httpx.Request(
            "POST",
            "https://service.test/streamed",
            content=iter((b"not", b"read")),
        )
        assert client.send(request).status_code == 204
    client.close()

    assert getattr_static(CustomTransport, "handle_request") is original
    spans = http_spans(ctx)
    assert len(spans) == 1
    assert "http.request.path" not in spans[0].attributes
    assert "http.request.path_hash" not in spans[0].attributes
    assert spans[0].attributes["http.request.body.unavailable"] is True
    assert spans[0].attributes["network.protocol.version"] == "HTTP/2"


def test_httpx_body_capture_handles_truncation_text_malformed_json_and_binary() -> None:
    def transport(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/binary":
            return httpx.Response(
                200,
                content=b"\x00\x01\x02\x03",
                headers={"content-type": "application/octet-stream"},
            )
        return httpx.Response(
            200,
            content=b"token=secret; value=kept",
            headers={"content-type": "text/plain"},
        )

    capture = HTTPXCapture(request_body=True, response_body=True, max_body_bytes=16)
    ctx = run_context(full=True)
    client = httpx.Client(transport=httpx.MockTransport(transport))
    with instrument(ctx, HTTPX(capture=capture)):
        assert (
            client.post(
                "https://service.test/text",
                content=b'{broken token="secret"}',
                headers={"content-type": "application/json"},
            ).status_code
            == 200
        )
        assert client.get("https://service.test/binary").content == b"\x00\x01\x02\x03"
    client.close()

    spans = http_spans(ctx)
    assert spans[0].attributes["http.request.body.truncated"] is True
    assert spans[0].attributes["http.response.body.truncated"] is True
    assert spans[1].attributes["http.response.body.binary"] is True
    assert all("secret" not in str(artifact.value) for artifact in ctx.artifacts)


def test_httpx_no_active_run_and_open_stream_manager_close_are_safe() -> None:
    stream = SyncChunks((b"later",))
    client = httpx.Client(
        transport=httpx.MockTransport(
            lambda request: httpx.Response(200, stream=stream, request=request)
        )
    )
    manager = InstrumentationManager()
    manager.install(HTTPX())
    assert client.get("https://service.test/outside").content == b"later"

    ctx = run_context()
    token = set_active_run_context(ctx)
    response = client.send(
        client.build_request("GET", "https://service.test/abandoned"), stream=True
    )
    reset_active_run_context(token)
    manager.close()
    assert response.read() == b"later"
    response.close()
    client.close()

    span = http_spans(ctx)[0]
    assert span.end_reason is EndReason.ABANDONED
    assert span.partial is True


async def test_httpx_late_async_stream_use_after_manager_close_does_not_leak_context() -> None:
    stream = AsyncChunks((b"late",))
    client = httpx.AsyncClient(
        transport=httpx.MockTransport(
            lambda request: httpx.Response(200, stream=stream, request=request)
        )
    )
    manager = InstrumentationManager()
    manager.install(HTTPX())
    ctx = run_context()
    token = set_active_run_context(ctx)
    response = await client.send(
        client.build_request("GET", "https://service.test/late"), stream=True
    )
    reset_active_run_context(token)
    manager.close()

    assert await response.aread() == b"late"
    await response.aclose()
    await client.aclose()

    span = http_spans(ctx)[0]
    assert span.end_reason is EndReason.ABANDONED


def test_httpx_internal_capture_failure_is_diagnostic_and_request_survives(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ctx = run_context()
    client = httpx.Client(
        transport=httpx.MockTransport(lambda request: httpx.Response(200, text="ok"))
    )

    def fail_span(*args: Any, **kwargs: Any) -> None:
        raise RuntimeError("capture failed")

    monkeypatch.setattr(ctx, "span", fail_span)
    with instrument(ctx, HTTPX()):
        assert client.get("https://service.test/diagnostic").text == "ok"
    client.close()

    assert not http_spans(ctx)
    assert any(
        diagnostic.code == "httpx_instrumentation_error" for diagnostic in ctx.trace.diagnostics
    )


def test_httpx_install_rolls_back_already_patched_transports(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = InstrumentationRuntime()
    original = getattr_static(httpx.HTTPTransport, "handle_request")
    patch_method = runtime.patch_method
    calls = 0

    def fail_second_patch(*args: Any, **kwargs: Any) -> Any:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("patch failed")
        return patch_method(*args, **kwargs)

    monkeypatch.setattr(runtime, "patch_method", fail_second_patch)
    with pytest.raises(RuntimeError, match="patch failed"):
        HTTPX().install(runtime)
    assert getattr_static(httpx.HTTPTransport, "handle_request") is original


async def test_httpx_composes_under_openai_without_duplicate_usage() -> None:
    def transport(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "id": "chatcmpl-httpx",
                "object": "chat.completion",
                "created": 1,
                "model": "gpt-test",
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "done"},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"prompt_tokens": 3, "completion_tokens": 2, "total_tokens": 5},
            },
            request=request,
        )

    ctx = run_context()
    http_client = httpx.AsyncClient(transport=httpx.MockTransport(transport))
    client = AsyncOpenAI(api_key="test", base_url="https://openai.test/v1", http_client=http_client)
    with instrument(ctx, OpenAIClient(), HTTPX()):
        result = await client.chat.completions.create(
            model="gpt-test", messages=[{"role": "user", "content": "hello"}]
        )
    await client.close()

    assert result.choices[0].message.content == "done"
    provider = next(span for span in ctx.trace.spans if span.operation == "openai.chat.completions")
    transport_span = http_spans(ctx)[0]
    assert transport_span.parent_span_id == provider.span_id
    assert provider.usage == {"input_tokens": 3, "output_tokens": 2, "total_tokens": 5}
    assert transport_span.usage == {}


async def test_httpx_composes_with_pydantic_ai_and_openai_layers() -> None:
    def transport(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "id": "chatcmpl-agent",
                "object": "chat.completion",
                "created": 1,
                "model": "gpt-test",
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "hello"},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"prompt_tokens": 2, "completion_tokens": 1, "total_tokens": 3},
            },
            request=request,
        )

    ctx = run_context()
    http_client = httpx.AsyncClient(transport=httpx.MockTransport(transport))
    openai_client = AsyncOpenAI(
        api_key="test", base_url="https://openai.test/v1", http_client=http_client
    )
    model = OpenAIChatModel("gpt-test", provider=OpenAIProvider(openai_client=openai_client))
    agent = Agent(model)
    with instrument(ctx, PydanticAI(), OpenAIClient(), HTTPX()):
        result = await agent.run("say hello")
    await openai_client.close()

    assert result.output == "hello"
    operations = {span.operation: span for span in ctx.trace.spans}
    assert operations["openai.chat.completions"].parent_span_id is not None
    assert (
        operations["httpx.request"].parent_span_id == operations["openai.chat.completions"].span_id
    )


def test_httpx_capture_validation_and_compatibility_failures(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with pytest.raises(ValidationError, match="cannot be empty"):
        HTTPXCapture(request_headers=(" ",))
    with pytest.raises(ValidationError, match="greater than or equal to 1"):
        HTTPXCapture(max_body_bytes=0)

    instrumentor = HTTPX()
    monkeypatch.setattr("autobench.instrumentation.httpx.client.find_spec", lambda name: None)
    unavailable = instrumentor.check()
    assert unavailable.status.value == "unavailable"

    monkeypatch.setattr("autobench.instrumentation.httpx.client.find_spec", lambda name: True)
    monkeypatch.delattr(httpx.WSGITransport, "handle_request")
    unsupported = instrumentor.check()
    assert unsupported.status.value == "unsupported"
