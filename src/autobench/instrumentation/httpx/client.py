from __future__ import annotations

import asyncio
import json
import re
from collections.abc import AsyncIterator, Iterator
from hashlib import sha256
from importlib.metadata import version
from importlib.util import find_spec
from typing import Any, Literal, TypeAlias, cast

import httpx
from pydantic import BaseModel, ConfigDict, Field, field_validator

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
from autobench.runtime.context import Span, active_run_context

PathCapture: TypeAlias = Literal["omit", "hash", "full"]
TransportType: TypeAlias = type[httpx.BaseTransport] | type[httpx.AsyncBaseTransport]

_SECRET_HEADER_PARTS = (
    "api-key",
    "apikey",
    "authorization",
    "cookie",
    "password",
    "secret",
    "token",
)
_SECRET_TEXT = re.compile(
    r"(?i)(authorization|api[_-]?key|password|secret|token)(\s*[:=]\s*)"
    r"([^\s,;\"']+|\"[^\"]*\"|'[^']*')"
)


class HTTPXCapture(BaseModel):
    """Privacy-first request and response capture settings."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    path: PathCapture = "hash"
    request_headers: tuple[str, ...] = ()
    response_headers: tuple[str, ...] = ()
    request_body: bool = False
    response_body: bool = False
    max_body_bytes: int = Field(default=65_536, ge=1)

    @field_validator("request_headers", "response_headers")
    @classmethod
    def normalize_headers(cls, values: tuple[str, ...]) -> tuple[str, ...]:
        normalized = tuple(dict.fromkeys(value.strip().lower() for value in values))
        if any(not value for value in normalized):
            raise ValueError("captured header names cannot be empty")
        return normalized


class _HTTPXCall:
    def __init__(
        self,
        handler: _HTTPXHandler,
        span: Span,
        capture: HTTPXCapture,
        request: httpx.Request,
    ) -> None:
        self._handler = handler
        self._span = span
        self._capture = capture
        self._request = request
        self._response_bytes = 0
        self._response_body = bytearray()
        self._response_digest = sha256()
        self._response_content_type: str | None = None
        self._chunk_count = 0
        self._first_chunk = True
        self._response_prepared = False
        self._finished = False

    def resume(self) -> None:
        self._span.resume()

    @property
    def finished(self) -> bool:
        return self._finished

    def suspend(self) -> None:
        self._span.suspend()

    def observe(self, item: Any) -> None:
        if not isinstance(item, bytes):
            error = TypeError("HTTPX response streams must yield bytes")
            self.finish(error=error)
            raise error
        if self._first_chunk:
            self._first_chunk = False
            self._span.event("stream.first_chunk", semantic_type=Semantic.STREAM_FIRST_CHUNK)
        self._chunk_count += 1
        self._response_bytes += len(item)
        self._response_digest.update(item)
        if self._capture.response_body and len(self._response_body) < self._capture.max_body_bytes:
            remaining = self._capture.max_body_bytes - len(self._response_body)
            self._response_body.extend(item[:remaining])

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
        if isinstance(result, httpx.Response) and not self._response_prepared:
            self._response_prepared = True
            self._capture_response(result)
            if result.is_stream_consumed:
                self.observe(result.content)
            elif isinstance(result.stream, httpx.SyncByteStream):
                result.stream = _SyncResponseStream(result.stream, self)
                self.suspend()
                return
            else:
                async_stream = cast(httpx.AsyncByteStream, result.stream)
                result.stream = _AsyncResponseStream(async_stream, self)
                self.suspend()
                return

        if isinstance(error, asyncio.CancelledError):
            reason = EndReason.CANCELLED
            partial = True
        self.resume()
        try:
            if error is not None:
                self._span.set_attribute(Semantic.ERROR_TYPE, type(error).__name__)
            self._span.set_attribute(Semantic.HTTP_RESPONSE_BODY_SIZE, self._response_bytes)
            if self._chunk_count:
                self._span.set_attribute("stream.chunk_count", self._chunk_count)
            if self._capture.response_body and self._response_body:
                self._record_body(
                    "http.response.body",
                    bytes(self._response_body),
                    self._response_bytes > len(self._response_body),
                    self._response_content_type,
                    digest=self._response_digest.hexdigest(),
                )
            self._span.finish(error=error, reason=reason, partial=partial)
        finally:
            self._finished = True
            self._handler.discard(self)

    def record_request_body(self, body: bytes) -> None:
        captured = body[: self._capture.max_body_bytes]
        self._record_body(
            "http.request.body",
            captured,
            len(body) > len(captured),
            self._request.headers.get("content-type"),
            digest=sha256(body).hexdigest(),
        )

    def _capture_response(self, response: httpx.Response) -> None:
        self._response_content_type = response.headers.get("content-type")
        self._span.set_attribute(Semantic.HTTP_RESPONSE_STATUS_CODE, response.status_code)
        response_headers = _selected_headers(response.headers, self._capture.response_headers)
        if response_headers:
            self._span.set_attribute(Semantic.HTTP_RESPONSE_HEADERS, response_headers)
        content_length = _content_length(response.headers)
        if content_length is not None:
            self._span.set_attribute("http.response.body.declared_size", content_length)
        http_version = response.extensions.get("http_version")
        if isinstance(http_version, bytes):
            self._span.set_attribute(Semantic.NETWORK_PROTOCOL_VERSION, http_version.decode())

    def _record_body(
        self,
        name: str,
        body: bytes,
        truncated: bool,
        content_type: str | None,
        *,
        digest: str,
    ) -> None:
        self._span.set_attribute(f"{name}.sha256", digest)
        if truncated:
            self._span.set_attribute(f"{name}.truncated", True)
        media_type = (content_type or "application/octet-stream").split(";", maxsplit=1)[0]
        if media_type == "application/json":
            try:
                payload: Any = _redact_payload(json.loads(body))
            except (UnicodeDecodeError, json.JSONDecodeError):
                payload = _redacted_text(body)
        elif media_type.startswith("text/") or media_type in {
            "application/graphql",
            "application/x-www-form-urlencoded",
            "application/xml",
        }:
            payload = _redacted_text(body)
        else:
            self._span.set_attribute(f"{name}.binary", True)
            return
        self._span.artifact(name, payload, media_type=media_type)


class _SyncResponseStream(httpx.SyncByteStream):
    def __init__(self, stream: httpx.SyncByteStream, call: _HTTPXCall) -> None:
        self._stream = stream
        self._call = call

    def __iter__(self) -> Iterator[bytes]:
        if self._call.finished:
            yield from self._stream
            return
        iterator = iter(self._stream)
        while True:
            self._call.resume()
            try:
                chunk = next(iterator)
            except StopIteration:
                self._call.finish()
                return
            except BaseException as exc:
                self._call.finish(error=exc)
                raise
            self._call.observe(chunk)
            self._call.suspend()
            yield chunk

    def close(self) -> None:
        if self._call.finished:
            self._stream.close()
            return
        self._call.resume()
        try:
            self._stream.close()
        except BaseException as exc:
            self._call.finish(error=exc)
            raise
        self._call.finish(reason=EndReason.CANCELLED, partial=True)


class _AsyncResponseStream(httpx.AsyncByteStream):
    def __init__(self, stream: httpx.AsyncByteStream, call: _HTTPXCall) -> None:
        self._stream = stream
        self._call = call

    async def __aiter__(self) -> AsyncIterator[bytes]:
        if self._call.finished:
            async for chunk in self._stream:
                yield chunk
            return
        iterator = self._stream.__aiter__()
        while True:
            self._call.resume()
            try:
                chunk = await anext(iterator)
            except StopAsyncIteration:
                self._call.finish()
                return
            except BaseException as exc:
                self._call.finish(error=exc)
                raise
            self._call.observe(chunk)
            self._call.suspend()
            yield chunk

    async def aclose(self) -> None:
        if self._call.finished:
            await self._stream.aclose()
            return
        self._call.resume()
        try:
            await self._stream.aclose()
        except BaseException as exc:
            self._call.finish(error=exc)
            raise
        self._call.finish(reason=EndReason.CANCELLED, partial=True)


class _HTTPXHandler:
    def __init__(
        self,
        runtime: InstrumentationRuntime,
        info: InstrumentorInfo,
        capture: HTTPXCapture,
    ) -> None:
        self._runtime = runtime
        self._info = info
        self._capture = capture
        self._scope = runtime.scope(info, target_version=version("httpx"))
        self._active: set[_HTTPXCall] = set()

    @property
    def suppression_keys(self) -> tuple[str, ...]:
        return self._info.id, "httpx", "transport"

    def begin(self, call: InstrumentCall) -> CallLifecycle | None:
        run_context = active_run_context()
        if run_context is None:
            return None
        request: httpx.Request = call.args[0]
        attributes: dict[str, Any] = {
            Semantic.HTTP_REQUEST_METHOD: request.method,
            Semantic.HTTP_REQUEST_SCHEME: request.url.scheme,
            Semantic.HTTP_REQUEST_HOST: request.url.host or "",
        }
        if request.url.port is not None:
            attributes[Semantic.HTTP_REQUEST_PORT] = request.url.port
        if self._capture.path == "hash":
            attributes[Semantic.HTTP_REQUEST_PATH_HASH] = sha256(
                request.url.path.encode()
            ).hexdigest()
        elif self._capture.path == "full":
            attributes[Semantic.HTTP_REQUEST_PATH] = request.url.path
        request_headers = _selected_headers(request.headers, self._capture.request_headers)
        if request_headers:
            attributes[Semantic.HTTP_REQUEST_HEADERS] = request_headers
        content_length = _content_length(request.headers)
        if content_length is not None:
            attributes[Semantic.HTTP_REQUEST_BODY_SIZE] = content_length

        span = run_context.span(
            "httpx.request",
            kind="http",
            attributes=attributes,
            instrumentation_scope=self._scope,
        )
        span.__enter__()
        active = _HTTPXCall(self, span, self._capture, request)
        self._active.add(active)
        if self._capture.request_body:
            try:
                body = request.content
            except httpx.RequestNotRead:
                span.set_attribute("http.request.body.unavailable", True)
            else:
                span.set_attribute(Semantic.HTTP_REQUEST_BODY_SIZE, len(body))
                active.record_request_body(body)
        return active

    def diagnose(self, stage: str, error: Exception) -> None:
        self._runtime.diagnose(
            self._info,
            "httpx_instrumentation_error",
            f"{stage}: {type(error).__name__}: {error}",
        )

    def discard(self, call: _HTTPXCall) -> None:
        self._active.discard(call)

    def close(self) -> None:
        for call in tuple(self._active):
            call.finish(reason=EndReason.ABANDONED, partial=True)


class HTTPX:
    """Install transport-level ABP capture for HTTPX public transport methods."""

    def __init__(
        self,
        *,
        capture: HTTPXCapture | None = None,
        transports: tuple[TransportType, ...] = (),
    ) -> None:
        self.capture = HTTPXCapture() if capture is None else capture
        self.transports = transports
        self._info = InstrumentorInfo(
            id="autobench.httpx",
            version=__version__,
            target_distribution="httpx",
            supported_versions=">=0.28,<0.29",
            mechanism=CaptureMechanism.PATCH,
            layer=AbstractionLayer.TRANSPORT,
            span_kinds=("http",),
            semantic_families=("http", "network", "stream", "error"),
            source_convention="httpx",
            source_convention_version="0.28",
            capabilities=InstrumentorCapabilities(sync=True, async_=True, streaming=True),
        )

    @property
    def info(self) -> InstrumentorInfo:
        return self._info

    def check(self) -> Compatibility:
        if find_spec("httpx") is None:
            return Compatibility(
                status=CompatibilityStatus.UNAVAILABLE,
                diagnostics=("HTTPX is unavailable; install Autobench with the 'httpx' extra",),
            )
        required = (
            (httpx.HTTPTransport, "handle_request"),
            (httpx.AsyncHTTPTransport, "handle_async_request"),
            (httpx.MockTransport, "handle_request"),
            (httpx.MockTransport, "handle_async_request"),
            (httpx.WSGITransport, "handle_request"),
            (httpx.ASGITransport, "handle_async_request"),
        )
        missing = tuple(
            f"{target.__name__}.{method}"
            for target, method in required
            if method not in target.__dict__
        )
        if missing:
            return Compatibility(
                status=CompatibilityStatus.UNSUPPORTED,
                diagnostics=(f"HTTPX lacks required public transport seams: {', '.join(missing)}",),
            )
        return Compatibility.compatible(target_version=version("httpx"))

    def install(self, runtime: InstrumentationRuntime) -> InstrumentationHandle:
        handler = _HTTPXHandler(runtime, self.info, self.capture)
        targets = tuple(
            dict.fromkeys(
                (
                    httpx.HTTPTransport,
                    httpx.AsyncHTTPTransport,
                    httpx.MockTransport,
                    httpx.WSGITransport,
                    httpx.ASGITransport,
                    *self.transports,
                )
            )
        )
        handles: list[InstrumentationHandle] = []
        try:
            for target in targets:
                for method in ("handle_request", "handle_async_request"):
                    if method in target.__dict__:
                        handles.append(runtime.patch_method(self.info, target, method, handler))
        except BaseException:
            for handle in reversed(handles):
                handle.close()
            raise

        def close() -> None:
            for handle in reversed(handles):
                handle.close()

        return InstrumentationHandle(close, info=self.info)


def _selected_headers(headers: httpx.Headers, names: tuple[str, ...]) -> dict[str, str]:
    selected: dict[str, str] = {}
    for name in names:
        if name not in headers:
            continue
        selected[name] = (
            "[REDACTED]"
            if any(part in name for part in _SECRET_HEADER_PARTS)
            else ", ".join(headers.get_list(name))
        )
    return selected


def _content_length(headers: httpx.Headers) -> int | None:
    value = headers.get("content-length")
    return int(value) if value is not None and value.isdigit() else None


def _redacted_text(body: bytes) -> str:
    text = body.decode(errors="replace")
    return _SECRET_TEXT.sub(lambda match: f"{match.group(1)}{match.group(2)}[REDACTED]", text)


def _redact_payload(value: Any, *, name: str | None = None) -> Any:
    if name is not None and any(
        part in name.lower().replace("_", "-") for part in _SECRET_HEADER_PARTS
    ):
        return "[REDACTED]"
    if isinstance(value, dict):
        return {key: _redact_payload(item, name=key) for key, item in value.items()}
    if isinstance(value, list):
        return [_redact_payload(item) for item in value]
    return value


__all__ = ("HTTPX", "HTTPXCapture", "PathCapture", "TransportType")
