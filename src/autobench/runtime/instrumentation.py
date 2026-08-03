from __future__ import annotations as _annotations

from collections.abc import Callable, Mapping
from contextvars import Token
from dataclasses import dataclass
from itertools import count
from typing import Any

from autobench._version import __version__
from autobench.instrumentation.manager import InstrumentationRuntime
from autobench.instrumentation.models import (
    InstrumentationHandle,
    InstrumentCall,
    InstrumentFactorSpec,
    InstrumentMetricSpec,
    InstrumentorCapabilities,
    InstrumentorInfo,
)
from autobench.instrumentation.patching import CallLifecycle, PatchManager
from autobench.metrics.observations import ObservationSource
from autobench.protocol.context import ActiveContext, attach_context, reset_context
from autobench.protocol.signals import (
    AbstractionLayer,
    CaptureMechanism,
    EndReason,
    InstrumentationScope,
)
from autobench.runtime.context import RunContext, Span, SpanKind, active_run_context


@dataclass(slots=True)
class _MethodInstrumentation:
    span: str | None
    span_kind: SpanKind | str
    metrics: tuple[InstrumentMetricSpec, ...]
    factors: tuple[InstrumentFactorSpec, ...]
    scope: InstrumentationScope
    operation_family: str


class _MethodCall:
    def __init__(
        self,
        handler: _MethodHandler,
        instrumentation: _MethodInstrumentation,
        ctx: RunContext,
        call: InstrumentCall,
        span: Span | None,
    ) -> None:
        self._handler = handler
        self._instrumentation = instrumentation
        self._ctx = ctx
        self._call = call
        self._span = span
        self._finished = False

    def resume(self) -> None:
        if self._finished:
            return
        if self._span is not None:
            self._span.resume()

    def suspend(self) -> None:
        if self._finished:
            return
        if self._span is not None:
            self._span.suspend()

    def observe(self, item: Any) -> None:
        if self._finished:
            return
        self._call.stream_item_count += 1
        self._call.last_stream_item = item

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
        self._call.result = result
        self._call.error = error
        try:
            span_id = None if self._span is None else self._span.id
            _emit_records(self._ctx, span_id, self._instrumentation, self._call)
            if self._span is not None:
                self._span.set_output(result)
                if self._call.stream_item_count:
                    self._span.set_attribute("stream_item_count", self._call.stream_item_count)
                self._span.finish(error=error, reason=reason, partial=partial)
        finally:
            if self._span is not None:
                self._span.suspend()
            self._finished = True
            self._handler.discard(self)


class _MethodHandler:
    def __init__(
        self,
        instrumentation: _MethodInstrumentation,
        runtime: InstrumentationRuntime,
        info: InstrumentorInfo,
    ) -> None:
        self._instrumentation = instrumentation
        self._runtime = runtime
        self._info = info
        self._active: set[_MethodCall] = set()

    @property
    def suppression_keys(self) -> tuple[str, ...]:
        return self._info.id, self._instrumentation.operation_family

    def begin(self, call: InstrumentCall) -> CallLifecycle | None:
        ctx = get_active_run_context()
        if ctx is None:
            return None
        span = None
        if self._instrumentation.span is not None:
            span = ctx.span(
                self._instrumentation.span,
                kind=self._instrumentation.span_kind,
                instrumentation_scope=self._instrumentation.scope,
            )
            span.__enter__()
        active_call = _MethodCall(self, self._instrumentation, ctx, call, span)
        self._active.add(active_call)
        return active_call

    def diagnose(self, stage: str, error: Exception) -> None:
        self._runtime.diagnose(
            self._info,
            "instrumentation_callback_error",
            f"{stage}: {type(error).__name__}: {error}",
        )

    def discard(self, call: _MethodCall) -> None:
        self._active.discard(call)

    def close(self) -> None:
        for call in tuple(self._active):
            try:
                call.finish(reason=EndReason.ABANDONED, partial=True)
            except Exception as exc:
                self.diagnose("close", exc)


_METHOD_INFO = InstrumentorInfo(
    id="autobench.instrument_method",
    version=__version__,
    mechanism=CaptureMechanism.PATCH,
    layer=AbstractionLayer.APPLICATION,
    capabilities=InstrumentorCapabilities(sync=True, async_=True, streaming=True),
)
_METHOD_PATCHES = PatchManager()
_METHOD_RUNTIME = InstrumentationRuntime(_METHOD_PATCHES)
_METHOD_OWNER_INDEX = count(1)


def set_active_run_context(ctx: RunContext | None) -> Token[ActiveContext | None]:
    return attach_context(None if ctx is None else ctx.active_context)


def reset_active_run_context(token: Token[ActiveContext | None]) -> None:
    reset_context(token)


def get_active_run_context() -> RunContext | None:
    return active_run_context()


def instrument_method(
    target: type[Any],
    method_name: str,
    *,
    span: str | None = None,
    span_kind: SpanKind | str = SpanKind.CUSTOM,
    metrics: list[InstrumentMetricSpec] | None = None,
    factors: list[InstrumentFactorSpec] | None = None,
    operation_family: str | None = None,
) -> InstrumentationHandle:
    family = operation_family or f"{target.__module__}.{target.__qualname__}.{method_name}"
    instrumentation = _MethodInstrumentation(
        span=span,
        span_kind=span_kind,
        metrics=tuple(metrics or ()),
        factors=tuple(factors or ()),
        scope=_METHOD_RUNTIME.scope(_METHOD_INFO),
        operation_family=family,
    )
    handler = _MethodHandler(instrumentation, _METHOD_RUNTIME, _METHOD_INFO)
    owner = f"{_METHOD_INFO.id}:{next(_METHOD_OWNER_INDEX)}"
    patch_handle = _METHOD_PATCHES.patch_method(
        target,
        method_name,
        owner=owner,
        handler=handler,
    )
    return InstrumentationHandle(patch_handle.close, info=_METHOD_INFO)


def _emit_records(
    ctx: RunContext,
    span_id: str | None,
    instrumentation: _MethodInstrumentation,
    call: InstrumentCall,
) -> None:
    for metric in instrumentation.metrics:
        try:
            ctx.metric(
                metric.name,
                _extract_value(metric.value_path, metric.value_factory, call),
                semantic_type=metric.semantic_type,
                unit=metric.unit,
                direction=metric.direction,
                role=metric.role,
                span_id=span_id,
                tags=metric.tags,
                source=ObservationSource.INSTRUMENTATION,
            )
        except Exception as exc:
            ctx.error(exc, span_id=span_id)

    for factor in instrumentation.factors:
        try:
            ctx.factor_observation(
                factor.name,
                _extract_value(factor.value_path, factor.value_factory, call),
                semantic_type=factor.semantic_type,
                span_id=span_id,
                tags=factor.tags,
                source=ObservationSource.INSTRUMENTATION,
            )
        except Exception as exc:
            ctx.error(exc, span_id=span_id)


def _extract_value(
    value_path: str | None,
    value_factory: Callable[[InstrumentCall], Any] | None,
    call: InstrumentCall,
) -> Any:
    if value_factory is not None:
        return value_factory(call)

    assert value_path is not None
    current: Any = call
    parts = value_path.split(".")
    for index, part in enumerate(parts):
        if callable(current):
            current = current()
        if isinstance(current, Mapping):
            for key, value in current.items():
                if key == part:
                    current = value
                    break
            else:
                raise KeyError(f"Instrumentation path segment '{part}' not found.")
        else:
            try:
                current = getattr(current, part)
            except AttributeError as exc:
                raise KeyError(f"Instrumentation path segment '{part}' not found.") from exc
        if callable(current) and index < len(parts) - 1:
            current = current()
    return current


__all__ = (
    "InstrumentCall",
    "InstrumentFactorSpec",
    "InstrumentationHandle",
    "InstrumentMetricSpec",
    "get_active_run_context",
    "instrument_method",
    "reset_active_run_context",
    "set_active_run_context",
)
