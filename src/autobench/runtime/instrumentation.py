from __future__ import annotations as _annotations

from collections.abc import Callable, Mapping
from contextlib import AbstractContextManager
from contextvars import ContextVar, Token
from dataclasses import dataclass
from functools import wraps
from inspect import getattr_static, iscoroutinefunction
from types import TracebackType
from typing import Any
from weakref import WeakKeyDictionary

from pydantic import BaseModel, Field, model_validator

from autobench.metrics.observations import Direction, ObservationRole, ObservationSource
from autobench.metrics.semantics import SemanticType
from autobench.runtime.context import RunContext

_ACTIVE_RUN_CONTEXT: ContextVar[RunContext | None] = ContextVar(
    "autobench_active_run_context",
    default=None,
)


class InstrumentMetricSpec(BaseModel):
    name: str = Field(min_length=1)
    semantic_type: SemanticType | None = None
    unit: str | None = None
    direction: Direction | None = None
    role: ObservationRole | None = None
    tags: dict[str, Any] = Field(default_factory=dict)
    value_path: str | None = None
    value_factory: Callable[[InstrumentCall], Any] | None = None

    @model_validator(mode="after")
    def _validate_extractor(self) -> InstrumentMetricSpec:
        if self.value_path is None and self.value_factory is None:
            raise ValueError("instrument metrics require value_path or value_factory")
        return self


class InstrumentFactorSpec(BaseModel):
    name: str = Field(min_length=1)
    semantic_type: SemanticType | None = None
    tags: dict[str, Any] = Field(default_factory=dict)
    value_path: str | None = None
    value_factory: Callable[[InstrumentCall], Any] | None = None

    @model_validator(mode="after")
    def _validate_extractor(self) -> InstrumentFactorSpec:
        if self.value_path is None and self.value_factory is None:
            raise ValueError("instrument factors require value_path or value_factory")
        return self


@dataclass(slots=True)
class InstrumentCall:
    instance: Any | None
    args: tuple[Any, ...]
    kwargs: dict[str, Any]
    result: Any = None
    error: BaseException | None = None


@dataclass(slots=True)
class _MethodInstrumentation:
    span: str | None
    metrics: list[InstrumentMetricSpec]
    factors: list[InstrumentFactorSpec]


@dataclass(slots=True)
class _WrappedMethodState:
    original_descriptor: Any
    instrumentations: list[_MethodInstrumentation]


@dataclass(slots=True)
class _MethodTarget:
    owner: type[Any]
    method_name: str
    descriptor_kind: str
    original_descriptor: Any
    wrapped: Callable[..., Any]


_WRAPPED_METHODS: WeakKeyDictionary[Callable[..., Any], _WrappedMethodState] = WeakKeyDictionary()


class InstrumentationHandle(AbstractContextManager["InstrumentationHandle"]):
    def __init__(
        self,
        *,
        method_target: _MethodTarget,
        instrumentation: _MethodInstrumentation,
    ) -> None:
        self._method_target = method_target
        self._instrumentation = instrumentation
        self._closed = False

    def close(self) -> None:
        if self._closed:
            return
        instrumentations = _instrumentations_for(self._method_target.wrapped)
        if self._instrumentation in instrumentations:
            instrumentations.remove(self._instrumentation)
        if not instrumentations:
            setattr(
                self._method_target.owner,
                self._method_target.method_name,
                self._method_target.original_descriptor,
            )
        self._closed = True

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> bool | None:
        self.close()
        return None


def set_active_run_context(ctx: RunContext | None) -> Token[RunContext | None]:
    return _ACTIVE_RUN_CONTEXT.set(ctx)


def reset_active_run_context(token: Token[RunContext | None]) -> None:
    _ACTIVE_RUN_CONTEXT.reset(token)


def get_active_run_context() -> RunContext | None:
    return _ACTIVE_RUN_CONTEXT.get()


def instrument_method(
    target: type[Any],
    method_name: str,
    *,
    span: str | None = None,
    metrics: list[InstrumentMetricSpec] | None = None,
    factors: list[InstrumentFactorSpec] | None = None,
) -> InstrumentationHandle:
    method_target = _resolve_method_target(target, method_name)
    entry = _MethodInstrumentation(
        span=span,
        metrics=list(metrics or []),
        factors=list(factors or []),
    )
    instrumentations = _instrumentations_for(method_target.wrapped)
    instrumentations.append(entry)
    return InstrumentationHandle(method_target=method_target, instrumentation=entry)


def _instrumentations_for(wrapped: Callable[..., Any]) -> list[_MethodInstrumentation]:
    try:
        return _WRAPPED_METHODS[wrapped].instrumentations
    except KeyError as exc:
        raise RuntimeError("method is not instrumented by Autobench") from exc


def _emit_instrumentation(
    instrumentations: list[_MethodInstrumentation],
    call: InstrumentCall,
) -> None:
    ctx = get_active_run_context()
    if ctx is None:
        return

    for instrumentation in instrumentations:
        if instrumentation.span is None:
            _emit_records(ctx, None, instrumentation, call)
            continue
        with ctx.span(instrumentation.span) as span:
            _emit_records(ctx, span.id, instrumentation, call)


def _emit_records(
    ctx: RunContext,
    span_id: str | None,
    instrumentation: _MethodInstrumentation,
    call: InstrumentCall,
) -> None:
    for metric in instrumentation.metrics:
        try:
            observation = ctx.metric(
                metric.name,
                _extract_value(metric.value_path, metric.value_factory, call),
                semantic_type=metric.semantic_type,
                unit=metric.unit,
                direction=metric.direction,
                role=metric.role,
                span_id=span_id,
                tags=metric.tags,
            )
        except Exception as exc:
            ctx.error(exc, span_id=span_id)
            continue
        observation.source = ObservationSource.INSTRUMENTATION

    for factor in instrumentation.factors:
        try:
            observation = ctx.factor_observation(
                factor.name,
                _extract_value(factor.value_path, factor.value_factory, call),
                semantic_type=factor.semantic_type,
                span_id=span_id,
                tags=factor.tags,
            )
        except Exception as exc:
            ctx.error(exc, span_id=span_id)
            continue
        observation.source = ObservationSource.INSTRUMENTATION


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
            if not hasattr(current, part):
                raise KeyError(f"Instrumentation path segment '{part}' not found.")
            current = getattr(current, part)
        if callable(current) and index < len(parts) - 1:
            current = current()
    return current


def _resolve_method_target(target: type[Any], method_name: str) -> _MethodTarget:
    descriptor = getattr_static(target, method_name)
    if isinstance(descriptor, property):
        raise TypeError("property instrumentation is not supported")

    descriptor_kind = "instance"
    original_descriptor = descriptor
    original_callable = descriptor
    if isinstance(descriptor, staticmethod):
        descriptor_kind = "staticmethod"
        original_callable = descriptor.__func__
    elif isinstance(descriptor, classmethod):
        descriptor_kind = "classmethod"
        original_callable = descriptor.__func__

    if not isinstance(original_callable, Callable):
        raise TypeError(f"{target.__name__}.{method_name} is not callable")

    wrapped = original_callable
    wrapped_state = _WRAPPED_METHODS.get(wrapped)
    if wrapped_state is None:
        wrapped = _wrap_callable(wrapped, descriptor_kind)
        _WRAPPED_METHODS[wrapped] = _WrappedMethodState(
            original_descriptor=original_descriptor,
            instrumentations=[],
        )
        setattr(target, method_name, _bind_descriptor(wrapped, descriptor_kind))
        original_for_restore = original_descriptor
    else:
        original_for_restore = wrapped_state.original_descriptor
    return _MethodTarget(
        owner=target,
        method_name=method_name,
        descriptor_kind=descriptor_kind,
        original_descriptor=original_for_restore,
        wrapped=wrapped,
    )


def _wrap_callable(original: Callable[..., Any], descriptor_kind: str) -> Callable[..., Any]:
    if iscoroutinefunction(original):

        @wraps(original)
        async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
            instance = _bound_instance(descriptor_kind, args)
            call_args = _call_args(descriptor_kind, args)
            call = InstrumentCall(instance=instance, args=call_args, kwargs=kwargs)
            try:
                call.result = await original(*args, **kwargs)
            except BaseException as exc:
                call.error = exc
                _emit_instrumentation(_instrumentations_for(async_wrapper), call)
                raise
            _emit_instrumentation(_instrumentations_for(async_wrapper), call)
            return call.result

        return async_wrapper

    @wraps(original)
    def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
        instance = _bound_instance(descriptor_kind, args)
        call_args = _call_args(descriptor_kind, args)
        call = InstrumentCall(instance=instance, args=call_args, kwargs=kwargs)
        try:
            call.result = original(*args, **kwargs)
        except BaseException as exc:
            call.error = exc
            _emit_instrumentation(_instrumentations_for(sync_wrapper), call)
            raise
        _emit_instrumentation(_instrumentations_for(sync_wrapper), call)
        return call.result

    return sync_wrapper


def _bind_descriptor(wrapped: Callable[..., Any], descriptor_kind: str) -> Any:
    if descriptor_kind == "staticmethod":
        return staticmethod(wrapped)
    if descriptor_kind == "classmethod":
        return classmethod(wrapped)
    return wrapped


def _bound_instance(descriptor_kind: str, args: tuple[Any, ...]) -> Any | None:
    if descriptor_kind in {"instance", "classmethod"} and args:
        return args[0]
    return None


def _call_args(descriptor_kind: str, args: tuple[Any, ...]) -> tuple[Any, ...]:
    if descriptor_kind in {"instance", "classmethod"} and args:
        return args[1:]
    return args


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
