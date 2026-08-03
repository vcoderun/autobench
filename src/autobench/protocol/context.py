from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from contextvars import Context, ContextVar, Token, copy_context
from dataclasses import dataclass, field, replace
from functools import wraps
from types import MappingProxyType
from typing import TYPE_CHECKING, ParamSpec, TypeVar

from autobench.protocol.capture import CapturePolicy
from autobench.protocol.ids import SpanId, TraceId
from autobench.protocol.signals import ExecutionRef
from autobench.protocol.values import SerializedValue

if TYPE_CHECKING:
    from autobench.protocol.collector import LocalCollector

P = ParamSpec("P")
R = TypeVar("R")


@dataclass(frozen=True, slots=True)
class ActiveContext:
    collector: LocalCollector
    trace_id: TraceId
    current_span_id: SpanId | None = None
    execution: ExecutionRef | None = None
    capture_policy: CapturePolicy | None = None
    suppressed: bool = False
    suppression_keys: frozenset[str] = frozenset()
    correlations: Mapping[str, SerializedValue] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "correlations", MappingProxyType(dict(self.correlations)))

    def with_span(self, span_id: SpanId | None) -> ActiveContext:
        return replace(self, current_span_id=span_id)

    def suppress(self, *keys: str) -> ActiveContext:
        if not keys:
            return replace(self, suppressed=True)
        return replace(self, suppression_keys=self.suppression_keys.union(keys))

    def is_suppressed(self, *keys: str) -> bool:
        return self.suppressed or any(key in self.suppression_keys for key in keys)


_ACTIVE_CONTEXT: ContextVar[ActiveContext | None] = ContextVar(
    "autobench_protocol_context",
    default=None,
)


def get_context() -> ActiveContext | None:
    return _ACTIVE_CONTEXT.get()


def attach_context(context: ActiveContext | None) -> Token[ActiveContext | None]:
    return _ACTIVE_CONTEXT.set(context)


def reset_context(token: Token[ActiveContext | None]) -> None:
    _ACTIVE_CONTEXT.reset(token)


@contextmanager
def use_context(context: ActiveContext) -> Iterator[ActiveContext]:
    token = attach_context(context)
    try:
        yield context
    finally:
        reset_context(token)


@contextmanager
def suppress_instrumentation(*keys: str) -> Iterator[None]:
    active = get_context()
    if active is None:
        yield
        return
    with use_context(active.suppress(*keys)):
        yield


def capture_context() -> Context:
    return copy_context()


def bind_context(
    callback: Callable[P, R],
    context: Context | None = None,
) -> Callable[P, R]:
    captured = capture_context() if context is None else context

    @wraps(callback)
    def bound(*args: P.args, **kwargs: P.kwargs) -> R:
        return captured.copy().run(callback, *args, **kwargs)

    return bound


__all__ = (
    "ActiveContext",
    "attach_context",
    "bind_context",
    "capture_context",
    "get_context",
    "reset_context",
    "suppress_instrumentation",
    "use_context",
)
