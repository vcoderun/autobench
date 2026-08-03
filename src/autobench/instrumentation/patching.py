from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from functools import wraps
from inspect import getattr_static, iscoroutinefunction, unwrap
from typing import Any, Literal, Protocol
from weakref import WeakKeyDictionary

from autobench.instrumentation.models import InstrumentationHandle, InstrumentCall
from autobench.instrumentation.streaming import StreamLifecycle, wrap_deferred_result
from autobench.protocol.context import get_context
from autobench.protocol.signals import EndReason

DescriptorKind = Literal["instance", "staticmethod", "classmethod"]


class InstrumentationConflictError(RuntimeError):
    """Raised when a patch target changed outside Autobench ownership."""


class CallLifecycle(StreamLifecycle, Protocol):
    pass


class CallHandler(Protocol):
    @property
    def suppression_keys(self) -> tuple[str, ...]: ...

    def begin(self, call: InstrumentCall) -> CallLifecycle | None: ...

    def diagnose(self, stage: str, error: Exception) -> None: ...

    def close(self) -> None: ...


@dataclass(slots=True)
class PatchDiagnostic:
    owner: str
    target: str
    message: str


@dataclass(slots=True)
class _PatchRegistration:
    owner: str
    handler: CallHandler
    references: int = 1


@dataclass(slots=True)
class _PatchState:
    attribute: str
    original_descriptor: Any
    installed_descriptor: Any
    was_local: bool
    registrations: dict[str, _PatchRegistration]


@dataclass(slots=True)
class _ActiveCall:
    handler: CallHandler
    lifecycle: CallLifecycle


class _LifecycleGroup:
    def __init__(self, calls: list[_ActiveCall]) -> None:
        self._calls = calls
        self._active = True
        self._finished = False

    def resume(self) -> None:
        if self._active or self._finished:
            return
        for active_call in self._calls:
            try:
                active_call.lifecycle.resume()
            except Exception as exc:
                active_call.handler.diagnose("resume", exc)
        self._active = True

    def suspend(self) -> None:
        for active_call in reversed(self._calls):
            try:
                active_call.lifecycle.suspend()
            except Exception as exc:
                active_call.handler.diagnose("suspend", exc)
        self._active = False

    def observe(self, item: Any) -> None:
        for active_call in self._calls:
            try:
                active_call.lifecycle.observe(item)
            except Exception as exc:
                active_call.handler.diagnose("stream_item", exc)

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
        self.resume()
        for active_call in reversed(self._calls):
            try:
                active_call.lifecycle.finish(
                    result=result,
                    error=error,
                    reason=reason,
                    partial=partial,
                )
            except Exception as exc:
                active_call.handler.diagnose("finish", exc)
                try:
                    active_call.lifecycle.suspend()
                except Exception as suspend_exc:
                    active_call.handler.diagnose("suspend", suspend_exc)
        self._active = False
        self._finished = True


class PatchManager:
    def __init__(self) -> None:
        self._states: WeakKeyDictionary[type[Any], dict[str, _PatchState]] = WeakKeyDictionary()
        self._diagnostics: list[PatchDiagnostic] = []

    @property
    def diagnostics(self) -> tuple[PatchDiagnostic, ...]:
        return tuple(self._diagnostics)

    def patch_method(
        self,
        target: type[Any],
        attribute: str,
        *,
        owner: str,
        handler: CallHandler,
        expected_descriptor: Any = None,
    ) -> InstrumentationHandle:
        target_states = self._states.setdefault(target, {})
        state = target_states.get(attribute)
        if state is not None:
            self._require_owned_descriptor(target, state)
            registration = state.registrations.get(owner)
            if registration is None:
                state.registrations[owner] = _PatchRegistration(owner=owner, handler=handler)
            else:
                registration.references += 1
            return InstrumentationHandle(lambda: self._release(target, attribute, owner))

        descriptor = getattr_static(target, attribute)
        if expected_descriptor is not None and descriptor is not expected_descriptor:
            raise InstrumentationConflictError(
                f"{target.__qualname__}.{attribute} does not match the expected descriptor"
            )
        descriptor_kind, original_callable = _descriptor_callable(target, attribute, descriptor)
        registrations = {owner: _PatchRegistration(owner=owner, handler=handler)}
        wrapped = _wrap_callable(original_callable, descriptor_kind, registrations)
        installed_descriptor = _bind_descriptor(wrapped, descriptor_kind)
        state = _PatchState(
            attribute=attribute,
            original_descriptor=descriptor,
            installed_descriptor=installed_descriptor,
            was_local=attribute in target.__dict__,
            registrations=registrations,
        )
        setattr(target, attribute, installed_descriptor)
        target_states[attribute] = state
        return InstrumentationHandle(lambda: self._release(target, attribute, owner))

    def close(self) -> None:
        while self._states:
            target, target_states = next(iter(self._states.items()))
            attribute, state = next(iter(target_states.items()))
            owner, registration = next(iter(state.registrations.items()))
            registration.references = 1
            self._release(target, attribute, owner)

    def _release(self, target: type[Any], attribute: str, owner: str) -> None:
        state = self._states.get(target, {}).get(attribute)
        if state is None:
            return
        registration = state.registrations.get(owner)
        if registration is None:
            return
        registration.references -= 1
        if registration.references > 0:
            return
        del state.registrations[owner]
        if not state.registrations:
            target_states = self._states[target]
            del target_states[attribute]
            if not target_states:
                del self._states[target]
            current = getattr_static(target, attribute)
            if current is not state.installed_descriptor:
                self._diagnostics.append(
                    PatchDiagnostic(
                        owner=owner,
                        target=f"{target.__module__}.{target.__qualname__}.{attribute}",
                        message="target descriptor changed after Autobench installed its wrapper",
                    )
                )
            elif state.was_local:
                setattr(target, attribute, state.original_descriptor)
            else:
                delattr(target, attribute)
        try:
            registration.handler.close()
        except Exception as exc:
            registration.handler.diagnose("close", exc)

    def _require_owned_descriptor(self, target: type[Any], state: _PatchState) -> None:
        current = getattr_static(target, state.attribute)
        if current is not state.installed_descriptor:
            raise InstrumentationConflictError(
                f"{target.__qualname__}.{state.attribute} changed after Autobench "
                "installed its wrapper"
            )


def _descriptor_callable(
    target: type[Any],
    attribute: str,
    descriptor: Any,
) -> tuple[DescriptorKind, Callable[..., Any]]:
    if isinstance(descriptor, property):
        raise TypeError("property instrumentation is not supported")
    if isinstance(descriptor, staticmethod):
        return "staticmethod", descriptor.__func__
    if isinstance(descriptor, classmethod):
        return "classmethod", descriptor.__func__
    if not isinstance(descriptor, Callable):
        raise TypeError(f"{target.__name__}.{attribute} is not callable")
    return "instance", descriptor


def _wrap_callable(
    original: Callable[..., Any],
    descriptor_kind: DescriptorKind,
    registrations: dict[str, _PatchRegistration],
) -> Callable[..., Any]:
    if iscoroutinefunction(original) or iscoroutinefunction(unwrap(original)):

        @wraps(original)
        async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
            group = _begin_calls(registrations, descriptor_kind, args, kwargs)
            try:
                result = await original(*args, **kwargs)
            except BaseException as exc:
                if group is not None:
                    group.finish(error=exc)
                raise
            if group is None:
                return result
            deferred = wrap_deferred_result(result, group)
            if deferred is not None:
                return deferred
            group.finish(result=result)
            return result

        return async_wrapper

    @wraps(original)
    def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
        group = _begin_calls(registrations, descriptor_kind, args, kwargs)
        try:
            result = original(*args, **kwargs)
        except BaseException as exc:
            if group is not None:
                group.finish(error=exc)
            raise
        if group is None:
            return result
        deferred = wrap_deferred_result(result, group)
        if deferred is not None:
            return deferred
        group.finish(result=result)
        return result

    return sync_wrapper


def _begin_calls(
    registrations: dict[str, _PatchRegistration],
    descriptor_kind: DescriptorKind,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> _LifecycleGroup | None:
    active_context = get_context()
    calls: list[_ActiveCall] = []
    for registration in reversed(tuple(registrations.values())):
        if active_context is not None and active_context.is_suppressed(
            *registration.handler.suppression_keys
        ):
            continue
        call = InstrumentCall(
            instance=_bound_instance(descriptor_kind, args),
            args=_call_args(descriptor_kind, args),
            kwargs=dict(kwargs),
        )
        try:
            lifecycle = registration.handler.begin(call)
        except Exception as exc:
            registration.handler.diagnose("begin", exc)
            continue
        if lifecycle is not None:
            calls.append(_ActiveCall(handler=registration.handler, lifecycle=lifecycle))
    if not calls:
        return None
    return _LifecycleGroup(calls)


def _bind_descriptor(
    wrapped: Callable[..., Any],
    descriptor_kind: DescriptorKind,
) -> Any:
    if descriptor_kind == "staticmethod":
        return staticmethod(wrapped)
    if descriptor_kind == "classmethod":
        return classmethod(wrapped)
    return wrapped


def _bound_instance(descriptor_kind: DescriptorKind, args: tuple[Any, ...]) -> Any | None:
    if descriptor_kind in {"instance", "classmethod"} and args:
        return args[0]
    return None


def _call_args(descriptor_kind: DescriptorKind, args: tuple[Any, ...]) -> tuple[Any, ...]:
    if descriptor_kind in {"instance", "classmethod"} and args:
        return args[1:]
    return args


__all__ = (
    "CallHandler",
    "CallLifecycle",
    "InstrumentationConflictError",
    "PatchDiagnostic",
    "PatchManager",
)
