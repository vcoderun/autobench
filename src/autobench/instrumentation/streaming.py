from __future__ import annotations

from collections.abc import AsyncGenerator, AsyncIterator, Generator, Iterator
from contextlib import AbstractAsyncContextManager, AbstractContextManager
from types import TracebackType
from typing import Any, Protocol, runtime_checkable

from autobench.protocol.signals import EndReason


class StreamLifecycle(Protocol):
    def resume(self) -> None: ...

    def suspend(self) -> None: ...

    def observe(self, item: Any) -> None: ...

    def finish(
        self,
        *,
        result: Any = None,
        error: BaseException | None = None,
        reason: EndReason = EndReason.COMPLETED,
        partial: bool = False,
    ) -> None: ...


@runtime_checkable
class Closable(Protocol):
    def close(self) -> None: ...


@runtime_checkable
class AsyncClosable(Protocol):
    async def aclose(self) -> None: ...


def _throw_exception(
    typ: type[BaseException],
    val: BaseException | None,
    traceback: TracebackType | None,
) -> BaseException:
    if val is None:
        exception = typ()
    elif isinstance(val, typ):
        exception = val
    else:
        exception = typ(val)
    return exception if traceback is None else exception.with_traceback(traceback)


class SyncIteratorProxy(Iterator[Any]):
    def __init__(self, stream: Iterator[Any], lifecycle: StreamLifecycle) -> None:
        self._stream = stream
        self._lifecycle = lifecycle

    def __iter__(self) -> SyncIteratorProxy:
        return self

    def __getattr__(self, name: str) -> Any:
        return getattr(self._stream, name)

    def __next__(self) -> Any:
        self._lifecycle.resume()
        try:
            item = next(self._stream)
        except StopIteration as stop:
            self._lifecycle.finish(result=stop.value)
            raise
        except BaseException as exc:
            self._lifecycle.finish(error=exc)
            raise
        self._lifecycle.observe(item)
        self._lifecycle.suspend()
        return item

    def close(self) -> None:
        self._lifecycle.resume()
        try:
            if isinstance(self._stream, Closable):
                self._stream.close()
        except BaseException as exc:
            self._lifecycle.finish(error=exc)
            raise
        self.cancel()

    def cancel(self) -> None:
        self._lifecycle.finish(reason=EndReason.CANCELLED, partial=True)


class SyncGeneratorProxy(Iterator[Any]):
    def __init__(self, stream: Generator[Any, Any, Any], lifecycle: StreamLifecycle) -> None:
        self._stream = stream
        self._lifecycle = lifecycle

    def __iter__(self) -> SyncGeneratorProxy:
        return self

    def __getattr__(self, name: str) -> Any:
        return getattr(self._stream, name)

    def __next__(self) -> Any:
        return self.send(None)

    def send(self, value: Any) -> Any:
        self._lifecycle.resume()
        try:
            item = self._stream.send(value)
        except StopIteration as stop:
            self._lifecycle.finish(result=stop.value)
            raise
        except BaseException as exc:
            self._lifecycle.finish(error=exc)
            raise
        self._lifecycle.observe(item)
        self._lifecycle.suspend()
        return item

    def throw(
        self,
        typ: type[BaseException],
        val: BaseException | None = None,
        tb: TracebackType | None = None,
    ) -> Any:
        self._lifecycle.resume()
        try:
            item = self._stream.throw(_throw_exception(typ, val, tb))
        except StopIteration as stop:
            self._lifecycle.finish(result=stop.value)
            raise
        except BaseException as exc:
            self._lifecycle.finish(error=exc)
            raise
        self._lifecycle.observe(item)
        self._lifecycle.suspend()
        return item

    def close(self) -> None:
        self._lifecycle.resume()
        try:
            self._stream.close()
        except BaseException as exc:
            self._lifecycle.finish(error=exc)
            raise
        self._lifecycle.finish(reason=EndReason.CANCELLED, partial=True)


class SyncIteratorContextProxy(SyncIteratorProxy, AbstractContextManager[Any]):
    def __init__(
        self,
        stream: Iterator[Any],
        context: AbstractContextManager[Any],
        lifecycle: StreamLifecycle,
    ) -> None:
        super().__init__(stream, lifecycle)
        self._context = context

    def __enter__(self) -> Any:
        self._lifecycle.resume()
        try:
            entered = self._context.__enter__()
        except BaseException as exc:
            self._lifecycle.finish(error=exc)
            raise
        if entered is self._stream:
            return self
        deferred = wrap_deferred_result(entered, self._lifecycle)
        return entered if deferred is None else deferred

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> bool | None:
        try:
            suppressed = self._context.__exit__(exc_type, exc_value, traceback)
        except BaseException as exc:
            self._lifecycle.finish(error=exc)
            raise
        if exc_value is not None and not suppressed:
            self._lifecycle.finish(error=exc_value)
        else:
            self._lifecycle.finish()
        return suppressed


class AsyncIteratorProxy(AsyncIterator[Any]):
    def __init__(self, stream: AsyncIterator[Any], lifecycle: StreamLifecycle) -> None:
        self._stream = stream
        self._lifecycle = lifecycle

    def __aiter__(self) -> AsyncIteratorProxy:
        return self

    def __getattr__(self, name: str) -> Any:
        return getattr(self._stream, name)

    async def __anext__(self) -> Any:
        self._lifecycle.resume()
        try:
            item = await anext(self._stream)
        except StopAsyncIteration:
            self._lifecycle.finish()
            raise
        except BaseException as exc:
            self._lifecycle.finish(error=exc)
            raise
        self._lifecycle.observe(item)
        self._lifecycle.suspend()
        return item

    async def aclose(self) -> None:
        self._lifecycle.resume()
        try:
            if isinstance(self._stream, AsyncClosable):
                await self._stream.aclose()
        except BaseException as exc:
            self._lifecycle.finish(error=exc)
            raise
        self.cancel()

    def cancel(self) -> None:
        self._lifecycle.finish(reason=EndReason.CANCELLED, partial=True)


class AsyncGeneratorProxy(AsyncIterator[Any]):
    def __init__(self, stream: AsyncGenerator[Any, Any], lifecycle: StreamLifecycle) -> None:
        self._stream = stream
        self._lifecycle = lifecycle

    def __aiter__(self) -> AsyncGeneratorProxy:
        return self

    def __getattr__(self, name: str) -> Any:
        return getattr(self._stream, name)

    async def __anext__(self) -> Any:
        return await self.asend(None)

    async def asend(self, value: Any) -> Any:
        self._lifecycle.resume()
        try:
            item = await self._stream.asend(value)
        except StopAsyncIteration:
            self._lifecycle.finish()
            raise
        except BaseException as exc:
            self._lifecycle.finish(error=exc)
            raise
        self._lifecycle.observe(item)
        self._lifecycle.suspend()
        return item

    async def athrow(
        self,
        typ: type[BaseException],
        val: BaseException | None = None,
        tb: TracebackType | None = None,
    ) -> Any:
        self._lifecycle.resume()
        try:
            item = await self._stream.athrow(_throw_exception(typ, val, tb))
        except StopAsyncIteration:
            self._lifecycle.finish()
            raise
        except BaseException as exc:
            self._lifecycle.finish(error=exc)
            raise
        self._lifecycle.observe(item)
        self._lifecycle.suspend()
        return item

    async def aclose(self) -> None:
        self._lifecycle.resume()
        try:
            await self._stream.aclose()
        except BaseException as exc:
            self._lifecycle.finish(error=exc)
            raise
        self._lifecycle.finish(reason=EndReason.CANCELLED, partial=True)


class AsyncIteratorContextProxy(AsyncIteratorProxy, AbstractAsyncContextManager[Any]):
    def __init__(
        self,
        stream: AsyncIterator[Any],
        context: AbstractAsyncContextManager[Any],
        lifecycle: StreamLifecycle,
    ) -> None:
        super().__init__(stream, lifecycle)
        self._context = context

    async def __aenter__(self) -> Any:
        self._lifecycle.resume()
        try:
            entered = await self._context.__aenter__()
        except BaseException as exc:
            self._lifecycle.finish(error=exc)
            raise
        if entered is self._stream:
            return self
        deferred = wrap_deferred_result(entered, self._lifecycle)
        return entered if deferred is None else deferred

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> bool | None:
        try:
            suppressed = await self._context.__aexit__(exc_type, exc_value, traceback)
        except BaseException as exc:
            self._lifecycle.finish(error=exc)
            raise
        if exc_value is not None and not suppressed:
            self._lifecycle.finish(error=exc_value)
        else:
            self._lifecycle.finish()
        return suppressed


class SyncContextProxy(AbstractContextManager[Any]):
    def __init__(
        self,
        context: AbstractContextManager[Any],
        lifecycle: StreamLifecycle,
    ) -> None:
        self._context = context
        self._lifecycle = lifecycle

    def __getattr__(self, name: str) -> Any:
        return getattr(self._context, name)

    def __enter__(self) -> Any:
        self._lifecycle.resume()
        try:
            entered = self._context.__enter__()
        except BaseException as exc:
            self._lifecycle.finish(error=exc)
            raise
        deferred = wrap_deferred_result(entered, self._lifecycle)
        return entered if deferred is None else deferred

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> bool | None:
        try:
            suppressed = self._context.__exit__(exc_type, exc_value, traceback)
        except BaseException as exc:
            self._lifecycle.finish(error=exc)
            raise
        if exc_value is not None and not suppressed:
            self._lifecycle.finish(error=exc_value)
        else:
            self._lifecycle.finish()
        return suppressed


class AsyncContextProxy(AbstractAsyncContextManager[Any]):
    def __init__(
        self,
        context: AbstractAsyncContextManager[Any],
        lifecycle: StreamLifecycle,
    ) -> None:
        self._context = context
        self._lifecycle = lifecycle

    def __getattr__(self, name: str) -> Any:
        return getattr(self._context, name)

    async def __aenter__(self) -> Any:
        self._lifecycle.resume()
        try:
            entered = await self._context.__aenter__()
        except BaseException as exc:
            self._lifecycle.finish(error=exc)
            raise
        deferred = wrap_deferred_result(entered, self._lifecycle)
        return entered if deferred is None else deferred

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> bool | None:
        try:
            suppressed = await self._context.__aexit__(exc_type, exc_value, traceback)
        except BaseException as exc:
            self._lifecycle.finish(error=exc)
            raise
        if exc_value is not None and not suppressed:
            self._lifecycle.finish(error=exc_value)
        else:
            self._lifecycle.finish()
        return suppressed


def wrap_deferred_result(result: Any, lifecycle: StreamLifecycle) -> Any | None:
    if isinstance(result, Iterator) and isinstance(result, AbstractContextManager):
        lifecycle.suspend()
        return SyncIteratorContextProxy(result, result, lifecycle)
    if isinstance(result, AsyncIterator) and isinstance(result, AbstractAsyncContextManager):
        lifecycle.suspend()
        return AsyncIteratorContextProxy(result, result, lifecycle)
    if isinstance(result, AbstractContextManager):
        lifecycle.suspend()
        return SyncContextProxy(result, lifecycle)
    if isinstance(result, AbstractAsyncContextManager):
        lifecycle.suspend()
        return AsyncContextProxy(result, lifecycle)
    if isinstance(result, Generator):
        lifecycle.suspend()
        return SyncGeneratorProxy(result, lifecycle)
    if isinstance(result, Iterator):
        lifecycle.suspend()
        return SyncIteratorProxy(result, lifecycle)
    if isinstance(result, AsyncGenerator):
        lifecycle.suspend()
        return AsyncGeneratorProxy(result, lifecycle)
    if isinstance(result, AsyncIterator):
        lifecycle.suspend()
        return AsyncIteratorProxy(result, lifecycle)
    return None


__all__ = (
    "AsyncContextProxy",
    "AsyncGeneratorProxy",
    "AsyncIteratorContextProxy",
    "AsyncIteratorProxy",
    "StreamLifecycle",
    "SyncContextProxy",
    "SyncGeneratorProxy",
    "SyncIteratorContextProxy",
    "SyncIteratorProxy",
    "wrap_deferred_result",
)
