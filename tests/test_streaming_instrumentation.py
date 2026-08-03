from __future__ import annotations

from collections.abc import AsyncGenerator, AsyncIterator, Generator, Iterator
from contextlib import (
    AbstractAsyncContextManager,
    AbstractContextManager,
    asynccontextmanager,
    contextmanager,
)
from types import TracebackType
from typing import Any

import pytest

from autobench import (
    Case,
    InstrumentMetricSpec,
    RunContext,
    Variant,
    instrument_method,
)
from autobench.instrumentation.streaming import (
    AsyncContextProxy,
    AsyncGeneratorProxy,
    AsyncIteratorContextProxy,
    AsyncIteratorProxy,
    SyncContextProxy,
    SyncGeneratorProxy,
    SyncIteratorContextProxy,
    SyncIteratorProxy,
    wrap_deferred_result,
)
from autobench.protocol.signals import EndReason
from autobench.runtime.instrumentation import reset_active_run_context, set_active_run_context


def test_sync_generator_records_stream_completion_send_and_true_lifecycle() -> None:
    class Worker:
        def stream(self) -> Generator[int, int, str]:
            received = yield 1
            yield received
            return "done"

    ctx = _run_context()
    handle = instrument_method(
        Worker,
        "stream",
        span="worker.stream",
        metrics=[
            InstrumentMetricSpec(
                name="items",
                value_factory=lambda call: call.stream_item_count,
            ),
            InstrumentMetricSpec(
                name="last_item",
                value_factory=lambda call: call.last_stream_item,
            ),
            InstrumentMetricSpec(name="return_value", value_path="result"),
        ],
    )
    token = set_active_run_context(ctx)
    try:
        stream = Worker().stream()
        assert ctx.spans[0].ended_at is None
        assert next(stream) == 1
        assert ctx.spans[0].ended_at is None
        assert stream.send(4) == 4
        with pytest.raises(StopIteration) as stop:
            next(stream)
    finally:
        reset_active_run_context(token)
        handle.close()

    assert stop.value.value == "done"
    assert [observation.value for observation in ctx.observations] == [2, 4, "done"]
    assert ctx.spans[0].ended_at is not None
    assert ctx.spans[0].duration_seconds is not None
    trace_span = next(span for span in ctx.trace.spans if span.operation == "worker.stream")
    assert trace_span.scope.instrumentor_name == "autobench.instrument_method"
    assert trace_span.end_reason is EndReason.COMPLETED
    assert trace_span.attributes["stream_item_count"] == 2


def test_sync_generator_throw_and_early_close_preserve_native_behavior() -> None:
    class Worker:
        def stream(self) -> Generator[str, None, None]:
            try:
                yield "ready"
            except ValueError:
                yield "recovered"
            yield "later"

    ctx = _run_context()
    handle = instrument_method(Worker, "stream", span="worker.stream")
    token = set_active_run_context(ctx)
    try:
        stream = Worker().stream()
        assert next(stream) == "ready"
        assert stream.throw(ValueError, ValueError("retry")) == "recovered"
        stream.close()
        stream.close()
    finally:
        reset_active_run_context(token)
        handle.close()

    trace_span = next(span for span in ctx.trace.spans if span.operation == "worker.stream")
    assert trace_span.end_reason is EndReason.CANCELLED
    assert trace_span.partial is True


def test_iterator_is_lazy_closable_and_preserves_stream_error_identity() -> None:
    failure = RuntimeError("stream failed")

    class Values(Iterator[int]):
        def __init__(self) -> None:
            self.index = 0
            self.closed = False

        def __iter__(self) -> Values:
            return self

        def __next__(self) -> int:
            self.index += 1
            if self.index == 1:
                return 1
            raise failure

        def close(self) -> None:
            self.closed = True

    class Worker:
        def stream(self) -> Values:
            return Values()

    ctx = _run_context()
    handle = instrument_method(Worker, "stream", span="worker.stream")
    token = set_active_run_context(ctx)
    try:
        stream = Worker().stream()
        assert stream.index == 0
        assert next(stream) == 1
        with pytest.raises(RuntimeError) as caught:
            next(stream)
    finally:
        reset_active_run_context(token)
        handle.close()

    assert caught.value is failure
    assert ctx.spans[0].error is not None
    assert ctx.spans[0].error.message == "stream failed"


async def test_async_generator_records_asend_athrow_completion_and_close() -> None:
    class Worker:
        async def stream(self) -> AsyncGenerator[str, str]:
            try:
                received = yield "ready"
                yield received
            except ValueError:
                yield "recovered"

    ctx = _run_context()
    handle = instrument_method(
        Worker,
        "stream",
        span="worker.async_stream",
        metrics=[
            InstrumentMetricSpec(
                name="items",
                value_factory=lambda call: call.stream_item_count,
            )
        ],
    )
    token = set_active_run_context(ctx)
    try:
        stream = Worker().stream()
        assert await anext(stream) == "ready"
        assert await stream.asend("sent") == "sent"
        with pytest.raises(StopAsyncIteration):
            await anext(stream)

        retry_stream = Worker().stream()
        assert await anext(retry_stream) == "ready"
        assert await retry_stream.athrow(ValueError, ValueError("retry")) == "recovered"
        await retry_stream.aclose()
    finally:
        reset_active_run_context(token)
        handle.close()

    assert [observation.value for observation in ctx.observations] == [2, 2]
    trace_spans = [span for span in ctx.trace.spans if span.operation == "worker.async_stream"]
    assert [span.end_reason for span in trace_spans] == [
        EndReason.COMPLETED,
        EndReason.CANCELLED,
    ]


async def test_async_iterator_aclose_and_error_are_observed_without_eager_consumption() -> None:
    failure = RuntimeError("async stream failed")

    class Values(AsyncIterator[int]):
        def __init__(self, *, fail: bool) -> None:
            self.index = 0
            self.fail = fail
            self.closed = False

        def __aiter__(self) -> Values:
            return self

        async def __anext__(self) -> int:
            self.index += 1
            if self.index == 1:
                return 1
            if self.fail:
                raise failure
            raise StopAsyncIteration

        async def aclose(self) -> None:
            self.closed = True

    class Worker:
        async def stream(self, *, fail: bool) -> Values:
            return Values(fail=fail)

    ctx = _run_context()
    handle = instrument_method(Worker, "stream", span="worker.async_iterator")
    token = set_active_run_context(ctx)
    try:
        stream = await Worker().stream(fail=False)
        assert stream.index == 0
        assert await anext(stream) == 1
        await stream.aclose()
        assert stream.closed is True

        failing = await Worker().stream(fail=True)
        assert await anext(failing) == 1
        with pytest.raises(RuntimeError) as caught:
            await anext(failing)
    finally:
        reset_active_run_context(token)
        handle.close()

    assert caught.value is failure
    trace_spans = [span for span in ctx.trace.spans if span.operation == "worker.async_iterator"]
    assert [span.end_reason for span in trace_spans] == [
        EndReason.CANCELLED,
        EndReason.FAILED,
    ]


def test_sync_context_manager_finishes_on_exit_and_respects_suppression() -> None:
    class Worker:
        @contextmanager
        def session(self, *, suppress: bool = False) -> Iterator[str]:
            try:
                yield "session"
            except ValueError:
                if not suppress:
                    raise

    ctx = _run_context()
    handle = instrument_method(Worker, "session", span="worker.session")
    token = set_active_run_context(ctx)
    try:
        context = Worker().session()
        assert ctx.spans[0].ended_at is None
        with context as value:
            assert value == "session"
        with Worker().session(suppress=True):
            raise ValueError("handled")
        error = ValueError("unhandled")
        with pytest.raises(ValueError) as caught, Worker().session():
            raise error
    finally:
        reset_active_run_context(token)
        handle.close()

    assert caught.value is error
    trace_spans = [span for span in ctx.trace.spans if span.operation == "worker.session"]
    assert [span.end_reason for span in trace_spans] == [
        EndReason.COMPLETED,
        EndReason.COMPLETED,
        EndReason.FAILED,
    ]


def test_sync_context_manager_observes_an_entered_iterator() -> None:
    @contextmanager
    def stream() -> Iterator[Iterator[int]]:
        yield iter((1, 2))

    lifecycle = LifecycleRecorder()
    with SyncContextProxy(stream(), lifecycle) as values:
        assert list(values) == [1, 2]

    assert "item:1" in lifecycle.events
    assert "item:2" in lifecycle.events


async def test_async_context_manager_finishes_success_and_entry_failure() -> None:
    class Worker:
        @asynccontextmanager
        async def session(self, *, fail_entry: bool = False) -> AsyncIterator[str]:
            if fail_entry:
                raise RuntimeError("entry failed")
            yield "session"

    ctx = _run_context()
    handle = instrument_method(Worker, "session", span="worker.async_session")
    token = set_active_run_context(ctx)
    try:
        async with Worker().session() as value:
            assert value == "session"
        with pytest.raises(RuntimeError, match="entry failed"):
            async with Worker().session(fail_entry=True):
                pass
    finally:
        reset_active_run_context(token)
        handle.close()

    trace_spans = [span for span in ctx.trace.spans if span.operation == "worker.async_session"]
    assert [span.end_reason for span in trace_spans] == [
        EndReason.COMPLETED,
        EndReason.FAILED,
    ]


async def test_async_context_manager_observes_an_entered_async_iterator() -> None:
    async def values() -> AsyncIterator[int]:
        yield 1
        yield 2

    @asynccontextmanager
    async def stream() -> AsyncIterator[AsyncIterator[int]]:
        yield values()

    lifecycle = LifecycleRecorder()
    async with AsyncContextProxy(stream(), lifecycle) as entered:
        assert [value async for value in entered] == [1, 2]

    assert "item:1" in lifecycle.events
    assert "item:2" in lifecycle.events


def test_closing_instrumentation_finalizes_never_iterated_stream_as_abandoned() -> None:
    class Worker:
        def stream(self) -> Generator[int, None, None]:
            yield 1

    ctx = _run_context()
    handle = instrument_method(Worker, "stream", span="worker.never_iterated")
    token = set_active_run_context(ctx)
    try:
        stream = Worker().stream()
        assert ctx.spans[0].ended_at is None
        handle.close()
    finally:
        reset_active_run_context(token)

    assert ctx.spans[0].ended_at is not None
    trace_span = next(span for span in ctx.trace.spans if span.operation == "worker.never_iterated")
    assert trace_span.end_reason is EndReason.ABANDONED
    assert trace_span.partial is True
    assert next(stream) == 1
    with pytest.raises(StopIteration):
        next(stream)


def test_stream_instrumentation_can_extract_without_creating_a_span() -> None:
    class Worker:
        def stream(self) -> Generator[int, None, str]:
            yield 1
            return "done"

    ctx = _run_context()
    handle = instrument_method(
        Worker,
        "stream",
        metrics=[InstrumentMetricSpec(name="result", value_path="result")],
    )
    token = set_active_run_context(ctx)
    try:
        assert list(Worker().stream()) == [1]
    finally:
        reset_active_run_context(token)
        handle.close()

    assert ctx.spans == []
    assert ctx.observations[0].value == "done"


def test_finalized_run_does_not_let_late_stream_cleanup_break_application_code() -> None:
    class Worker:
        def stream(self) -> Generator[int, None, None]:
            yield 1

    ctx = _run_context()
    handle = instrument_method(
        Worker,
        "stream",
        span="worker.late_stream",
        metrics=[InstrumentMetricSpec(name="items", value_path="stream_item_count")],
    )
    token = set_active_run_context(ctx)
    try:
        stream = Worker().stream()
        ctx.finalize()
        handle.close()
    finally:
        reset_active_run_context(token)

    assert list(stream) == [1]


class LifecycleRecorder:
    def __init__(self) -> None:
        self.events: list[str] = []
        self.results: list[Any] = []
        self.errors: list[BaseException] = []
        self.reasons: list[tuple[EndReason, bool]] = []

    def resume(self) -> None:
        self.events.append("resume")

    def suspend(self) -> None:
        self.events.append("suspend")

    def observe(self, item: Any) -> None:
        self.events.append(f"item:{item}")

    def finish(
        self,
        *,
        result: Any = None,
        error: BaseException | None = None,
        reason: EndReason = EndReason.COMPLETED,
        partial: bool = False,
    ) -> None:
        self.events.append("finish")
        self.results.append(result)
        if error is not None:
            self.errors.append(error)
        self.reasons.append((reason, partial))


def test_sync_iterator_proxy_preserves_iteration_attributes_and_close_contracts() -> None:
    class EmptyIterator(Iterator[int]):
        label = "empty"

        def __iter__(self) -> EmptyIterator:
            return self

        def __next__(self) -> int:
            raise StopIteration("done")

    lifecycle = LifecycleRecorder()
    proxy = SyncIteratorProxy(EmptyIterator(), lifecycle)
    assert iter(proxy) is proxy
    assert proxy.label == "empty"
    with pytest.raises(StopIteration) as stopped:
        next(proxy)
    assert stopped.value.value == "done"
    assert lifecycle.results == ["done"]

    plain_lifecycle = LifecycleRecorder()
    SyncIteratorProxy(iter(()), plain_lifecycle).close()
    assert plain_lifecycle.reasons == [(EndReason.CANCELLED, True)]

    class ClosingIterator(EmptyIterator):
        def __init__(self, error: RuntimeError | None = None) -> None:
            self.closed = False
            self.error = error

        def close(self) -> None:
            self.closed = True
            if self.error is not None:
                raise self.error

    closing = ClosingIterator()
    closing_lifecycle = LifecycleRecorder()
    closing_proxy = SyncIteratorProxy(closing, closing_lifecycle)
    closing_proxy.close()
    assert closing.closed is True

    failure = RuntimeError("close failed")
    broken_lifecycle = LifecycleRecorder()
    with pytest.raises(RuntimeError) as caught:
        SyncIteratorProxy(ClosingIterator(failure), broken_lifecycle).close()
    assert caught.value is failure
    assert broken_lifecycle.errors == [failure]


def test_sync_generator_proxy_preserves_throw_send_and_close_failures() -> None:
    send_failure = RuntimeError("send failed")

    def failing_send() -> Generator[int, None, None]:
        yield 1
        raise send_failure

    send_lifecycle = LifecycleRecorder()
    send_proxy = SyncGeneratorProxy(failing_send(), send_lifecycle)
    assert iter(send_proxy) is send_proxy
    assert send_proxy.gi_code.co_name == "failing_send"
    assert next(send_proxy) == 1
    with pytest.raises(RuntimeError) as caught_send:
        next(send_proxy)
    assert caught_send.value is send_failure
    assert send_lifecycle.errors == [send_failure]

    def recovering() -> Generator[str, None, str]:
        try:
            yield "ready"
        except ValueError as exc:
            if str(exc) == "finish":
                return "done"
            yield "recovered"
        return "complete"

    no_value = SyncGeneratorProxy(recovering(), LifecycleRecorder())
    assert next(no_value) == "ready"
    assert no_value.throw(ValueError) == "recovered"

    def no_handler() -> Generator[str, None, None]:
        yield "ready"

    wrapped_value = SyncGeneratorProxy(no_handler(), LifecycleRecorder())
    assert next(wrapped_value) == "ready"
    wrapped_error = TypeError("wrapped")
    with pytest.raises(ValueError) as wrapped:
        wrapped_value.throw(ValueError, wrapped_error)
    assert wrapped.value.args == (wrapped_error,)

    with_traceback = SyncGeneratorProxy(recovering(), LifecycleRecorder())
    assert next(with_traceback) == "ready"
    error = ValueError("retry")
    try:
        raise error
    except ValueError as captured:
        traceback: TracebackType | None = captured.__traceback__
    assert traceback is not None
    assert with_traceback.throw(ValueError, error, traceback) == "recovered"

    finishing_lifecycle = LifecycleRecorder()
    finishing = SyncGeneratorProxy(recovering(), finishing_lifecycle)
    assert next(finishing) == "ready"
    with pytest.raises(StopIteration) as stopped:
        finishing.throw(ValueError, ValueError("finish"))
    assert stopped.value.value == "done"
    assert finishing_lifecycle.results[-1] == "done"

    throw_failure = ValueError("unhandled")

    def unhandled() -> Generator[str, None, None]:
        yield "ready"

    error_lifecycle = LifecycleRecorder()
    error_proxy = SyncGeneratorProxy(unhandled(), error_lifecycle)
    assert next(error_proxy) == "ready"
    with pytest.raises(ValueError) as caught_throw:
        error_proxy.throw(ValueError, throw_failure)
    assert caught_throw.value is throw_failure

    close_failure = RuntimeError("generator close failed")

    def broken_close() -> Generator[str, None, None]:
        try:
            yield "ready"
        except GeneratorExit:
            raise close_failure from None

    close_lifecycle = LifecycleRecorder()
    close_proxy = SyncGeneratorProxy(broken_close(), close_lifecycle)
    assert next(close_proxy) == "ready"
    with pytest.raises(RuntimeError) as caught_close:
        close_proxy.close()
    assert caught_close.value is close_failure
    assert close_lifecycle.errors == [close_failure]


async def test_async_iterator_proxy_preserves_exhaustion_and_close_contracts() -> None:
    class EmptyIterator(AsyncIterator[int]):
        label = "empty"

        def __aiter__(self) -> EmptyIterator:
            return self

        async def __anext__(self) -> int:
            raise StopAsyncIteration

    lifecycle = LifecycleRecorder()
    proxy = AsyncIteratorProxy(EmptyIterator(), lifecycle)
    assert aiter(proxy) is proxy
    assert proxy.label == "empty"
    with pytest.raises(StopAsyncIteration):
        await anext(proxy)
    assert lifecycle.reasons == [(EndReason.COMPLETED, False)]

    plain_lifecycle = LifecycleRecorder()
    await AsyncIteratorProxy(EmptyIterator(), plain_lifecycle).aclose()
    assert plain_lifecycle.reasons == [(EndReason.CANCELLED, True)]

    close_failure = RuntimeError("async close failed")

    class BrokenClose(EmptyIterator):
        async def aclose(self) -> None:
            raise close_failure

    broken_lifecycle = LifecycleRecorder()
    with pytest.raises(RuntimeError) as caught:
        await AsyncIteratorProxy(BrokenClose(), broken_lifecycle).aclose()
    assert caught.value is close_failure
    assert broken_lifecycle.errors == [close_failure]


async def test_async_generator_proxy_preserves_throw_send_and_close_failures() -> None:
    send_failure = RuntimeError("async send failed")

    async def failing_send() -> AsyncGenerator[int, None]:
        yield 1
        raise send_failure

    send_lifecycle = LifecycleRecorder()
    send_proxy = AsyncGeneratorProxy(failing_send(), send_lifecycle)
    assert aiter(send_proxy) is send_proxy
    assert send_proxy.ag_code.co_name == "failing_send"
    assert await anext(send_proxy) == 1
    with pytest.raises(RuntimeError) as caught_send:
        await anext(send_proxy)
    assert caught_send.value is send_failure

    async def recovering() -> AsyncGenerator[str, None]:
        try:
            yield "ready"
        except ValueError as exc:
            if str(exc) == "finish":
                return
            yield "recovered"

    no_value = AsyncGeneratorProxy(recovering(), LifecycleRecorder())
    assert await anext(no_value) == "ready"
    assert await no_value.athrow(ValueError) == "recovered"

    async def no_handler() -> AsyncGenerator[str, None]:
        yield "ready"

    wrapped_value = AsyncGeneratorProxy(no_handler(), LifecycleRecorder())
    assert await anext(wrapped_value) == "ready"
    wrapped_error = TypeError("wrapped")
    with pytest.raises(ValueError) as wrapped:
        await wrapped_value.athrow(ValueError, wrapped_error)
    assert wrapped.value.args == (wrapped_error,)

    with_traceback = AsyncGeneratorProxy(recovering(), LifecycleRecorder())
    assert await anext(with_traceback) == "ready"
    error = ValueError("retry")
    try:
        raise error
    except ValueError as captured:
        traceback: TracebackType | None = captured.__traceback__
    assert traceback is not None
    assert await with_traceback.athrow(ValueError, error, traceback) == "recovered"

    finishing_lifecycle = LifecycleRecorder()
    finishing = AsyncGeneratorProxy(recovering(), finishing_lifecycle)
    assert await anext(finishing) == "ready"
    with pytest.raises(StopAsyncIteration):
        await finishing.athrow(ValueError, ValueError("finish"))
    assert finishing_lifecycle.reasons[-1] == (EndReason.COMPLETED, False)

    throw_failure = ValueError("unhandled")

    async def unhandled() -> AsyncGenerator[str, None]:
        yield "ready"

    error_lifecycle = LifecycleRecorder()
    error_proxy = AsyncGeneratorProxy(unhandled(), error_lifecycle)
    assert await anext(error_proxy) == "ready"
    with pytest.raises(ValueError) as caught_throw:
        await error_proxy.athrow(ValueError, throw_failure)
    assert caught_throw.value is throw_failure

    close_failure = RuntimeError("async generator close failed")

    async def broken_close() -> AsyncGenerator[str, None]:
        try:
            yield "ready"
        except GeneratorExit:
            raise close_failure from None

    close_lifecycle = LifecycleRecorder()
    close_proxy = AsyncGeneratorProxy(broken_close(), close_lifecycle)
    assert await anext(close_proxy) == "ready"
    with pytest.raises(RuntimeError) as caught_close:
        await close_proxy.aclose()
    assert caught_close.value is close_failure
    assert close_lifecycle.errors == [close_failure]


def test_sync_context_proxy_preserves_attributes_and_context_failures() -> None:
    entry_failure = RuntimeError("entry failed")
    exit_failure = RuntimeError("exit failed")

    class Context(AbstractContextManager[str]):
        label = "context"

        def __init__(self, *, fail_entry: bool = False, fail_exit: bool = False) -> None:
            self._fail_entry = fail_entry
            self._fail_exit = fail_exit

        def __enter__(self) -> str:
            if self._fail_entry:
                raise entry_failure
            return "value"

        def __exit__(
            self,
            exc_type: type[BaseException] | None,
            exc_value: BaseException | None,
            traceback: TracebackType | None,
        ) -> bool | None:
            if self._fail_exit:
                raise exit_failure
            return None

    lifecycle = LifecycleRecorder()
    proxy = SyncContextProxy(Context(), lifecycle)
    assert proxy.label == "context"
    with proxy as value:
        assert value == "value"

    entry_lifecycle = LifecycleRecorder()
    with (
        pytest.raises(RuntimeError) as caught_entry,
        SyncContextProxy(Context(fail_entry=True), entry_lifecycle),
    ):
        pass
    assert caught_entry.value is entry_failure
    assert entry_lifecycle.errors == [entry_failure]

    exit_lifecycle = LifecycleRecorder()
    with (
        pytest.raises(RuntimeError) as caught_exit,
        SyncContextProxy(Context(fail_exit=True), exit_lifecycle),
    ):
        pass
    assert caught_exit.value is exit_failure
    assert exit_lifecycle.errors == [exit_failure]


async def test_async_context_proxy_preserves_attributes_and_exit_failures() -> None:
    exit_failure = RuntimeError("async exit failed")

    class Context(AbstractAsyncContextManager[str]):
        label = "async-context"

        def __init__(self, *, fail_exit: bool = False) -> None:
            self._fail_exit = fail_exit

        async def __aenter__(self) -> str:
            return "value"

        async def __aexit__(
            self,
            exc_type: type[BaseException] | None,
            exc_value: BaseException | None,
            traceback: TracebackType | None,
        ) -> bool | None:
            if self._fail_exit:
                raise exit_failure
            return None

    lifecycle = LifecycleRecorder()
    proxy = AsyncContextProxy(Context(), lifecycle)
    assert proxy.label == "async-context"
    error = ValueError("body failed")
    with pytest.raises(ValueError) as caught_body:
        async with proxy:
            raise error
    assert caught_body.value is error
    assert lifecycle.errors == [error]

    exit_lifecycle = LifecycleRecorder()
    with pytest.raises(RuntimeError) as caught_exit:
        async with AsyncContextProxy(Context(fail_exit=True), exit_lifecycle):
            pass
    assert caught_exit.value is exit_failure
    assert exit_lifecycle.errors == [exit_failure]


def test_sync_iterator_context_proxy_preserves_entered_values_and_body_outcomes() -> None:
    class IteratorContext(Iterator[int], AbstractContextManager[Any]):
        def __init__(self, entered: Any = None, *, suppress: bool = False) -> None:
            self._values = iter((1, 2))
            self._entered = self if entered is None else entered
            self._suppress = suppress

        def __iter__(self) -> IteratorContext:
            return self

        def __next__(self) -> int:
            return next(self._values)

        def __enter__(self) -> Any:
            return self._entered

        def __exit__(
            self,
            exc_type: type[BaseException] | None,
            exc_value: BaseException | None,
            traceback: TracebackType | None,
        ) -> bool:
            return self._suppress

    lifecycle = LifecycleRecorder()
    wrapped = wrap_deferred_result(IteratorContext(), lifecycle)
    assert isinstance(wrapped, SyncIteratorContextProxy)
    with wrapped as entered:
        assert entered is wrapped
        assert list(entered) == [1, 2]

    iterator_lifecycle = LifecycleRecorder()
    replacement = iter((3, 4))
    replacement_context = IteratorContext(replacement)
    replacement_proxy = SyncIteratorContextProxy(
        replacement_context,
        replacement_context,
        iterator_lifecycle,
    )
    with replacement_proxy as entered:
        assert list(entered) == [3, 4]

    plain_lifecycle = LifecycleRecorder()
    plain_context = IteratorContext("entered")
    with SyncIteratorContextProxy(plain_context, plain_context, plain_lifecycle) as entered:
        assert entered == "entered"

    body_failure = ValueError("body failed")
    failure_lifecycle = LifecycleRecorder()
    with pytest.raises(ValueError) as caught:
        failing_context = IteratorContext()
        with SyncIteratorContextProxy(
            failing_context,
            failing_context,
            failure_lifecycle,
        ):
            raise body_failure
    assert caught.value is body_failure
    assert failure_lifecycle.errors == [body_failure]

    suppressed_lifecycle = LifecycleRecorder()
    suppressing_context = IteratorContext(suppress=True)
    with SyncIteratorContextProxy(
        suppressing_context,
        suppressing_context,
        suppressed_lifecycle,
    ):
        raise ValueError("suppressed")
    assert suppressed_lifecycle.errors == []


def test_sync_iterator_context_proxy_preserves_entry_and_exit_failures() -> None:
    entry_failure = RuntimeError("entry failed")
    exit_failure = RuntimeError("exit failed")

    class BrokenContext(Iterator[int], AbstractContextManager[Any]):
        def __init__(self, failure: RuntimeError, *, fail_entry: bool) -> None:
            self._failure = failure
            self._fail_entry = fail_entry

        def __iter__(self) -> BrokenContext:
            return self

        def __next__(self) -> int:
            raise StopIteration

        def __enter__(self) -> BrokenContext:
            if self._fail_entry:
                raise self._failure
            return self

        def __exit__(
            self,
            exc_type: type[BaseException] | None,
            exc_value: BaseException | None,
            traceback: TracebackType | None,
        ) -> bool | None:
            if not self._fail_entry:
                raise self._failure
            return None

    entry_lifecycle = LifecycleRecorder()
    entry_context = BrokenContext(entry_failure, fail_entry=True)
    with (
        pytest.raises(RuntimeError) as caught_entry,
        SyncIteratorContextProxy(entry_context, entry_context, entry_lifecycle),
    ):
        pass
    assert caught_entry.value is entry_failure
    assert entry_lifecycle.errors == [entry_failure]

    exit_lifecycle = LifecycleRecorder()
    exit_context = BrokenContext(exit_failure, fail_entry=False)
    with (
        pytest.raises(RuntimeError) as caught_exit,
        SyncIteratorContextProxy(exit_context, exit_context, exit_lifecycle),
    ):
        pass
    assert caught_exit.value is exit_failure
    assert exit_lifecycle.errors == [exit_failure]


async def test_async_iterator_context_proxy_preserves_entered_values_and_outcomes() -> None:
    class AsyncIteratorContext(AsyncIterator[int], AbstractAsyncContextManager[Any]):
        def __init__(self, entered: Any = None, *, suppress: bool = False) -> None:
            self._values = iter((1, 2))
            self._entered = self if entered is None else entered
            self._suppress = suppress

        def __aiter__(self) -> AsyncIteratorContext:
            return self

        async def __anext__(self) -> int:
            try:
                return next(self._values)
            except StopIteration as error:
                raise StopAsyncIteration from error

        async def __aenter__(self) -> Any:
            return self._entered

        async def __aexit__(
            self,
            exc_type: type[BaseException] | None,
            exc_value: BaseException | None,
            traceback: TracebackType | None,
        ) -> bool:
            return self._suppress

    lifecycle = LifecycleRecorder()
    context = AsyncIteratorContext()
    wrapped = wrap_deferred_result(context, lifecycle)
    assert isinstance(wrapped, AsyncIteratorContextProxy)
    async with wrapped as entered:
        assert entered is wrapped
        assert [value async for value in entered] == [1, 2]

    async def values() -> AsyncIterator[int]:
        yield 3
        yield 4

    iterator_lifecycle = LifecycleRecorder()
    replacement = values()
    replacement_context = AsyncIteratorContext(replacement)
    async with AsyncIteratorContextProxy(
        replacement_context,
        replacement_context,
        iterator_lifecycle,
    ) as entered:
        assert [value async for value in entered] == [3, 4]

    plain_lifecycle = LifecycleRecorder()
    plain_context = AsyncIteratorContext("entered")
    async with AsyncIteratorContextProxy(
        plain_context,
        plain_context,
        plain_lifecycle,
    ) as entered:
        assert entered == "entered"

    body_failure = ValueError("async body failed")
    failure_lifecycle = LifecycleRecorder()
    with pytest.raises(ValueError) as caught:
        failing_context = AsyncIteratorContext()
        async with AsyncIteratorContextProxy(
            failing_context,
            failing_context,
            failure_lifecycle,
        ):
            raise body_failure
    assert caught.value is body_failure
    assert failure_lifecycle.errors == [body_failure]

    suppressed_lifecycle = LifecycleRecorder()
    suppressing_context = AsyncIteratorContext(suppress=True)
    async with AsyncIteratorContextProxy(
        suppressing_context,
        suppressing_context,
        suppressed_lifecycle,
    ):
        raise ValueError("suppressed")
    assert suppressed_lifecycle.errors == []


async def test_async_iterator_context_proxy_preserves_entry_and_exit_failures() -> None:
    entry_failure = RuntimeError("async entry failed")
    exit_failure = RuntimeError("async exit failed")

    class BrokenContext(AsyncIterator[int], AbstractAsyncContextManager[Any]):
        def __init__(self, failure: RuntimeError, *, fail_entry: bool) -> None:
            self._failure = failure
            self._fail_entry = fail_entry

        def __aiter__(self) -> BrokenContext:
            return self

        async def __anext__(self) -> int:
            raise StopAsyncIteration

        async def __aenter__(self) -> BrokenContext:
            if self._fail_entry:
                raise self._failure
            return self

        async def __aexit__(
            self,
            exc_type: type[BaseException] | None,
            exc_value: BaseException | None,
            traceback: TracebackType | None,
        ) -> bool | None:
            if not self._fail_entry:
                raise self._failure
            return None

    entry_lifecycle = LifecycleRecorder()
    entry_context = BrokenContext(entry_failure, fail_entry=True)
    with pytest.raises(RuntimeError) as caught_entry:
        async with AsyncIteratorContextProxy(entry_context, entry_context, entry_lifecycle):
            pass
    assert caught_entry.value is entry_failure
    assert entry_lifecycle.errors == [entry_failure]

    exit_lifecycle = LifecycleRecorder()
    exit_context = BrokenContext(exit_failure, fail_entry=False)
    with pytest.raises(RuntimeError) as caught_exit:
        async with AsyncIteratorContextProxy(exit_context, exit_context, exit_lifecycle):
            pass
    assert caught_exit.value is exit_failure
    assert exit_lifecycle.errors == [exit_failure]


def _run_context() -> RunContext:
    return RunContext(
        benchmark_id="streaming",
        case=Case(id="case"),
        variant=Variant(id="variant"),
    )
