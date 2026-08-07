from __future__ import annotations

import asyncio
import signal
from collections.abc import Coroutine
from typing import Any, TypeVar

ResultT = TypeVar("ResultT")

_retained_tasks: set[asyncio.Task[Any]] = set()


class ProcessSignalInterrupt(BaseException):
    def __init__(self, signal_number: int) -> None:
        self.signal_number = signal_number
        super().__init__(f"Process signal {signal_number} interrupted benchmark execution.")


def run_sync(awaitable: Coroutine[Any, Any, ResultT]) -> ResultT:
    """Run one coroutine without replacing the host thread's configured event loop."""

    with asyncio.Runner(loop_factory=asyncio.new_event_loop) as runner:
        return runner.run(awaitable)


def run_sync_cooperatively(awaitable: Coroutine[Any, Any, ResultT]) -> ResultT:
    """Run a coroutine and translate supported process signals into task cancellation."""

    with asyncio.Runner(loop_factory=asyncio.new_event_loop) as runner:
        return runner.run(_run_cooperatively(awaitable))


async def settle_task(
    task: asyncio.Task[Any],
    *,
    timeout_seconds: float,
    cancellation_grace_seconds: float = 0.1,
    cancel_on_timeout: bool = True,
    description: str = "Task",
) -> BaseException | None:
    """Bound a task wait without ever leaving a still-running task unowned."""

    if timeout_seconds < 0:
        raise ValueError("timeout_seconds must be non-negative")
    if cancellation_grace_seconds < 0:
        raise ValueError("cancellation_grace_seconds must be non-negative")
    if await _wait_for_task(task, timeout_seconds):
        return _task_error(task)
    if not cancel_on_timeout:
        retain_task(task, description=description)
        return TimeoutError(
            f"{description} did not finish within {timeout_seconds:g} seconds and remains active."
        )

    task.cancel()
    if await _wait_for_task(task, cancellation_grace_seconds):
        error = _task_error(task)
        if error is not None and not isinstance(error, asyncio.CancelledError):
            return error
        return TimeoutError(
            f"{description} did not finish within {timeout_seconds:g} seconds; "
            "cancellation was acknowledged."
        )

    retain_task(task, description=description)
    return TimeoutError(
        f"{description} did not finish within {timeout_seconds:g} seconds and ignored "
        f"cancellation for {cancellation_grace_seconds:g} seconds; it remains active."
    )


def retain_task(task: asyncio.Task[Any], *, description: str) -> None:
    """Retain and observe a task that outlives its bounded caller."""

    if task in _retained_tasks:
        return
    _retained_tasks.add(task)
    loop = task.get_loop()

    def observe(completed: asyncio.Task[Any]) -> None:
        _retained_tasks.discard(completed)
        if completed.cancelled():
            return
        error = completed.exception()
        if error is not None:
            loop.call_exception_handler(
                {
                    "message": f"{description} failed after its caller stopped waiting.",
                    "exception": error,
                    "task": completed,
                }
            )

    task.add_done_callback(observe)


async def _wait_for_task(task: asyncio.Task[Any], timeout_seconds: float) -> bool:
    deadline = asyncio.get_running_loop().time() + timeout_seconds
    while not task.done():
        remaining = max(deadline - asyncio.get_running_loop().time(), 0)
        if remaining == 0:
            return False
        try:
            done, _ = await asyncio.wait((task,), timeout=remaining)
        except asyncio.CancelledError:
            continue
        if not done:
            return False
    return True


def _task_error(task: asyncio.Task[Any]) -> BaseException | None:
    if task.cancelled():
        return asyncio.CancelledError()
    try:
        task.result()
    except BaseException as error:
        return error
    return None


async def _run_cooperatively(awaitable: Coroutine[Any, Any, ResultT]) -> ResultT:
    loop = asyncio.get_running_loop()
    task = asyncio.create_task(awaitable)
    received_signal: int | None = None
    handler_installed = False

    def cancel_for_signal() -> None:
        nonlocal received_signal
        received_signal = signal.SIGTERM
        task.cancel("SIGTERM")

    try:
        loop.add_signal_handler(signal.SIGTERM, cancel_for_signal)
        handler_installed = True
    except (NotImplementedError, RuntimeError):
        pass
    try:
        return await task
    except asyncio.CancelledError as cancellation:
        if received_signal is not None:
            raise ProcessSignalInterrupt(received_signal) from cancellation
        raise
    finally:
        if handler_installed:
            loop.remove_signal_handler(signal.SIGTERM)


__all__ = ("ProcessSignalInterrupt", "run_sync", "run_sync_cooperatively")
