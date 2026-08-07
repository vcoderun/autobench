from __future__ import annotations

import asyncio
import signal
from collections.abc import Coroutine
from typing import Any, TypeVar

ResultT = TypeVar("ResultT")


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
