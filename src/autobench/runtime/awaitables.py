from __future__ import annotations

import asyncio
from collections.abc import Coroutine
from typing import Any, TypeVar

ResultT = TypeVar("ResultT")


def run_sync(awaitable: Coroutine[Any, Any, ResultT]) -> ResultT:
    """Run one coroutine without replacing the host thread's configured event loop."""

    with asyncio.Runner(loop_factory=asyncio.new_event_loop) as runner:
        return runner.run(awaitable)


__all__ = ("run_sync",)
