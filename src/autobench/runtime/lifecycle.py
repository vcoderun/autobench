from __future__ import annotations as _annotations

from enum import StrEnum


class RunPhase(StrEnum):
    RESOLVING = "resolving"
    EXECUTING = "executing"
    SCORING = "scoring"
    DERIVING = "deriving"
    POST_PROCESSING = "post_processing"
    FINALIZING = "finalizing"


__all__ = ("RunPhase",)
