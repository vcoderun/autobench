from __future__ import annotations

from secrets import token_hex
from typing import Annotated, TypeAlias
from uuid import UUID, uuid4

from pydantic import AfterValidator, StringConstraints

TRACE_ID_PATTERN = r"^[0-9a-f]{32}$"
SPAN_ID_PATTERN = r"^[0-9a-f]{16}$"
SIGNAL_ID_PATTERN = r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$"


def validate_trace_id(value: str) -> str:
    if value == "0" * 32:
        raise ValueError("trace IDs cannot be all zero")
    return value


def validate_span_id(value: str) -> str:
    if value == "0" * 16:
        raise ValueError("span IDs cannot be all zero")
    return value


def validate_signal_id(value: str) -> str:
    if UUID(value).version != 4:
        raise ValueError("signal IDs must be UUID4 values")
    return value


TraceId: TypeAlias = Annotated[
    str,
    StringConstraints(pattern=TRACE_ID_PATTERN),
    AfterValidator(validate_trace_id),
]
SpanId: TypeAlias = Annotated[
    str,
    StringConstraints(pattern=SPAN_ID_PATTERN),
    AfterValidator(validate_span_id),
]
SignalId: TypeAlias = Annotated[
    str,
    StringConstraints(pattern=SIGNAL_ID_PATTERN),
    AfterValidator(validate_signal_id),
]


def new_trace_id() -> TraceId:
    return token_hex(16)


def new_span_id() -> SpanId:
    return token_hex(8)


def new_signal_id() -> SignalId:
    return str(uuid4())


__all__ = (
    "SignalId",
    "SpanId",
    "TraceId",
    "new_signal_id",
    "new_span_id",
    "new_trace_id",
)
