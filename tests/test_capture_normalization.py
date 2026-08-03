from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, date, datetime
from decimal import Decimal
from enum import Enum, StrEnum
from pathlib import Path
from uuid import UUID

from pydantic import BaseModel

from autobench.protocol import CapturePolicy, CaptureSession, EncoderRegistry


@dataclass
class NativePayload:
    name: str
    values: list[int]


class ModelPayload(BaseModel):
    name: str
    count: int


class Choice(StrEnum):
    FIRST = "first"


class NumericChoice(Enum):
    FIRST = 1


class Unknown:
    def __repr__(self) -> str:
        return "SENSITIVE_REPR"


def test_dataclass_pydantic_mapping_sequence_exception_and_standard_values_normalize() -> None:
    session = CaptureSession(CapturePolicy.full())
    payload = {
        "dataclass": NativePayload(name="native", values=[1, 2]),
        "model": ModelPayload(name="model", count=3),
        "sequence": (Choice.FIRST, Decimal("1.25"), UUID(int=1), Path("file.txt")),
        "enum": NumericChoice.FIRST,
        "float": 1.5,
        "datetime": datetime(2026, 8, 3, 12, tzinfo=UTC),
        "date": date(2026, 8, 3),
        "error": ValueError("bad input"),
    }

    result = session.capture(payload)

    assert result.value == {
        "dataclass": {"name": "native", "values": [1, 2]},
        "model": {"name": "model", "count": 3},
        "sequence": ["first", "1.25", "00000000-0000-0000-0000-000000000001", "file.txt"],
        "enum": 1,
        "float": 1.5,
        "datetime": "2026-08-03T12:00:00+00:00",
        "date": "2026-08-03",
        "error": {"type": "ValueError", "message": "bad input"},
    }
    assert result.omitted is False


def test_limits_truncate_without_mutating_input_and_report_every_decision() -> None:
    original = {
        "long": "abcdefgh",
        "items": [1, 2, 3],
        "mapping": {"a": 1, "b": 2, "c": 3},
        "deep": {"one": {"two": {"three": "hidden"}}},
        "bad_key": {1: "omitted", "ok": "kept"},
        "binary": b"nested",
    }
    policy = CapturePolicy.full(
        max_string_length=4,
        max_collection_items=2,
        max_depth=3,
        max_inline_bytes=10_000,
    )

    result = CaptureSession(policy).capture(original)

    assert original["long"] == "abcdefgh"
    assert original["items"] == [1, 2, 3]
    assert result.truncated is True
    assert result.value == {
        "long": "abcd",
        "items": [1, 2],
    }
    assert {diagnostic.code for diagnostic in result.diagnostics} == {
        "capture_collection_truncated",
        "capture_string_truncated",
    }
    diagnostics = {diagnostic.code: diagnostic for diagnostic in result.diagnostics}
    assert diagnostics["capture_string_truncated"].details == {
        "original_length": 8,
        "captured_length": 4,
    }
    assert diagnostics["capture_collection_truncated"].details == {
        "original_items": 6,
        "captured_items": 2,
    }

    sequence = CaptureSession(policy).capture([1, 2, 3])
    assert sequence.value == [1, 2]
    assert sequence.diagnostics[0].details == {
        "original_items": 3,
        "captured_items": 2,
    }

    detailed = CaptureSession(
        CapturePolicy.full(
            max_collection_items=20,
            max_depth=2,
            max_inline_bytes=10_000,
        )
    ).capture(original)
    assert isinstance(detailed.value, dict)
    assert detailed.value["bad_key"] == {"ok": "kept"}
    assert detailed.value["binary"] is None
    assert detailed.value["deep"] == {"one": {"two": None}}
    assert {
        "capture_depth_limit",
        "nested_binary_omitted",
        "non_string_key",
    } <= {diagnostic.code for diagnostic in detailed.diagnostics}
    depth_diagnostic = next(
        diagnostic
        for diagnostic in detailed.diagnostics
        if diagnostic.code == "capture_depth_limit"
    )
    assert depth_diagnostic.details == {"depth": 3, "max_depth": 2}


def test_unknown_and_non_finite_values_are_omitted_without_repr_fallback() -> None:
    session = CaptureSession(CapturePolicy.full())

    unknown = session.capture(Unknown())
    not_a_number = session.capture(float("nan"))
    infinity = session.capture(float("inf"))

    assert unknown.omitted is True
    assert unknown.diagnostics[0].code == "unknown_capture_type"
    assert "SENSITIVE_REPR" not in str(unknown)
    assert not_a_number.omitted is True
    assert not_a_number.diagnostics[0].code == "non_finite_number"
    assert infinity.omitted is True

    nested = session.capture(
        {
            "mapping": Unknown(),
            "sequence": [Unknown(), "kept"],
        }
    )
    assert nested.value == {"sequence": ["kept"]}
    assert nested.truncated is True
    assert [diagnostic.code for diagnostic in nested.diagnostics] == [
        "unknown_capture_type",
        "unknown_capture_type",
    ]


def test_registered_encoders_receive_copies_and_fail_without_affecting_capture() -> None:
    registry = EncoderRegistry()
    original = NativePayload(name="native", values=[1, 2])

    def encode(payload: NativePayload) -> dict[str, str | int]:
        removed = payload.values.pop()
        return {"name": payload.name, "removed": removed}

    registry.register(NativePayload, encode)
    result = CaptureSession(CapturePolicy.full(), encoders=registry).capture(original)

    assert result.value == {"name": "native", "removed": 2}
    assert original.values == [1, 2]

    failing = EncoderRegistry()

    def fail(payload: NativePayload) -> dict[str, str]:
        raise RuntimeError(payload.name)

    failing.register(NativePayload, fail)
    failure = CaptureSession(CapturePolicy.full(), encoders=failing).capture(original)
    assert failure.omitted is True
    assert failure.diagnostics[0].code == "encoder_failed"
    assert "RuntimeError" in failure.diagnostics[0].message

    recursive = EncoderRegistry()
    recursive.register(NativePayload, lambda payload: payload)
    recursion = CaptureSession(CapturePolicy.full(), encoders=recursive).capture(original)
    assert recursion.omitted is True
    assert recursion.diagnostics[0].code == "encoder_recursion"


def test_custom_redactors_run_in_order_and_failures_are_isolated() -> None:
    calls: list[str] = []

    def first(value: str, path: tuple[str, ...], semantic: str | None) -> str:
        calls.append(f"first:{'.'.join(path)}:{semantic}")
        return value.replace("alice", "user")

    def failing(value: str, path: tuple[str, ...], semantic: str | None) -> str:
        calls.append("failing")
        raise RuntimeError(value)

    def last(value: str, path: tuple[str, ...], semantic: str | None) -> str:
        calls.append("last")
        return value.upper()

    result = CaptureSession(
        CapturePolicy.redacted(),
        redactors=(first, failing, last),
    ).capture("alice", path=("user", "name"), semantic_type="profile.name")

    assert result.value == "USER"
    assert calls == ["first:user.name:profile.name", "failing", "last"]
    assert {diagnostic.code for diagnostic in result.diagnostics} == {
        "content_redacted",
        "redactor_failed",
    }
