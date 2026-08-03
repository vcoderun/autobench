from __future__ import annotations

from dataclasses import dataclass

import pytest
from pydantic import ValidationError

from autobench.protocol import (
    AbstractionLayer,
    ActiveContext,
    CaptureLevel,
    CaptureMechanism,
    CapturePolicy,
    CaptureSession,
    Emitter,
    InstrumentationScope,
    LocalCollector,
)


@dataclass
class FailureError(Exception):
    message: str


@dataclass
class NativeMetadata:
    name: str


def test_capture_levels_and_semantic_defaults_do_not_expose_sensitive_content() -> None:
    value = "private prompt"

    omitted = CaptureSession(CapturePolicy.none()).capture(value, semantic_type="prompt.text")
    metadata = CaptureSession(CapturePolicy.metadata()).capture(value)
    hashed = CaptureSession(CapturePolicy.hashed()).capture(value)
    redacted = CaptureSession(
        CapturePolicy.redacted(),
        redactors=(lambda text, path, semantic: text.replace("private", "[PII]"),),
    ).capture(value)
    full = CaptureSession(CapturePolicy.full()).capture(value)
    prompt_default = CaptureSession().capture(value, semantic_type="prompt.text")
    environment_default = CaptureSession().capture(value, semantic_type="environment.api_key")
    exception_default = CaptureSession().capture(FailureError("bad"))
    explicit_none = CaptureSession(CapturePolicy.full()).capture(
        value,
        level=CaptureLevel.NONE,
    )
    empty_path = CaptureSession(CapturePolicy.full()).capture(value, path=())

    assert omitted.omitted is True
    assert omitted.diagnostics[0].code == "capture_omitted"
    assert metadata.value == {"type": "str", "length": len(value)}
    assert isinstance(hashed.value, dict)
    assert hashed.value["sha256"]
    assert value not in str(hashed.value)
    assert redacted.value == "[PII] prompt"
    assert full.value == value
    assert prompt_default.level is CaptureLevel.HASH
    assert environment_default.level is CaptureLevel.NONE
    assert environment_default.omitted is True
    assert exception_default.level is CaptureLevel.REDACTED
    assert explicit_none.omitted is True
    assert empty_path.value == value


def test_metadata_retains_safe_scalar_semantics_and_longest_override_wins() -> None:
    policy = CapturePolicy(
        semantic_overrides={
            "tool": CaptureLevel.HASH,
            "tool.name": CaptureLevel.FULL,
        }
    )
    session = CaptureSession(policy)

    model = session.capture("gpt-x", semantic_type="llm.model.name")
    tool = session.capture("search", semantic_type="tool.name")
    numeric = session.capture(17, semantic_type="llm.tokens.input")
    boolean = session.capture(True, semantic_type="result.success")
    null = session.capture(None, semantic_type="result.value")
    structured = session.capture(NativeMetadata(name="payload"))

    assert model.value == "gpt-x"
    assert tool.level is CaptureLevel.FULL
    assert tool.value == "search"
    assert numeric.value == 17
    assert boolean.value is True
    assert null.value is None
    assert structured.value == {"type": "NativeMetadata"}


def test_path_allow_and_deny_rules_apply_to_roots_and_nested_fields() -> None:
    policy = CapturePolicy.full(
        allow_paths=("payload.allowed", "payload.secret"),
        deny_paths=("payload.secret",),
    )
    session = CaptureSession(policy)

    result = session.capture(
        {"allowed": "yes", "secret": "no", "other": "no"},
        path="payload",
    )
    denied_root = session.capture("value", path="payload.secret")
    outside_root = session.capture("value", path="outside")

    assert result.value == {"allowed": "yes"}
    assert result.truncated is True
    assert {diagnostic.code for diagnostic in result.diagnostics} == {
        "capture_denied",
        "capture_not_allowed",
    }
    assert denied_root.omitted is True
    assert denied_root.diagnostics[0].code == "capture_denied"
    assert outside_root.omitted is True
    assert outside_root.diagnostics[0].code == "capture_not_allowed"


def test_semantic_allow_and_deny_rules_use_families_and_globs() -> None:
    session = CaptureSession(
        CapturePolicy.full(
            allow_semantics=("llm.*",),
            deny_semantics=("llm.prompt",),
        )
    )

    allowed = session.capture("gpt-x", semantic_type="llm.model.name")
    denied = session.capture("private", semantic_type="llm.prompt.system")
    outside = session.capture("search", semantic_type="tool.name")

    assert allowed.value == "gpt-x"
    assert denied.omitted is True
    assert denied.diagnostics[0].code == "capture_semantic_denied"
    assert outside.omitted is True
    assert outside.diagnostics[0].code == "capture_semantic_not_allowed"


def test_secret_paths_stay_redacted_even_with_full_capture() -> None:
    result = CaptureSession(CapturePolicy.full()).capture(
        {
            "authorization": "Bearer secret",
            "nested": {"api-key": "secret", "tokens": 12},
        },
        path="request.headers",
    )

    assert result.value == {
        "authorization": "[REDACTED]",
        "nested": {"api-key": "[REDACTED]", "tokens": 12},
    }
    assert [diagnostic.path for diagnostic in result.diagnostics] == [
        "request.headers.authorization",
        "request.headers.nested.api-key",
    ]


def test_policy_validation_factories_and_active_context_capture_reference() -> None:
    assert CapturePolicy.metadata().default_level is CaptureLevel.METADATA
    assert CapturePolicy.redacted().default_level is CaptureLevel.REDACTED
    assert CapturePolicy.full().default_level is CaptureLevel.FULL

    with pytest.raises(ValidationError, match="capture paths and semantics cannot be empty"):
        CapturePolicy(allow_paths=("",))
    with pytest.raises(ValidationError, match="capture paths and semantics cannot be empty"):
        CapturePolicy(allow_semantics=("",))

    collector = LocalCollector()
    policy = CapturePolicy.full(max_inline_bytes=64)
    emitter = Emitter(collector, manual_scope())
    context = ActiveContext(
        collector=collector,
        trace_id=emitter.trace_id,
        capture_policy=policy,
    )
    assert context.capture_policy == policy


def manual_scope() -> InstrumentationScope:
    return InstrumentationScope(
        instrumentor_name="autobench.manual",
        instrumentor_version="0.1.0",
        package_name="autobench",
        package_version="0.1.0",
        mechanism=CaptureMechanism.MANUAL,
        layer=AbstractionLayer.APPLICATION,
    )
