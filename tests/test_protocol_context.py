from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

from autobench.protocol import (
    AbstractionLayer,
    ActiveContext,
    CaptureMechanism,
    CapturePolicy,
    Emitter,
    InstrumentationScope,
    LocalCollector,
    attach_context,
    bind_context,
    capture_context,
    get_context,
    new_trace_id,
    reset_context,
    suppress_instrumentation,
    use_context,
)


def test_context_attach_reset_scope_and_suppression_are_token_safe() -> None:
    collector = LocalCollector()
    context = ActiveContext(
        collector=collector,
        trace_id=new_trace_id(),
        capture_policy=CapturePolicy.metadata(),
        correlations={"request_id": "req-1"},
    )

    assert get_context() is None
    token = attach_context(context)
    assert get_context() == context
    assert context.with_span("1" * 16).current_span_id == "1" * 16
    with suppress_instrumentation():
        suppressed = get_context()
        assert suppressed is not None
        assert suppressed.suppressed is True
        assert suppressed.is_suppressed("any.family") is True
    assert get_context() == context

    with suppress_instrumentation("family.a"):
        scoped = get_context()
        assert scoped is not None
        assert scoped.is_suppressed("family.a") is True
        assert scoped.is_suppressed("family.b") is False
        with suppress_instrumentation("family.b"):
            nested = get_context()
            assert nested is not None
            assert nested.is_suppressed("family.a", "family.b") is True
    assert get_context() == context
    reset_context(token)
    assert get_context() is None

    with use_context(context):
        assert get_context() == context
    assert get_context() is None

    with suppress_instrumentation():
        assert get_context() is None


def test_captured_context_binds_exact_callable_to_thread_boundaries() -> None:
    collector = LocalCollector()
    emitter = Emitter(collector, manual_scope())
    context = ActiveContext(collector=collector, trace_id=emitter.trace_id)

    def inspect(prefix: str, *, suffix: str) -> str:
        active = get_context()
        assert active is not None
        return f"{prefix}:{active.trace_id}:{suffix}"

    with use_context(context):
        captured = capture_context()
        bound = bind_context(inspect, captured)
        implicit = bind_context(inspect)

    with ThreadPoolExecutor(max_workers=2) as executor:
        first = executor.submit(bound, "a", suffix="b").result()
        second = executor.submit(implicit, "c", suffix="d").result()

    assert first == f"a:{emitter.trace_id}:b"
    assert second == f"c:{emitter.trace_id}:d"
    assert get_context() is None


def manual_scope() -> InstrumentationScope:
    return InstrumentationScope(
        instrumentor_name="autobench.manual",
        instrumentor_version="0.1.0",
        package_name="autobench",
        package_version="0.1.0",
        mechanism=CaptureMechanism.MANUAL,
        layer=AbstractionLayer.APPLICATION,
    )
