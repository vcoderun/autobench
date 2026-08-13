from __future__ import annotations

from collections.abc import Awaitable, Callable, Iterator
from dataclasses import dataclass
from functools import wraps
from inspect import signature
from typing import Any, ParamSpec, TypeVar

import pytest
from pydantic import ValidationError

from autobench import (
    Compatibility,
    CompatibilityStatus,
    InstrumentationError,
    InstrumentationHandle,
    InstrumentationManager,
    InstrumentationRuntime,
    InstrumentCall,
    InstrumentorCapabilities,
    InstrumentorInfo,
    RunContext,
    check_package_compatibility,
    suppress_instrumentation,
)
from autobench.data.datasets import Case
from autobench.data.variants import Variant
from autobench.instrumentation.patching import (
    CallLifecycle,
    InstrumentationConflictError,
    PatchManager,
)
from autobench.protocol.collector import LocalCollector
from autobench.protocol.context import ActiveContext, use_context
from autobench.protocol.ids import new_trace_id
from autobench.protocol.signals import (
    AbstractionLayer,
    CaptureMechanism,
    EndReason,
    SpanStatus,
)
from autobench.protocol.traces import DiagnosticSeverity
from autobench.runtime.instrumentation import reset_active_run_context, set_active_run_context


class NativeInstrumentor:
    def __init__(
        self,
        instrumentor_id: str = "native.test",
        version: str = "1",
        compatibility: Compatibility | None = None,
        *,
        target_distribution: str | None = None,
        optional_dependencies: tuple[str, ...] = (),
    ) -> None:
        self._info = InstrumentorInfo(
            id=instrumentor_id,
            version=version,
            mechanism=CaptureMechanism.HOOK,
            layer=AbstractionLayer.FRAMEWORK,
            target_distribution=target_distribution,
            optional_dependencies=optional_dependencies,
            capabilities=InstrumentorCapabilities(native_hooks=True),
        )
        self._compatibility = compatibility or Compatibility.compatible()
        self.install_count = 0
        self.close_count = 0

    @property
    def info(self) -> InstrumentorInfo:
        return self._info

    def check(self) -> Compatibility:
        return self._compatibility

    def install(self, runtime: InstrumentationRuntime) -> InstrumentationHandle:
        assert isinstance(runtime, InstrumentationRuntime)
        self.install_count += 1
        return InstrumentationHandle(self._close, info=self.info)

    def _close(self) -> None:
        self.close_count += 1


def test_manager_reference_counts_duplicate_installs_and_closes_native_hooks() -> None:
    instrumentor = NativeInstrumentor()
    manager = InstrumentationManager()

    first = manager.install(instrumentor)
    second = manager.install(instrumentor)

    assert manager.installed == (instrumentor.info,)
    assert instrumentor.install_count == 1
    first.close()
    first.close()
    assert instrumentor.close_count == 0
    second.close()
    assert instrumentor.close_count == 1
    assert manager.installed == ()

    late = manager.install(instrumentor)
    manager.close()
    manager.close()
    late.close()
    assert instrumentor.close_count == 2
    with pytest.raises(InstrumentationError, match="manager is closed"):
        manager.install(instrumentor)


def test_manager_context_closes_in_reverse_order_and_rejects_version_conflicts() -> None:
    closed: list[str] = []

    class OrderedInstrumentor(NativeInstrumentor):
        def _close(self) -> None:
            closed.append(self.info.id)
            super()._close()

    first = OrderedInstrumentor("first")
    second = OrderedInstrumentor("second")
    with InstrumentationManager() as manager:
        manager.install(first)
        manager.install(second)
        with pytest.raises(InstrumentationError, match="cannot install 2"):
            manager.install(NativeInstrumentor("first", version="2"))

    assert closed == ["second", "first"]


@pytest.mark.parametrize(
    ("compatibility", "message"),
    [
        (
            Compatibility(
                status=CompatibilityStatus.UNAVAILABLE,
                diagnostics=("dependency missing",),
            ),
            "dependency missing",
        ),
        (
            Compatibility(
                status=CompatibilityStatus.CONFLICT,
                conflicts=("other wrapper",),
            ),
            "other wrapper",
        ),
    ],
)
def test_manager_rejects_non_installable_compatibility(
    compatibility: Compatibility,
    message: str,
) -> None:
    manager = InstrumentationManager()
    with pytest.raises(InstrumentationError, match=message):
        manager.install(NativeInstrumentor(compatibility=compatibility))


def test_package_compatibility_checks_absent_supported_unsupported_and_invalid_versions() -> None:
    absent = InstrumentorInfo(
        id="absent",
        version="1",
        target_distribution="definitely-not-an-installed-autobench-package",
        mechanism=CaptureMechanism.PATCH,
        layer=AbstractionLayer.CLIENT,
    )
    supported = absent.model_copy(
        update={"id": "supported", "target_distribution": "pydantic", "supported_versions": ">=2"}
    )
    unsupported = supported.model_copy(update={"id": "unsupported", "supported_versions": "<1"})
    invalid = supported.model_copy(update={"id": "invalid", "supported_versions": "not a range"})
    without_range = supported.model_copy(update={"id": "without-range", "supported_versions": None})

    assert check_package_compatibility(absent).status is CompatibilityStatus.UNAVAILABLE
    assert check_package_compatibility(supported).status is CompatibilityStatus.COMPATIBLE
    assert check_package_compatibility(without_range).status is CompatibilityStatus.COMPATIBLE
    assert check_package_compatibility(unsupported).status is CompatibilityStatus.UNSUPPORTED
    invalid_result = check_package_compatibility(invalid)
    assert invalid_result.status is CompatibilityStatus.UNSUPPORTED
    assert "invalid version compatibility declaration" in invalid_result.diagnostics[0]


def test_package_compatibility_reports_optional_dependencies_as_degraded_features() -> None:
    info = InstrumentorInfo(
        id="optional",
        version="1",
        mechanism=CaptureMechanism.PATCH,
        layer=AbstractionLayer.CLIENT,
        optional_dependencies=(
            "pydantic>=2",
            "definitely-not-an-installed-autobench-package>=1",
            "pydantic<1",
            "not a valid requirement ???",
            "typing-extensions; python_version < '1'",
        ),
    )

    compatibility = check_package_compatibility(info)

    assert compatibility.status is CompatibilityStatus.DEGRADED
    assert compatibility.installable is True
    assert compatibility.degraded_features == (
        "definitely-not-an-installed-autobench-package",
        "pydantic",
        "not a valid requirement ???",
    )
    assert len(compatibility.diagnostics) == 3


def test_manager_combines_declared_and_package_compatibility() -> None:
    instrumentor = NativeInstrumentor(
        compatibility=Compatibility(
            status=CompatibilityStatus.DEGRADED,
            degraded_features=("private_hook",),
            diagnostics=("using public fallback",),
            private_seam_supported=False,
        ),
        optional_dependencies=("definitely-not-an-installed-autobench-package",),
    )

    compatibility = InstrumentationManager().check(instrumentor)

    assert compatibility.status is CompatibilityStatus.DEGRADED
    assert compatibility.degraded_features == (
        "private_hook",
        "definitely-not-an-installed-autobench-package",
    )
    assert compatibility.private_seam_supported is False
    assert compatibility.diagnostics[0] == "using public fallback"

    unavailable = NativeInstrumentor(
        target_distribution="definitely-not-an-installed-autobench-package"
    )
    assert InstrumentationManager().check(unavailable).status is CompatibilityStatus.UNAVAILABLE


def test_instrumentor_metadata_validates_version_and_convention_relationships() -> None:
    with pytest.raises(ValidationError, match="supported_versions requires target_distribution"):
        InstrumentorInfo(
            id="invalid-target",
            version="1",
            supported_versions=">=1",
            mechanism=CaptureMechanism.PATCH,
            layer=AbstractionLayer.CLIENT,
        )
    with pytest.raises(ValidationError, match="requires source_convention"):
        InstrumentorInfo(
            id="invalid-convention",
            version="1",
            source_convention_version="1",
            mechanism=CaptureMechanism.PATCH,
            layer=AbstractionLayer.CLIENT,
        )
    capabilities = InstrumentorCapabilities.model_validate(
        {"sync": True, "async": True, "streaming": True, "native_hooks": False}
    )
    assert capabilities.async_ is True


def test_compatibility_properties_and_handle_context_manager_expose_lifecycle_state() -> None:
    unavailable = Compatibility(status=CompatibilityStatus.UNAVAILABLE)
    unsupported = Compatibility(status=CompatibilityStatus.UNSUPPORTED)
    conflict = Compatibility(status=CompatibilityStatus.CONFLICT)
    degraded = Compatibility(status=CompatibilityStatus.DEGRADED)

    assert unavailable.available is False
    assert unavailable.supported is False
    assert unsupported.available is True
    assert unsupported.supported is False
    assert conflict.supported is True
    assert conflict.installable is False
    assert degraded.installable is True

    closed: list[bool] = []
    with InstrumentationHandle(lambda: closed.append(True)) as handle:
        assert handle.closed is False
    assert handle.closed is True
    assert closed == [True]


class RecordingLifecycle:
    def __init__(self, events: list[str]) -> None:
        self._events = events

    def resume(self) -> None:
        self._events.append("resume")

    def suspend(self) -> None:
        self._events.append("suspend")

    def observe(self, item: Any) -> None:
        self._events.append(f"item:{item}")

    def finish(
        self,
        *,
        result: Any = None,
        error: BaseException | None = None,
        reason: EndReason = EndReason.COMPLETED,
        partial: bool = False,
    ) -> None:
        self._events.append(
            f"finish:{result}:{type(error).__name__ if error else reason}:{partial}"
        )


@dataclass
class RecordingHandler:
    name: str
    events: list[str]
    suppression_keys: tuple[str, ...] = ()
    begin_error: bool = False

    def begin(self, call: InstrumentCall) -> CallLifecycle | None:
        self.events.append(f"begin:{self.name}:{call.args}")
        if self.begin_error:
            raise RuntimeError("handler failed")
        return RecordingLifecycle(self.events)

    def diagnose(self, stage: str, error: Exception) -> None:
        self.events.append(f"diagnose:{self.name}:{stage}:{type(error).__name__}")

    def close(self) -> None:
        self.events.append(f"close:{self.name}")


def test_patch_manager_reference_counts_owners_and_preserves_signature_and_inheritance() -> None:
    class Base:
        def execute(self, value: int, *, scale: int = 1) -> int:
            return value * scale

    class Worker(Base):
        pass

    events: list[str] = []
    patches = PatchManager()
    original_signature = signature(Worker.execute)
    first = patches.patch_method(
        Worker,
        "execute",
        owner="first",
        handler=RecordingHandler("first", events),
    )
    duplicate = patches.patch_method(
        Worker,
        "execute",
        owner="first",
        handler=RecordingHandler("ignored", events),
    )
    second = patches.patch_method(
        Worker,
        "execute",
        owner="second",
        handler=RecordingHandler("second", events),
    )

    assert Worker().execute(3, scale=2) == 6
    assert signature(Worker.execute) == original_signature
    assert "execute" in Worker.__dict__
    first.close()
    assert "execute" in Worker.__dict__
    duplicate.close()
    assert "execute" in Worker.__dict__
    second.close()
    assert "execute" not in Worker.__dict__
    assert Worker().execute(2) == 2
    assert events.count("close:first") == 1
    assert events.count("close:second") == 1


async def test_patch_manager_awaits_decorated_async_methods() -> None:
    parameters = ParamSpec("parameters")
    result_type = TypeVar("result_type")

    def passthrough(
        method: Callable[parameters, Awaitable[result_type]],
    ) -> Callable[parameters, Awaitable[result_type]]:
        @wraps(method)
        def wrapper(
            *args: parameters.args,
            **kwargs: parameters.kwargs,
        ) -> Awaitable[result_type]:
            return method(*args, **kwargs)

        return wrapper

    class Worker:
        @passthrough
        async def execute(self, value: int) -> int:
            return value * 2

    events: list[str] = []
    patches = PatchManager()
    handle = patches.patch_method(
        Worker,
        "execute",
        owner="decorated-async",
        handler=RecordingHandler("decorated-async", events),
    )

    assert await Worker().execute(3) == 6
    handle.close()
    assert "finish:6:completed:False" in events


def test_patch_manager_reports_external_conflicts_without_overwriting_external_wrapper() -> None:
    class Worker:
        def execute(self) -> str:
            return "original"

    patches = PatchManager()
    events: list[str] = []
    handle = patches.patch_method(
        Worker,
        "execute",
        owner="first",
        handler=RecordingHandler("first", events),
    )
    autobench_wrapper = Worker.execute

    def external(self: Worker) -> str:
        return f"external:{autobench_wrapper(self)}"

    type.__setattr__(Worker, "execute", external)
    with pytest.raises(InstrumentationConflictError, match="changed after Autobench"):
        patches.patch_method(
            Worker,
            "execute",
            owner="second",
            handler=RecordingHandler("second", events),
        )

    handle.close()
    assert Worker().execute() == "external:original"
    assert patches.diagnostics[0].owner == "first"
    assert "changed after Autobench" in patches.diagnostics[0].message


def test_patch_manager_rejects_unexpected_descriptors_and_isolates_handler_errors() -> None:
    class Worker:
        @staticmethod
        def execute(value: int) -> int:
            return value

    patches = PatchManager()
    events: list[str] = []
    with pytest.raises(InstrumentationConflictError, match="expected descriptor"):
        patches.patch_method(
            Worker,
            "execute",
            owner="wrong",
            handler=RecordingHandler("wrong", events),
            expected_descriptor=Worker.execute,
        )
    handle = patches.patch_method(
        Worker,
        "execute",
        owner="broken",
        handler=RecordingHandler("broken", events, begin_error=True),
    )
    assert Worker.execute(3) == 3
    assert "diagnose:broken:begin:RuntimeError" in events
    handle.close()


def test_scoped_suppression_skips_only_matching_instrumentation_family() -> None:
    class Worker:
        def execute(self) -> int:
            return 1

    ctx = RunContext(benchmark_id="demo", case=Case(id="case"), variant=Variant(id="variant"))
    events: list[str] = []
    patches = PatchManager()
    patches.patch_method(
        Worker,
        "execute",
        owner="matched",
        handler=RecordingHandler("matched", events, suppression_keys=("family.a",)),
    )
    patches.patch_method(
        Worker,
        "execute",
        owner="retained",
        handler=RecordingHandler("retained", events, suppression_keys=("family.b",)),
    )
    token = set_active_run_context(ctx)
    try:
        with suppress_instrumentation("family.a"):
            assert Worker().execute() == 1
    finally:
        reset_active_run_context(token)
        patches.close()

    assert not any(event.startswith("begin:matched") for event in events)
    assert any(event.startswith("begin:retained") for event in events)


def test_runtime_patch_and_diagnostics_share_the_active_abp_trace() -> None:
    class Worker:
        def execute(self, value: int) -> int:
            return value * 2

    info = InstrumentorInfo(
        id="runtime.test",
        version="1",
        mechanism=CaptureMechanism.PATCH,
        layer=AbstractionLayer.CLIENT,
    )
    events: list[str] = []
    runtime = InstrumentationRuntime()
    handle = runtime.patch_method(
        info,
        Worker,
        "execute",
        RecordingHandler("runtime", events),
    )

    assert Worker().execute(3) == 6
    assert runtime.diagnose(info, "outside", "no active trace") is False

    ctx = RunContext(benchmark_id="demo", case=Case(id="case"), variant=Variant(id="variant"))
    token = set_active_run_context(ctx)
    try:
        assert (
            runtime.diagnose(
                info,
                "fallback",
                "using a public hook",
                severity=DiagnosticSeverity.ERROR,
            )
            is True
        )
        assert ctx.trace.diagnostics[0].code == "fallback"
        ctx.finalize()
        assert runtime.diagnose(info, "late", "trace already finished") is False
    finally:
        reset_active_run_context(token)
        handle.close()


def test_runtime_keyed_spans_preserve_explicit_parentage_and_targeted_evidence() -> None:
    info = InstrumentorInfo(
        id="runtime.keyed",
        version="1",
        mechanism=CaptureMechanism.CALLBACK,
        layer=AbstractionLayer.FRAMEWORK,
    )
    runtime = InstrumentationRuntime()
    assert runtime.start_span(info, "outside", "outside") is None
    assert runtime.end_span(info, "outside") is False

    ctx = RunContext(benchmark_id="demo", case=Case(id="case"), variant=Variant(id="variant"))
    token = set_active_run_context(ctx)
    try:
        parent = runtime.start_span(info, "run", "optimizer", kind="optimization")
        assert parent is not None
        first = runtime.start_span(
            info,
            "eval:1",
            "evaluation",
            parent_key="run",
            kind="evaluation",
        )
        second = runtime.start_span(
            info,
            "eval:2",
            "evaluation",
            parent_key="run",
            kind="evaluation",
        )
        assert first is not None
        assert second is not None
        assert runtime.start_span(info, "eval:1", "duplicate") is not None
        assert runtime.start_span(info, "orphan", "orphan", parent_key="missing") is None

        metric = runtime.metric(info, "score", 0.75, span_key="eval:2")
        event = runtime.event(info, "selected", span_key="eval:1")
        assert metric is not None and metric.span_id == second.id
        assert event is not None and event.span_id == first.id
        assert runtime.metric(info, "missing", 1, span_key="missing") is None
        assert runtime.event(info, "missing", span_key="missing") is None
        with pytest.raises(ValueError, match="span_key or span_id"):
            runtime.metric(info, "invalid", 1, span_key="eval:1", span_id=first.id)
        with pytest.raises(ValueError, match="span_key or span_id"):
            runtime.event(info, "invalid", span_key="eval:1", span_id=first.id)

        assert runtime.end_span(info, "eval:2", output={"score": 0.75}) is True
        assert runtime.end_span(info, "eval:1", error="candidate failed", partial=True) is True
        assert runtime.end_span(info, "eval:1") is False
        assert runtime.span_for_key(info, "eval:1") is None
        assert runtime.end_span(info, "run", attributes={"winner": "eval:2"}) is True
        ctx.finalize()
    finally:
        reset_active_run_context(token)

    spans = {span.id: span for span in ctx.spans}
    assert spans[first.id].parent_id == parent.id
    assert spans[second.id].parent_id == parent.id
    assert spans[second.id].output == {"score": 0.75}
    assert spans[parent.id].attributes["winner"] == "eval:2"
    assert spans[first.id].error is not None
    diagnostic_codes = {diagnostic.code for diagnostic in ctx.trace.diagnostics}
    assert {
        "keyed_span_duplicate_start",
        "keyed_span_parent_missing",
        "keyed_span_metric_target_missing",
        "keyed_span_event_target_missing",
        "keyed_span_missing_end",
    }.issubset(diagnostic_codes)


def test_runtime_keyed_span_validation_and_finalize_prune_stale_state() -> None:
    info = InstrumentorInfo(
        id="runtime.keyed",
        version="1",
        mechanism=CaptureMechanism.CALLBACK,
        layer=AbstractionLayer.FRAMEWORK,
    )
    runtime = InstrumentationRuntime()
    ctx = RunContext(benchmark_id="demo", case=Case(id="case"), variant=Variant(id="variant"))
    token = set_active_run_context(ctx)
    try:
        with pytest.raises(ValueError, match="must not be empty"):
            runtime.start_span(info, "", "invalid")
        with pytest.raises(ValueError, match="parent_key or parent_span_id"):
            runtime.start_span(
                info,
                "invalid",
                "invalid",
                parent_key="parent",
                parent_span_id="span_1",
            )
        active = runtime.start_span(info, "active", "active")
        assert active is not None and active.is_recording()
        ctx.finalize()
        assert runtime.span_for_key(info, "active") is None
        assert runtime.end_span(info, "active") is False
    finally:
        reset_active_run_context(token)


def test_runtime_keyed_span_mutation_links_status_and_non_run_context_boundaries() -> None:
    info = InstrumentorInfo(
        id="runtime.keyed.boundaries",
        version="1",
        mechanism=CaptureMechanism.CALLBACK,
        layer=AbstractionLayer.FRAMEWORK,
    )
    runtime = InstrumentationRuntime()
    protocol_context = ActiveContext(collector=LocalCollector(), trace_id=new_trace_id())

    with use_context(protocol_context):
        assert runtime.start_span(info, "outside", "outside") is None
        assert runtime.metric(info, "outside", 1) is None
        assert runtime.event(info, "outside") is None
        assert runtime.set_extension(info, "outside", {"value": 1}) is False

    context = RunContext(
        benchmark_id="demo",
        case=Case(id="case"),
        variant=Variant(id="variant"),
    )
    token = set_active_run_context(context)
    try:
        parent = runtime.start_span(info, "parent", "parent")
        child = runtime.start_span(info, "child", "child", parent_key="parent")
        assert parent is not None
        assert child is not None
        assert child.set_output({"result": "candidate"}) is True
        assert child.set_attribute("candidate", "v2") is True
        assert child.set_usage("evaluations", 3) is True
        assert child.link_to(parent) is True

        missing = type(child)(context, "missing-span")
        assert child.link_to(missing) is False
        assert runtime.event(info, "root-event") is not None
        assert runtime.set_extension(info, "integration/v1", {"status": "captured"}) is True
        assert runtime.end_span(
            info,
            "child",
            attributes={"selected": True},
            usage={"optimizer_calls": 2},
            status=SpanStatus.ERROR,
        )
        assert child.set_output("late") is False
        assert child.link_to(parent) is False
        assert runtime.end_span(info, "parent")
    finally:
        reset_active_run_context(token)

    child_record = context._span_by_id(child.id)
    assert child_record is not None
    assert child_record.output == {"result": "candidate"}
    assert child_record.attributes == {"candidate": "v2", "selected": True}
    assert child_record.usage == {"evaluations": 3, "optimizer_calls": 2}
    child_trace = next(span for span in context.trace.spans if span.operation == "child")
    assert child_trace.status is SpanStatus.ERROR
    assert child_trace.end_reason is EndReason.FAILED
    assert context.extensions["integration/v1"] == {"status": "captured"}

    suppressed = RunContext(
        benchmark_id="demo",
        case=Case(id="suppressed"),
        variant=Variant(id="variant"),
    )
    token = set_active_run_context(suppressed)
    try:
        with suppress_instrumentation(info.id):
            assert runtime.span_for_key(info, "parent") is None
            assert runtime.event(info, "suppressed") is None
            assert runtime.set_extension(info, "integration/v1", True) is False
    finally:
        reset_active_run_context(token)

    finalized = RunContext(
        benchmark_id="demo",
        case=Case(id="finalized"),
        variant=Variant(id="variant"),
    )
    token = set_active_run_context(finalized)
    try:
        stale = runtime.start_span(info, "stale", "stale")
        assert stale is not None
        finalized.finalize()
        assert runtime.end_span(info, "stale") is False
    finally:
        reset_active_run_context(token)


class FaultyLifecycle(RecordingLifecycle):
    def __init__(self, events: list[str], failure_stage: str) -> None:
        super().__init__(events)
        self._failure_stage = failure_stage

    def resume(self) -> None:
        if self._failure_stage == "resume":
            raise RuntimeError("resume callback failed")
        super().resume()

    def suspend(self) -> None:
        if self._failure_stage in {"suspend", "finish_and_suspend"}:
            raise RuntimeError("suspend callback failed")
        super().suspend()

    def observe(self, item: Any) -> None:
        if self._failure_stage == "stream_item":
            raise RuntimeError("stream callback failed")
        super().observe(item)

    def finish(
        self,
        *,
        result: Any = None,
        error: BaseException | None = None,
        reason: EndReason = EndReason.COMPLETED,
        partial: bool = False,
    ) -> None:
        if self._failure_stage in {"finish", "finish_and_suspend"}:
            raise RuntimeError("finish callback failed")
        super().finish(result=result, error=error, reason=reason, partial=partial)


@dataclass
class FaultyHandler:
    name: str
    events: list[str]
    failure_stage: str
    suppression_keys: tuple[str, ...] = ()

    def begin(self, call: InstrumentCall) -> CallLifecycle:
        self.events.append(f"begin:{self.name}")
        return FaultyLifecycle(self.events, self.failure_stage)

    def diagnose(self, stage: str, error: Exception) -> None:
        self.events.append(f"diagnose:{self.name}:{stage}:{type(error).__name__}")

    def close(self) -> None:
        self.events.append(f"close:{self.name}")


def test_patch_lifecycle_callback_failures_never_change_application_stream_behavior() -> None:
    class Worker:
        def stream(self) -> Iterator[int]:
            return iter((1,))

    events: list[str] = []
    patches = PatchManager()
    handles = [
        patches.patch_method(
            Worker,
            "stream",
            owner=stage,
            handler=FaultyHandler(stage, events, stage),
        )
        for stage in ("resume", "suspend", "stream_item", "finish")
    ]

    assert list(Worker().stream()) == [1]
    for stage in ("resume", "suspend", "stream_item", "finish"):
        assert f"diagnose:{stage}:{stage}:RuntimeError" in events
    for handle in handles:
        handle.close()

    double_fault = patches.patch_method(
        Worker,
        "stream",
        owner="double-fault",
        handler=FaultyHandler("double-fault", events, "finish_and_suspend"),
    )
    assert list(Worker().stream()) == [1]
    assert "diagnose:double-fault:finish:RuntimeError" in events
    assert "diagnose:double-fault:suspend:RuntimeError" in events
    double_fault.close()


async def test_async_patch_preserves_error_identity_with_and_without_lifecycle() -> None:
    failure = RuntimeError("application failed")

    class Worker:
        async def execute(self) -> None:
            raise failure

    events: list[str] = []
    patches = PatchManager()
    handle = patches.patch_method(
        Worker,
        "execute",
        owner="active",
        handler=RecordingHandler("active", events),
    )
    with pytest.raises(RuntimeError) as caught:
        await Worker().execute()
    assert caught.value is failure
    assert any(event.startswith("finish:None:RuntimeError") for event in events)
    handle.close()

    class NullHandler(RecordingHandler):
        def begin(self, call: InstrumentCall) -> None:
            self.events.append("begin:null")
            return None

    null_handle = patches.patch_method(
        Worker,
        "execute",
        owner="null",
        handler=NullHandler("null", events),
    )
    with pytest.raises(RuntimeError) as caught_without_lifecycle:
        await Worker().execute()
    assert caught_without_lifecycle.value is failure
    null_handle.close()

    class SuccessfulWorker:
        async def execute(self) -> str:
            return "ok"

    success_handle = patches.patch_method(
        SuccessfulWorker,
        "execute",
        owner="null-success",
        handler=NullHandler("null-success", events),
    )
    assert await SuccessfulWorker().execute() == "ok"
    success_handle.close()

    class SyncWorker:
        def execute(self) -> None:
            raise failure

    sync_handle = patches.patch_method(
        SyncWorker,
        "execute",
        owner="null-sync",
        handler=NullHandler("null-sync", events),
    )
    with pytest.raises(RuntimeError) as caught_sync:
        SyncWorker().execute()
    assert caught_sync.value is failure
    sync_handle.close()


def test_patch_rejects_properties_and_non_callable_attributes() -> None:
    class Worker:
        value = 1

        @property
        def label(self) -> str:
            return "worker"

    patches = PatchManager()
    events: list[str] = []
    with pytest.raises(TypeError, match="property instrumentation"):
        patches.patch_method(
            Worker,
            "label",
            owner="property",
            handler=RecordingHandler("property", events),
        )
    with pytest.raises(TypeError, match="is not callable"):
        patches.patch_method(
            Worker,
            "value",
            owner="value",
            handler=RecordingHandler("value", events),
        )


def test_patch_close_is_reentrant_and_restores_targets_when_handler_cleanup_fails() -> None:
    class Worker:
        def execute(self) -> str:
            return "ok"

    events: list[str] = []

    class ReentrantHandler(RecordingHandler):
        def __init__(self) -> None:
            super().__init__("reentrant", events)
            self.handle: InstrumentationHandle | None = None

        def close(self) -> None:
            events.append("close:reentrant")
            assert self.handle is not None
            self.handle.close()

    class BrokenCloseHandler(RecordingHandler):
        def close(self) -> None:
            raise RuntimeError("cleanup failed")

    patches = PatchManager()
    reentrant_handler = ReentrantHandler()
    reentrant_handle = patches.patch_method(
        Worker,
        "execute",
        owner="reentrant",
        handler=reentrant_handler,
    )
    reentrant_handler.handle = reentrant_handle
    retained_handle = patches.patch_method(
        Worker,
        "execute",
        owner="retained",
        handler=RecordingHandler("retained", events),
    )
    patches.close()
    retained_handle.close()
    assert Worker().execute() == "ok"
    assert events.count("close:reentrant") == 1
    assert events.count("close:retained") == 1

    broken_handle = patches.patch_method(
        Worker,
        "execute",
        owner="broken-close",
        handler=BrokenCloseHandler("broken-close", events),
    )
    broken_handle.close()
    assert Worker().execute() == "ok"
    assert "diagnose:broken-close:close:RuntimeError" in events
