from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable, Iterator
from contextlib import AbstractContextManager, ExitStack, contextmanager
from dataclasses import dataclass
from pathlib import Path
from threading import RLock
from typing import Protocol, TypeVar

import pytest
from pydantic_ai import Agent
from pydantic_ai.models.test import TestModel

import autobench.runtime.pipeline as pipeline_module
from autobench import (
    DEFAULT_SEMANTIC_REGISTRY,
    BenchmarkInfo,
    BenchmarkSpec,
    Case,
    Compatibility,
    CompositeExtractor,
    CurrentSpan,
    EndReason,
    ExtractionContext,
    FileRecorder,
    InstrumentationHandle,
    InstrumentationManager,
    InstrumentationRuntime,
    InstrumentorCapabilities,
    InstrumentorInfo,
    RunContext,
    Semantic,
    Span,
    TaskResult,
    TaskSpec,
    TaskStatus,
    Variant,
    finalize_staging,
    inspect_staging,
    recover_staging,
    replay_experiment,
    run_benchmark_spec,
    suppress_instrumentation,
)
from autobench.data.datasets import DatasetSpec
from autobench.instrumentation.pydantic_ai import PydanticAI
from autobench.metrics.observations import ObservationSource
from autobench.protocol.collector import Emitter, LocalCollector
from autobench.protocol.context import ActiveContext, attach_context, reset_context
from autobench.protocol.signals import (
    AbstractionLayer,
    CaptureMechanism,
)
from autobench.runtime.instrumentation import reset_active_run_context, set_active_run_context

ResultT = TypeVar("ResultT")
AttributeValue = bool | str | int | float


class BackendSpan(Protocol):
    def set_attribute(self, name: str, value: AttributeValue) -> None: ...


class Backend(Protocol):
    def start_span(
        self,
        name: str,
        *,
        attributes: dict[str, AttributeValue] | None = None,
    ) -> AbstractContextManager[BackendSpan]: ...


class NoopSpan:
    def set_attribute(self, name: str, value: AttributeValue) -> None:
        return None


@dataclass(slots=True)
class RecordedBackendSpan:
    attributes: dict[str, AttributeValue]

    def set_attribute(self, name: str, value: AttributeValue) -> None:
        self.attributes[name] = value


class RecordingBackend:
    def __init__(self) -> None:
        self.started: list[str] = []
        self.finished: list[str] = []
        self.spans: list[RecordedBackendSpan] = []

    @contextmanager
    def start_span(
        self,
        name: str,
        *,
        attributes: dict[str, AttributeValue] | None = None,
    ) -> Iterator[BackendSpan]:
        self.started.append(name)
        span = RecordedBackendSpan(dict(attributes or {}))
        self.spans.append(span)
        try:
            yield span
        finally:
            self.finished.append(name)


class CompositeSpan:
    def __init__(self, spans: tuple[BackendSpan, ...]) -> None:
        self._spans = spans

    def set_attribute(self, name: str, value: AttributeValue) -> None:
        for span in self._spans:
            span.set_attribute(name, value)


class CompositeBackend:
    def __init__(self, backends: tuple[Backend, ...]) -> None:
        self._backends = backends

    @contextmanager
    def start_span(
        self,
        name: str,
        *,
        attributes: dict[str, AttributeValue] | None = None,
    ) -> Iterator[BackendSpan]:
        with ExitStack() as stack:
            spans = tuple(
                stack.enter_context(backend.start_span(name, attributes=attributes))
                for backend in self._backends
            )
            yield CompositeSpan(spans)


class ExternalTelemetry:
    def __init__(self, backend: Backend) -> None:
        self._backend = backend
        self._lock = RLock()

    @property
    def backend(self) -> Backend:
        with self._lock:
            return self._backend

    def install(self, backend: Backend) -> Backend:
        with self._lock:
            previous = self._backend
            self._backend = backend
            return previous

    def restore_if_current(self, *, expected: Backend, replacement: Backend) -> bool:
        with self._lock:
            if self._backend is not expected:
                return False
            self._backend = replacement
            return True


class ExternalSDK:
    def __init__(self, telemetry: ExternalTelemetry) -> None:
        self._telemetry = telemetry

    def run(self, callback: Callable[[], ResultT]) -> ResultT:
        with self._telemetry.backend.start_span(
            "external.workflow",
            attributes={"external.operation": "run"},
        ) as span:
            span.set_attribute("external.mode", "sync")
            return callback()

    async def run_async(self, callback: Callable[[], Awaitable[ResultT]]) -> ResultT:
        with self._telemetry.backend.start_span(
            "external.workflow",
            attributes={"external.operation": "run"},
        ) as span:
            span.set_attribute("external.mode", "async")
            return await callback()


class ABPBackend:
    def __init__(
        self,
        runtime: InstrumentationRuntime,
        info: InstrumentorInfo,
    ) -> None:
        self._runtime = runtime
        self._info = info

    @contextmanager
    def start_span(
        self,
        name: str,
        *,
        attributes: dict[str, AttributeValue] | None = None,
    ) -> Iterator[BackendSpan]:
        span = self._runtime.span(
            self._info,
            name,
            kind="workflow",
            attributes=attributes,
            target_version="1.4.0",
            suppression_keys=("external",),
        )
        if span is None:
            yield NoopSpan()
            return
        with span:
            yield ABPSpan(span)


class ABPSpan:
    def __init__(self, span: Span) -> None:
        self._span = span

    def set_attribute(self, name: str, value: AttributeValue) -> None:
        self._span.set_attribute(name, value)


class ExternalInstrumentor:
    def __init__(self, telemetry: ExternalTelemetry) -> None:
        self._telemetry = telemetry
        self._info = InstrumentorInfo(
            id="external.instrumentor",
            version="1",
            mechanism=CaptureMechanism.HOOK,
            layer=AbstractionLayer.FRAMEWORK,
            span_kinds=("workflow",),
            semantic_families=("external",),
            capabilities=InstrumentorCapabilities(sync=True, async_=True, native_hooks=True),
        )
        self.install_count = 0
        self.close_count = 0
        self.restored: bool | None = None

    @property
    def info(self) -> InstrumentorInfo:
        return self._info

    def check(self) -> Compatibility:
        return Compatibility.compatible(target_version="1.4.0")

    def install(self, runtime: InstrumentationRuntime) -> InstrumentationHandle:
        self.install_count += 1
        previous = self._telemetry.backend
        installed = CompositeBackend((previous, ABPBackend(runtime, self.info)))
        self._telemetry.install(installed)

        def close() -> None:
            self.close_count += 1
            self.restored = self._telemetry.restore_if_current(
                expected=installed,
                replacement=previous,
            )

        return InstrumentationHandle(close, info=self.info)


def run_context() -> RunContext:
    return RunContext(
        benchmark_id="external",
        case=Case(id="case"),
        variant=Variant(id="variant"),
    )


def benchmark_spec() -> BenchmarkSpec:
    return BenchmarkSpec(
        benchmark=BenchmarkInfo(id="external"),
        dataset=DatasetSpec(cases=[Case(id="case")]),
        task=TaskSpec(kind="python", target="unused:run"),
        variants=[Variant(id="variant")],
    )


def test_external_instrumentor_composes_with_existing_backend_and_pydantic_ai() -> None:
    existing = RecordingBackend()
    telemetry = ExternalTelemetry(existing)
    sdk = ExternalSDK(telemetry)
    instrumentor = ExternalInstrumentor(telemetry)
    agent = Agent[None, str](
        TestModel(custom_output_text="composed"),
        deps_type=type(None),
    )
    context = run_context()
    manager = InstrumentationManager()
    assert manager.runtime.installed_ids == ()
    assert manager.runtime.is_installed(instrumentor.info.id) is False
    first = manager.install(instrumentor)
    assert manager.runtime.installed_ids == (instrumentor.info.id,)
    assert manager.runtime.is_installed(instrumentor.info.id) is True
    duplicate = manager.install(instrumentor)
    manager.install(PydanticAI())

    assert sdk.run(lambda: "outside") == "outside"
    assert context.spans == []

    token = set_active_run_context(context)
    try:
        with suppress_instrumentation("external"):
            assert sdk.run(lambda: "suppressed") == "suppressed"
        assert sdk.run(lambda: agent.run_sync("compose").output) == "composed"
    finally:
        reset_active_run_context(token)

    first.close()
    assert telemetry.backend is not existing
    duplicate.close()
    assert telemetry.backend is existing
    assert manager.runtime.is_installed(instrumentor.info.id) is False
    manager.close()

    trace = context.finalize()
    external = next(span for span in trace.spans if span.operation == "external.workflow")
    agent_span = next(span for span in trace.spans if span.operation == "pydantic_ai.agent.run")
    model_spans = [span for span in trace.spans if span.operation == "pydantic_ai.model.request"]
    assert agent_span.parent_span_id == external.span_id
    assert model_spans
    assert all(span.usage.get("requests") == 1 for span in model_spans)
    assert external.usage == {}
    assert external.scope.instrumentor_name == instrumentor.info.id
    assert external.scope.package_version == "1.4.0"
    assert instrumentor.install_count == 1
    assert instrumentor.close_count == 1
    assert instrumentor.restored is True
    assert existing.started == [
        "external.workflow",
        "external.workflow",
        "external.workflow",
    ]
    assert existing.finished == existing.started
    extracted = CompositeExtractor().extract(
        trace,
        registry=DEFAULT_SEMANTIC_REGISTRY,
        context=ExtractionContext(
            run_id=context.run_id,
            benchmark_id=context.benchmark_id,
            experiment_id=context.experiment_id,
            case_id=context.case.id,
            variant_id=context.variant.id,
        ),
    )
    request_summary = next(
        observation
        for observation in extracted.observations
        if observation.semantic_type == Semantic.LLM_REQUEST_COUNT
        and observation.tags.get("abp.summary") is True
    )
    assert request_summary.value == len(model_spans)


def test_runtime_span_rejects_a_protocol_context_without_a_benchmark_run() -> None:
    telemetry = ExternalTelemetry(RecordingBackend())
    instrumentor = ExternalInstrumentor(telemetry)
    runtime = InstrumentationRuntime()
    collector = LocalCollector()
    emitter = Emitter(collector, runtime.scope(instrumentor.info))
    token = attach_context(
        ActiveContext(
            collector=collector,
            trace_id=emitter.trace_id,
            current_span_id=None,
        )
    )
    try:
        assert runtime.span(instrumentor.info, "unowned") is None
        assert runtime.metric(instrumentor.info, "unowned", 1) is None
        assert runtime.current_span(instrumentor.info) is None
    finally:
        reset_context(token)


def test_runtime_span_and_metric_follow_active_run_and_suppression() -> None:
    telemetry = ExternalTelemetry(RecordingBackend())
    instrumentor = ExternalInstrumentor(telemetry)
    runtime = InstrumentationRuntime()
    context = run_context()

    assert context.active_span_id is None
    assert runtime.metric(instrumentor.info, "outside", 1) is None
    assert runtime.current_span(instrumentor.info) is None

    token = set_active_run_context(context)
    try:
        assert context.active_span_id is None
        assert runtime.current_span(instrumentor.info) is None
        with suppress_instrumentation(instrumentor.info.id):
            assert runtime.metric(instrumentor.info, "suppressed", 1) is None
            assert runtime.current_span(instrumentor.info) is None
        span = runtime.span(instrumentor.info, "external.measured")
        assert span is not None
        with span:
            current = runtime.current_span(instrumentor.info)
            assert current is not None
            assert current.id == span.id
            assert current.is_recording() is True
            assert current.get_attribute("external.phase") is None
            assert current.set_attribute("external.phase", "active") is True
            assert current.get_attribute("external.phase") == "active"
            assert current.set_usage("external.items", 3) is True
            recorded_error = current.record_exception(RuntimeError("external failure"))
            event = current.event(
                "external.ready",
                semantic_type="external.ready",
                tags={"scope": "fixture"},
            )
            observation = runtime.metric(
                instrumentor.info,
                "external.items",
                3,
                semantic_type="external.items",
                unit="item",
                tags={"scope": "fixture"},
            )
            assert context.active_span_id == span.id
    finally:
        reset_active_run_context(token)

    assert observation is not None
    assert observation.span_id == span.id
    assert observation.source is ObservationSource.INSTRUMENTATION
    assert observation.tags == {"scope": "fixture"}
    assert event.span_id == span.id
    assert event.source is ObservationSource.INSTRUMENTATION
    assert recorded_error.error_type == "RuntimeError"
    assert recorded_error.message == "external failure"
    assert span.record.usage == {"external.items": 3}
    assert current.is_recording() is False
    assert current.set_attribute("external.phase", "late") is False
    assert current.set_usage("external.items", 4) is False

    stale = CurrentSpan(context, "missing-span")
    assert stale.get_attribute("external.phase") is None

    unrelated = run_context()
    unrelated_token = set_active_run_context(unrelated)
    try:
        assert context.active_span_id is None
    finally:
        reset_active_run_context(unrelated_token)


def test_external_instrumentor_does_not_clobber_a_newer_backend() -> None:
    original = RecordingBackend()
    telemetry = ExternalTelemetry(original)
    instrumentor = ExternalInstrumentor(telemetry)
    manager = InstrumentationManager()
    handle = manager.install(instrumentor)
    newer = RecordingBackend()
    telemetry.install(newer)

    handle.close()

    assert telemetry.backend is newer
    assert instrumentor.restored is False
    manager.close()


async def test_external_instrumentor_evidence_is_staged_finalized_and_replayed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    existing = RecordingBackend()
    telemetry = ExternalTelemetry(existing)
    sdk = ExternalSDK(telemetry)
    instrumentor = ExternalInstrumentor(telemetry)

    async def task(
        target: str,
        *,
        ctx: RunContext,
        case: Case,
        search_paths: tuple[str, ...] = (),
    ) -> TaskResult:
        del target, case, search_paths
        token = set_active_run_context(ctx)
        try:
            output = sdk.run(lambda: "recorded")
            await ctx.checkpoint("external-complete")
            return TaskResult(output=output, status=TaskStatus.PASSED)
        finally:
            reset_active_run_context(token)

    monkeypatch.setattr(pipeline_module, "run_python_task", task)
    output_dir = tmp_path / "record"
    recorder = FileRecorder(output_dir)

    result = await run_benchmark_spec(
        benchmark_spec(),
        experiment_id="external-final",
        recorder=recorder,
        instrumentors=(instrumentor,),
    )

    assert result.runs[0].trace is not None
    assert any(span.operation == "external.workflow" for span in result.runs[0].trace.spans)
    replayed = replay_experiment(output_dir)
    assert replayed.runs[0].trace is not None
    assert any(span.operation == "external.workflow" for span in replayed.runs[0].trace.spans)
    assert not recorder.staging_dir.exists()


async def test_external_instrumentor_cancellation_survives_partial_recovery(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    telemetry = ExternalTelemetry(RecordingBackend())
    sdk = ExternalSDK(telemetry)
    instrumentor = ExternalInstrumentor(telemetry)
    entered = asyncio.Event()

    async def task(
        target: str,
        *,
        ctx: RunContext,
        case: Case,
        search_paths: tuple[str, ...] = (),
    ) -> TaskResult:
        del target, case, search_paths

        async def wait_forever() -> str:
            entered.set()
            await asyncio.Event().wait()
            return "unreachable"

        token = set_active_run_context(ctx)
        try:
            return TaskResult(
                output=await sdk.run_async(wait_forever),
                status=TaskStatus.PASSED,
            )
        finally:
            reset_active_run_context(token)

    monkeypatch.setattr(pipeline_module, "run_python_task", task)
    output_dir = tmp_path / "partial-record"
    recorder = FileRecorder(output_dir)
    execution = asyncio.create_task(
        run_benchmark_spec(
            benchmark_spec(),
            experiment_id="external-cancelled",
            recorder=recorder,
            instrumentors=(instrumentor,),
        )
    )
    await entered.wait()
    execution.cancel()

    with pytest.raises(asyncio.CancelledError):
        await execution

    inspection = inspect_staging(recorder.staging_dir)
    assert inspection.checkpointed_run_ids == (inspection.planned_run_ids[0],)
    recovered = recover_staging(recorder.staging_dir)
    checkpoint_trace = recovered.checkpoints[0].trace
    assert checkpoint_trace is not None
    assert any(span.operation == "external.workflow" for span in checkpoint_trace.spans)
    finalize_staging(recorder.staging_dir, output_dir, allow_partial=True)
    replayed = replay_experiment(output_dir)
    run = replayed.runs[0]
    assert run.end_reason is EndReason.CANCELLED
    assert run.partial is True
    assert run.trace is not None
    external = next(span for span in run.trace.spans if span.operation == "external.workflow")
    assert external.end_reason is EndReason.CANCELLED
    assert external.partial is True
