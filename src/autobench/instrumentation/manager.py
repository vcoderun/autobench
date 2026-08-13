from __future__ import annotations

from contextlib import AbstractContextManager
from dataclasses import dataclass
from enum import Enum, auto
from importlib.metadata import PackageNotFoundError, version
from threading import RLock
from types import TracebackType
from typing import TYPE_CHECKING, Any

from packaging.requirements import InvalidRequirement, Requirement
from packaging.specifiers import InvalidSpecifier, SpecifierSet
from packaging.version import InvalidVersion, Version
from pydantic import JsonValue

from autobench.instrumentation.models import (
    Compatibility,
    CompatibilityStatus,
    InstrumentationError,
    InstrumentationHandle,
    Instrumentor,
    InstrumentorInfo,
)
from autobench.instrumentation.patching import CallHandler, PatchManager
from autobench.metrics.observations import Direction, ObservationRole, ObservationSource
from autobench.metrics.semantics import SemanticType
from autobench.protocol.collector import Emitter
from autobench.protocol.context import get_context
from autobench.protocol.ids import TraceId
from autobench.protocol.signals import (
    EndReason,
    InstrumentationScope,
    LinkRelation,
    LinkTarget,
    SpanStatus,
)
from autobench.protocol.traces import DiagnosticSeverity
from autobench.protocol.values import SerializedValue
from autobench.tracking import AssetCandidate, RegisteredAsset, TrackingRegistry, track

if TYPE_CHECKING:
    from autobench.errors import ErrorRecord
    from autobench.metrics.observations import Observation
    from autobench.runtime.context import RunContext, Span


@dataclass(slots=True)
class _Installation:
    instrumentor: Instrumentor
    handle: InstrumentationHandle
    compatibility: Compatibility
    references: int = 1


@dataclass(slots=True)
class _KeyedSpan:
    context: RunContext
    span: Span


class _Unset(Enum):
    VALUE = auto()


class CurrentSpan:
    """Mutable view of the active Autobench span for an external backend."""

    def __init__(self, context: RunContext, span_id: str) -> None:
        self._context = context
        self._span_id = span_id

    @property
    def id(self) -> str:
        return self._span_id

    def is_recording(self) -> bool:
        record = self._context._span_by_id(self._span_id)
        return record is not None and record.ended_at is None and not self._context.finalized

    def get_attribute(self, name: str) -> Any | None:
        record = self._context._span_by_id(self._span_id)
        if record is None:
            return None
        return record.attributes.get(name)

    def set_attribute(self, name: str, value: Any) -> bool:
        record = self._context._span_by_id(self._span_id)
        if record is None or record.ended_at is not None or self._context.finalized:
            return False
        record.attributes[name] = value
        return True

    def set_usage(self, name: str, value: Any) -> bool:
        record = self._context._span_by_id(self._span_id)
        if record is None or record.ended_at is not None or self._context.finalized:
            return False
        record.usage[name] = value
        return True

    def set_output(self, value: Any) -> bool:
        record = self._context._span_by_id(self._span_id)
        if record is None or record.ended_at is not None or self._context.finalized:
            return False
        record.output = value
        return True

    def event(
        self,
        name: str,
        value: Any = True,
        *,
        semantic_type: SemanticType | None = None,
        tags: dict[str, Any] | None = None,
    ) -> Observation:
        return self._context.event(
            name,
            value,
            semantic_type=semantic_type,
            span_id=self._span_id,
            tags=tags,
            source=ObservationSource.INSTRUMENTATION,
        )

    def record_exception(self, error: BaseException | str) -> ErrorRecord:
        return self._context.error(error, span_id=self._span_id)

    def link_to(
        self,
        target: CurrentSpan,
        *,
        relation: LinkRelation = LinkRelation.RUN_LINEAGE,
        attributes: dict[str, SerializedValue] | None = None,
    ) -> bool:
        if not self.is_recording() or target._context._span_by_id(target.id) is None:
            return False
        self._context._emitter_for_legacy_span(self._span_id).link(
            self._context._abp_span_id(self._span_id),
            relation,
            LinkTarget(
                trace_id=target._context._emitter.trace_id,
                span_id=target._context._abp_span_id(target.id),
            ),
            attributes=attributes,
        )
        return True


class InstrumentationRuntime:
    def __init__(
        self,
        patches: PatchManager | None = None,
        *,
        registry: TrackingRegistry = track,
    ) -> None:
        self.patches = PatchManager() if patches is None else patches
        self.registry = registry
        self._installed_ids: set[str] = set()
        self._keyed_spans: dict[tuple[TraceId, str, str], _KeyedSpan] = {}
        self._keyed_span_lock = RLock()

    @property
    def installed_ids(self) -> tuple[str, ...]:
        return tuple(sorted(self._installed_ids))

    def is_installed(self, instrumentor_id: str) -> bool:
        return instrumentor_id in self._installed_ids

    def patch_method(
        self,
        info: InstrumentorInfo,
        target: type[Any],
        attribute: str,
        handler: CallHandler,
        *,
        expected_descriptor: Any = None,
    ) -> InstrumentationHandle:
        return self.patches.patch_method(
            target,
            attribute,
            owner=info.id,
            handler=handler,
            expected_descriptor=expected_descriptor,
        )

    def scope(
        self,
        info: InstrumentorInfo,
        *,
        target_version: str | None = None,
    ) -> InstrumentationScope:
        return InstrumentationScope(
            instrumentor_name=info.id,
            instrumentor_version=info.version,
            package_name=info.target_distribution or "autobench",
            package_version=target_version or info.version,
            mechanism=info.mechanism,
            layer=info.layer,
            source_convention=info.source_convention,
            source_convention_version=info.source_convention_version,
        )

    def diagnose(
        self,
        info: InstrumentorInfo,
        code: str,
        message: str,
        *,
        severity: DiagnosticSeverity = DiagnosticSeverity.WARNING,
    ) -> bool:
        active = get_context()
        if active is None:
            return False
        emitter = Emitter(
            active.collector,
            self.scope(info),
            trace_id=active.trace_id,
            execution=active.execution,
        )
        try:
            emitter.diagnostic(
                code,
                message,
                severity=severity,
                span_id=active.current_span_id,
            )
        except RuntimeError:
            return False
        return True

    def span(
        self,
        info: InstrumentorInfo,
        operation: str,
        *,
        kind: str = "custom",
        input: Any = None,
        attributes: dict[str, Any] | None = None,
        usage: dict[str, Any] | None = None,
        tags: dict[str, Any] | None = None,
        target_version: str | None = None,
        suppression_keys: tuple[str, ...] = (),
        parent_span_id: str | None = None,
    ) -> Span | None:
        """Create an unentered span in the active benchmark run, when one exists."""

        active = get_context()
        if active is None or active.is_suppressed(info.id, *suppression_keys):
            return None

        from autobench.runtime.context import active_run_context

        run_context = active_run_context()
        if run_context is None:
            return None
        return run_context.span(
            operation,
            kind=kind,
            input=input,
            attributes=attributes,
            usage=usage,
            tags=tags,
            instrumentation_scope=self.scope(info, target_version=target_version),
            parent_span_id=parent_span_id,
        )

    def start_span(
        self,
        info: InstrumentorInfo,
        key: str,
        operation: str,
        *,
        parent_key: str | None = None,
        parent_span_id: str | None = None,
        kind: str = "custom",
        input: Any = None,
        attributes: dict[str, Any] | None = None,
        usage: dict[str, Any] | None = None,
        tags: dict[str, Any] | None = None,
        target_version: str | None = None,
        suppression_keys: tuple[str, ...] = (),
    ) -> CurrentSpan | None:
        """Start a detached span identified by an external lifecycle key."""

        if not key:
            raise ValueError("Span keys must not be empty.")
        if parent_key is not None and parent_span_id is not None:
            raise ValueError("Use parent_key or parent_span_id, not both.")
        active = get_context()
        if active is None or active.is_suppressed(info.id, *suppression_keys):
            return None

        from autobench.runtime.context import active_run_context

        run_context = active_run_context()
        if run_context is None:
            return None
        lookup = (active.trace_id, info.id, key)
        with self._keyed_span_lock:
            existing = self._keyed_spans.get(lookup)
            if existing is not None:
                self.diagnose(
                    info,
                    "keyed_span_duplicate_start",
                    f"span key '{key}' is already active",
                )
                return CurrentSpan(existing.context, existing.span.id)

            resolved_parent_id = parent_span_id
            if parent_key is not None:
                parent = self._keyed_spans.get((active.trace_id, info.id, parent_key))
                if parent is None or parent.span.record.ended_at is not None:
                    self.diagnose(
                        info,
                        "keyed_span_parent_missing",
                        f"parent span key '{parent_key}' is not active for '{key}'",
                    )
                    return None
                resolved_parent_id = parent.span.id

            span = run_context.span(
                operation,
                kind=kind,
                input=input,
                attributes=attributes,
                usage=usage,
                tags=tags,
                instrumentation_scope=self.scope(info, target_version=target_version),
                parent_span_id=resolved_parent_id,
            )
            span.start()
            self._keyed_spans[lookup] = _KeyedSpan(run_context, span)
            return CurrentSpan(run_context, span.id)

    def span_for_key(
        self,
        info: InstrumentorInfo,
        key: str,
        *,
        suppression_keys: tuple[str, ...] = (),
    ) -> CurrentSpan | None:
        active = get_context()
        if active is None or active.is_suppressed(info.id, *suppression_keys):
            return None
        lookup = (active.trace_id, info.id, key)
        with self._keyed_span_lock:
            keyed = self._keyed_spans.get(lookup)
            if keyed is None:
                return None
            if keyed.context.finalized or keyed.span.record.ended_at is not None:
                del self._keyed_spans[lookup]
                return None
            return CurrentSpan(keyed.context, keyed.span.id)

    def end_span(
        self,
        info: InstrumentorInfo,
        key: str,
        *,
        output: Any | _Unset = _Unset.VALUE,
        attributes: dict[str, Any] | None = None,
        usage: dict[str, Any] | None = None,
        error: BaseException | str | None = None,
        status: SpanStatus | None = None,
        reason: EndReason | None = None,
        partial: bool | None = None,
        suppression_keys: tuple[str, ...] = (),
    ) -> bool:
        """Finish one keyed span without changing the active context stack."""

        active = get_context()
        if active is None or active.is_suppressed(info.id, *suppression_keys):
            return False
        lookup = (active.trace_id, info.id, key)
        with self._keyed_span_lock:
            keyed = self._keyed_spans.pop(lookup, None)
        if keyed is None:
            self.diagnose(
                info,
                "keyed_span_missing_end",
                f"span key '{key}' is not active",
            )
            return False
        if keyed.context.finalized or keyed.span.record.ended_at is not None:
            self.diagnose(
                info,
                "keyed_span_duplicate_end",
                f"span key '{key}' has already ended",
            )
            return False
        if output is not _Unset.VALUE:
            keyed.span.set_output(output)
        for name, value in (attributes or {}).items():
            keyed.span.set_attribute(name, value)
        for name, value in (usage or {}).items():
            keyed.span.set_usage(name, value)
        if isinstance(error, str):
            keyed.context.error(error, span_id=keyed.span.id)
            keyed.span.finish(
                status=SpanStatus.ERROR,
                reason=EndReason.FAILED if reason is None else reason,
                partial=partial,
            )
        else:
            keyed.span.finish(error=error, status=status, reason=reason, partial=partial)
        return True

    def metric(
        self,
        info: InstrumentorInfo,
        name: str,
        value: Any,
        *,
        semantic_type: SemanticType | None = None,
        unit: str | None = None,
        direction: Direction | None = None,
        role: ObservationRole | None = None,
        tags: dict[str, Any] | None = None,
        suppression_keys: tuple[str, ...] = (),
        span_key: str | None = None,
        span_id: str | None = None,
    ) -> Observation | None:
        """Record an instrumentation observation in the active benchmark run."""

        active = get_context()
        if active is None or active.is_suppressed(info.id, *suppression_keys):
            return None

        from autobench.runtime.context import active_run_context

        run_context = active_run_context()
        if run_context is None:
            return None
        if span_key is not None and span_id is not None:
            raise ValueError("Use span_key or span_id, not both.")
        target_span_id = span_id
        if span_key is not None:
            target = self.span_for_key(info, span_key, suppression_keys=suppression_keys)
            if target is None:
                self.diagnose(
                    info,
                    "keyed_span_metric_target_missing",
                    f"span key '{span_key}' is not active",
                )
                return None
            target_span_id = target.id
        if target_span_id is None:
            target_span_id = run_context.active_span_id
        return run_context.metric(
            name,
            value,
            semantic_type=semantic_type,
            unit=unit,
            direction=direction,
            role=role,
            span_id=target_span_id,
            tags=tags,
            source=ObservationSource.INSTRUMENTATION,
        )

    def event(
        self,
        info: InstrumentorInfo,
        name: str,
        value: Any = True,
        *,
        semantic_type: SemanticType | None = None,
        tags: dict[str, Any] | None = None,
        span_key: str | None = None,
        span_id: str | None = None,
        suppression_keys: tuple[str, ...] = (),
    ) -> Observation | None:
        """Record an instrumentation event on the active or selected span."""

        active = get_context()
        if active is None or active.is_suppressed(info.id, *suppression_keys):
            return None

        from autobench.runtime.context import active_run_context

        run_context = active_run_context()
        if run_context is None:
            return None
        if span_key is not None and span_id is not None:
            raise ValueError("Use span_key or span_id, not both.")
        target_span_id = span_id
        if span_key is not None:
            target = self.span_for_key(info, span_key, suppression_keys=suppression_keys)
            if target is None:
                self.diagnose(
                    info,
                    "keyed_span_event_target_missing",
                    f"span key '{span_key}' is not active",
                )
                return None
            target_span_id = target.id
        if target_span_id is None:
            target_span_id = run_context.active_span_id
        return run_context.event(
            name,
            value,
            semantic_type=semantic_type,
            span_id=target_span_id,
            tags=tags,
            source=ObservationSource.INSTRUMENTATION,
        )

    def set_extension(
        self,
        info: InstrumentorInfo,
        name: str,
        value: JsonValue,
        *,
        suppression_keys: tuple[str, ...] = (),
    ) -> bool:
        active = get_context()
        if active is None or active.is_suppressed(info.id, *suppression_keys):
            return False

        from autobench.runtime.context import active_run_context

        run_context = active_run_context()
        if run_context is None:
            return False
        run_context.set_extension(name, value)
        return True

    def current_span(
        self,
        info: InstrumentorInfo,
        *,
        suppression_keys: tuple[str, ...] = (),
    ) -> CurrentSpan | None:
        """Return a mutable view of the active span owned by the benchmark run."""

        active = get_context()
        if active is None or active.is_suppressed(info.id, *suppression_keys):
            return None

        from autobench.runtime.context import active_run_context

        run_context = active_run_context()
        if run_context is None:
            return None
        span_id = run_context.active_span_id
        if span_id is None:
            return None
        return CurrentSpan(run_context, span_id)

    def asset(
        self,
        info: InstrumentorInfo,
        candidate: AssetCandidate,
        *,
        span_id: str | None = None,
        registry: TrackingRegistry | None = None,
    ) -> RegisteredAsset | None:
        """Register and attach one SDK-observed asset without affecting the host call."""

        active = get_context()
        if active is None or active.is_suppressed(info.id, "assets"):
            return None
        try:
            from autobench.runtime.context import active_run_context

            run_context = active_run_context()
            if run_context is None:
                return None
            active_registry = self.registry if registry is None else registry
            prepared = run_context.prepare_discovered_asset(candidate, span_id=span_id)
            registered = active_registry.register_candidate(prepared, span_id=span_id)
            run_context.attach_discovered_asset(registered)
            return registered
        except Exception as error:
            self.diagnose(
                info,
                "asset_discovery_failed",
                f"{candidate.source_locator}: {type(error).__name__}: {error}",
            )
            return None


class InstrumentationManager(AbstractContextManager["InstrumentationManager"]):
    def __init__(self, runtime: InstrumentationRuntime | None = None) -> None:
        self.runtime = InstrumentationRuntime() if runtime is None else runtime
        self._installations: dict[str, _Installation] = {}
        self._closed = False

    @property
    def installed(self) -> tuple[InstrumentorInfo, ...]:
        return tuple(
            installation.instrumentor.info for installation in self._installations.values()
        )

    def check(self, instrumentor: Instrumentor) -> Compatibility:
        declared = instrumentor.check()
        if not declared.installable:
            return declared
        package_compatibility = check_package_compatibility(instrumentor.info)
        if not package_compatibility.installable:
            return package_compatibility
        degraded_features = declared.degraded_features + package_compatibility.degraded_features
        diagnostics = declared.diagnostics + package_compatibility.diagnostics
        status = (
            CompatibilityStatus.DEGRADED
            if degraded_features or declared.status is CompatibilityStatus.DEGRADED
            else CompatibilityStatus.COMPATIBLE
        )
        return Compatibility(
            status=status,
            target_version=package_compatibility.target_version or declared.target_version,
            degraded_features=degraded_features,
            conflicts=declared.conflicts,
            diagnostics=diagnostics,
            private_seam_supported=declared.private_seam_supported,
        )

    def install(self, instrumentor: Instrumentor) -> InstrumentationHandle:
        if self._closed:
            raise InstrumentationError("instrumentation manager is closed")
        info = instrumentor.info
        existing = self._installations.get(info.id)
        if existing is not None:
            if existing.instrumentor.info.version != info.version:
                raise InstrumentationError(
                    f"instrumentor '{info.id}' version {existing.instrumentor.info.version} "
                    f"is already installed; cannot install {info.version}"
                )
            existing.references += 1
            return InstrumentationHandle(lambda: self._release(info.id), info=info)

        compatibility = self.check(instrumentor)
        if not compatibility.installable:
            detail = (
                "; ".join(compatibility.conflicts + compatibility.diagnostics)
                or compatibility.status.value
            )
            raise InstrumentationError(f"instrumentor '{info.id}' is not installable: {detail}")
        native_handle = instrumentor.install(self.runtime)
        self._installations[info.id] = _Installation(
            instrumentor=instrumentor,
            handle=native_handle,
            compatibility=compatibility,
        )
        self.runtime._installed_ids.add(info.id)
        return InstrumentationHandle(lambda: self._release(info.id), info=info)

    def close(self) -> None:
        if self._closed:
            return
        for instrumentor_id in tuple(reversed(self._installations)):
            installation = self._installations[instrumentor_id]
            installation.references = 1
            self._release(instrumentor_id)
        self.runtime.patches.close()
        self._closed = True

    def _release(self, instrumentor_id: str) -> None:
        installation = self._installations.get(instrumentor_id)
        if installation is None:
            return
        installation.references -= 1
        if installation.references > 0:
            return
        installation.handle.close()
        del self._installations[instrumentor_id]
        self.runtime._installed_ids.discard(instrumentor_id)

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> bool | None:
        self.close()
        return None


def check_package_compatibility(info: InstrumentorInfo) -> Compatibility:
    distribution = info.target_distribution
    target_version = None
    if distribution is not None:
        try:
            target_version = version(distribution)
        except PackageNotFoundError:
            return Compatibility(
                status=CompatibilityStatus.UNAVAILABLE,
                diagnostics=(f"distribution '{distribution}' is not installed",),
            )
        if info.supported_versions is not None:
            try:
                supported = Version(target_version) in SpecifierSet(info.supported_versions)
            except (InvalidSpecifier, InvalidVersion) as exc:
                return Compatibility(
                    status=CompatibilityStatus.UNSUPPORTED,
                    target_version=target_version,
                    diagnostics=(f"invalid version compatibility declaration: {exc}",),
                )
            if not supported:
                return Compatibility(
                    status=CompatibilityStatus.UNSUPPORTED,
                    target_version=target_version,
                    diagnostics=(
                        f"distribution '{distribution}' {target_version} is outside "
                        f"{info.supported_versions}",
                    ),
                )

    degraded_features: list[str] = []
    diagnostics: list[str] = []
    for declaration in info.optional_dependencies:
        try:
            requirement = Requirement(declaration)
        except InvalidRequirement as exc:
            degraded_features.append(declaration)
            diagnostics.append(f"invalid optional dependency declaration '{declaration}': {exc}")
            continue
        if requirement.marker is not None and not requirement.marker.evaluate():
            continue
        try:
            dependency_version = Version(version(requirement.name))
        except (PackageNotFoundError, InvalidVersion):
            degraded_features.append(requirement.name)
            diagnostics.append(f"optional dependency '{declaration}' is unavailable")
            continue
        if requirement.specifier and dependency_version not in requirement.specifier:
            degraded_features.append(requirement.name)
            diagnostics.append(
                f"optional dependency '{requirement.name}' {dependency_version} is outside "
                f"{requirement.specifier}"
            )

    if degraded_features:
        return Compatibility(
            status=CompatibilityStatus.DEGRADED,
            target_version=target_version,
            degraded_features=tuple(degraded_features),
            diagnostics=tuple(diagnostics),
        )
    return Compatibility.compatible(target_version=target_version)


__all__ = (
    "InstrumentationManager",
    "InstrumentationRuntime",
    "check_package_compatibility",
)
