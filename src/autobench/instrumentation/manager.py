from __future__ import annotations

from contextlib import AbstractContextManager
from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError, version
from types import TracebackType
from typing import Any

from packaging.requirements import InvalidRequirement, Requirement
from packaging.specifiers import InvalidSpecifier, SpecifierSet
from packaging.version import InvalidVersion, Version

from autobench.instrumentation.models import (
    Compatibility,
    CompatibilityStatus,
    InstrumentationError,
    InstrumentationHandle,
    Instrumentor,
    InstrumentorInfo,
)
from autobench.instrumentation.patching import CallHandler, PatchManager
from autobench.protocol.collector import Emitter
from autobench.protocol.context import get_context
from autobench.protocol.signals import InstrumentationScope
from autobench.protocol.traces import DiagnosticSeverity


@dataclass(slots=True)
class _Installation:
    instrumentor: Instrumentor
    handle: InstrumentationHandle
    compatibility: Compatibility
    references: int = 1


class InstrumentationRuntime:
    def __init__(self, patches: PatchManager | None = None) -> None:
        self.patches = PatchManager() if patches is None else patches

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
