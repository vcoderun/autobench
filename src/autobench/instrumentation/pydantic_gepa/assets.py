from __future__ import annotations as _annotations

from pydantic_gepa.candidates import Candidate, CandidateComponent

from autobench.instrumentation.config import AssetDiscoverySettings
from autobench.instrumentation.manager import InstrumentationRuntime
from autobench.instrumentation.models import InstrumentorInfo
from autobench.metrics.semantics import Semantic, SemanticType
from autobench.protocol.values import SerializedValue
from autobench.runtime.context import active_run_context
from autobench.tracking import (
    AssetCandidate,
    AssetProvenance,
    AssetRepresentation,
    AssetSensitivity,
    RegisteredAsset,
)


class CandidateAssets:
    def __init__(
        self,
        runtime: InstrumentationRuntime,
        info: InstrumentorInfo,
        settings: AssetDiscoverySettings,
        *,
        target_version: str,
    ) -> None:
        self._runtime = runtime
        self._info = info
        self._settings = settings
        self._target_version = target_version

    def definition(
        self,
        component: CandidateComponent,
        *,
        execution_id: str,
        span_id: str | None,
    ) -> RegisteredAsset | None:
        kind = self._kind(component)
        if not self._settings.allows(kind, AssetRepresentation.DEFINITION):
            return None
        locator = self._locator(component)
        return self._runtime.asset(
            self._info,
            AssetCandidate(
                kind=kind,
                local_id=component.name,
                name=component.name,
                source_locator=locator,
                canonical_content={
                    "name": component.name,
                    "kind": component.kind,
                    "initial_text": component.initial_text,
                    "injection_target": component.injection_target,
                    "source": component.source,
                    "path": component.path,
                    "serialization": component.serialization,
                    "coupled_components": list(component.coupled_components),
                    "optimizable": component.optimizable,
                },
                representation=AssetRepresentation.DEFINITION,
                semantic_type=self._semantic(component),
                scope=f"pydantic-gepa:{execution_id}",
                explicit_asset_id=component.asset_ref,
                owner_locator=self._owner_locator(component),
                provenance=AssetProvenance(
                    system="pydantic-gepa",
                    key="component.definition",
                    path=(component.name,),
                    instrumentor=self._info.id,
                    instrumented_library_version=self._target_version,
                ),
                metadata={
                    "component_kind": component.kind,
                    "optimizable": component.optimizable,
                    "representation": AssetRepresentation.DEFINITION.value,
                },
                sensitivity=AssetSensitivity.SENSITIVE,
            ),
            span_id=span_id,
        )

    def effective(
        self,
        component: CandidateComponent,
        candidate: Candidate,
        *,
        candidate_id: str,
        execution_id: str,
        iteration: int | None,
        status: str,
        span_id: str | None,
    ) -> RegisteredAsset | None:
        kind = self._kind(component)
        if not self._settings.allows(kind, AssetRepresentation.EFFECTIVE):
            return None
        encoded = candidate.values.get(component.name)
        if encoded is None:
            return None
        definition_locator = self._locator(component)
        locator = f"{definition_locator}:effective"
        metadata: dict[str, SerializedValue] = {
            "component_kind": component.kind,
            "optimizable": component.optimizable,
            "representation": AssetRepresentation.EFFECTIVE.value,
        }
        return self._runtime.asset(
            self._info,
            AssetCandidate(
                kind=kind,
                local_id=component.name,
                name=component.name,
                source_locator=locator,
                canonical_content=component.decode(encoded),
                representation=AssetRepresentation.EFFECTIVE,
                semantic_type=self._semantic(component),
                scope=f"pydantic-gepa:{execution_id}",
                explicit_asset_id=component.asset_ref,
                owner_locator=self._owner_locator(component),
                definition_locator=definition_locator,
                provenance=AssetProvenance(
                    system="pydantic-gepa",
                    key="candidate.component",
                    path=(
                        candidate_id,
                        status,
                        str(candidate.generation),
                        str(iteration),
                        component.name,
                    ),
                    instrumentor=self._info.id,
                    instrumented_library_version=self._target_version,
                ),
                metadata=metadata,
                sensitivity=AssetSensitivity.SENSITIVE,
            ),
            span_id=span_id,
        )

    @staticmethod
    def _kind(component: CandidateComponent) -> str:
        return {
            "instructions": "prompt",
            "system_prompt": "prompt",
            "input_schema": "input_schema",
            "output_schema": "output_schema",
            "tool_schema": "tool",
            "field_description": "field_description",
            "schema_description": "schema_description",
            "custom": "optimization_component",
        }[component.kind]

    @staticmethod
    def _semantic(component: CandidateComponent) -> SemanticType | None:
        if component.semantic_type is not None:
            return component.semantic_type
        return {
            "instructions": Semantic.PROMPT_VERSION,
            "system_prompt": Semantic.PROMPT_VERSION,
            "input_schema": None,
            "output_schema": Semantic.OUTPUT_SCHEMA_VERSION,
            "tool_schema": Semantic.TOOL_VERSION,
            "field_description": Semantic.OUTPUT_SCHEMA_VERSION,
            "schema_description": Semantic.OUTPUT_SCHEMA_VERSION,
            "custom": None,
        }[component.kind]

    @staticmethod
    def _owner_locator(component: CandidateComponent) -> str | None:
        if component.kind not in {"field_description", "schema_description"}:
            return None
        if component.source is not None:
            return component.source
        return component.path

    @staticmethod
    def _locator(component: CandidateComponent) -> str:
        if component.asset_ref is not None:
            return component.asset_ref
        source = component.source or component.path
        if source is not None:
            return f"pydantic-gepa:{source}#{component.name}"
        context = active_run_context()
        benchmark_id = "benchmark" if context is None else context.benchmark_id
        return f"pydantic-gepa:{benchmark_id}:component:{component.name}"


__all__ = ("CandidateAssets",)
