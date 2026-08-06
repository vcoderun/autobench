from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from openai import NotGiven, Omit
from pydantic import BaseModel

from autobench.instrumentation.config import AssetDiscoverySettings
from autobench.instrumentation.manager import InstrumentationRuntime
from autobench.instrumentation.models import InstrumentCall, InstrumentorInfo
from autobench.metrics.semantics import Semantic
from autobench.runtime.context import active_run_context
from autobench.tracking import (
    AssetCandidate,
    AssetDefinition,
    AssetProvenance,
    AssetRepresentation,
    AssetSensitivity,
    TrackingRegistry,
    canonical_asset_content,
)


class AssetDiscovery:
    """Extract semantic assets from final OpenAI client request parameters."""

    def __init__(
        self,
        runtime: InstrumentationRuntime,
        info: InstrumentorInfo,
        *,
        target_version: str,
        registry: TrackingRegistry,
        settings: AssetDiscoverySettings,
    ) -> None:
        self.runtime = runtime
        self.info = info
        self.target_version = target_version
        self.registry = registry
        self.settings = settings

    def capture(self, endpoint: str, call: InstrumentCall, *, span_id: str) -> None:
        if endpoint == "openai.chat.completions":
            self._chat(call, span_id=span_id)
        elif endpoint == "openai.responses":
            self._responses(call, span_id=span_id)

    def _chat(self, call: InstrumentCall, *, span_id: str) -> None:
        messages = _sequence(call.kwargs.get("messages"))
        prompt_parts = [
            message
            for message in (_mapping(item) for item in messages)
            if message is not None and message.get("role") in {"system", "developer"}
        ]
        if prompt_parts:
            self._capture(
                endpoint="chat",
                kind="prompt",
                local_id="system",
                name="system",
                content=prompt_parts,
                semantic_type=Semantic.ASSET_RENDERING_VERSION,
                span_id=span_id,
                sensitivity=AssetSensitivity.SENSITIVE,
            )
        tools = [*(_sequence(call.kwargs.get("tools"))), *(_sequence(call.kwargs.get("functions")))]
        self._tools("chat", tools, span_id=span_id)
        response_format = call.kwargs.get("response_format")
        if not _missing(response_format):
            self._output("chat", response_format, span_id=span_id)

    def _responses(self, call: InstrumentCall, *, span_id: str) -> None:
        instructions = call.kwargs.get("instructions")
        if not _missing(instructions):
            self._capture(
                endpoint="responses",
                kind="prompt",
                local_id="instructions",
                name="instructions",
                content=instructions,
                semantic_type=Semantic.ASSET_RENDERING_VERSION,
                span_id=span_id,
                sensitivity=AssetSensitivity.SENSITIVE,
            )
        prompt = call.kwargs.get("prompt")
        if not _missing(prompt):
            self._capture(
                endpoint="responses",
                kind="prompt",
                local_id="managed_prompt",
                name="managed_prompt",
                content=prompt,
                semantic_type=Semantic.ASSET_EXTERNAL_VERSION,
                span_id=span_id,
                sensitivity=AssetSensitivity.PUBLIC,
            )
        input_messages = _sequence(call.kwargs.get("input"))
        prompt_parts = [
            message
            for message in (_mapping(item) for item in input_messages)
            if message is not None and message.get("role") in {"system", "developer"}
        ]
        if prompt_parts:
            self._capture(
                endpoint="responses",
                kind="prompt",
                local_id="input_instructions",
                name="input_instructions",
                content=prompt_parts,
                semantic_type=Semantic.ASSET_RENDERING_VERSION,
                span_id=span_id,
                sensitivity=AssetSensitivity.SENSITIVE,
            )
        self._tools("responses", _sequence(call.kwargs.get("tools")), span_id=span_id)
        text_format = call.kwargs.get("text_format")
        if not _missing(text_format):
            self._output("responses", text_format, span_id=span_id)
            return
        text = _mapping(call.kwargs.get("text"))
        if text is not None and "format" in text:
            self._output("responses", text["format"], span_id=span_id)

    def _tools(self, endpoint: str, tools: Sequence[Any], *, span_id: str) -> None:
        for tool in tools:
            definition = _mapping(tool)
            if definition is None:
                continue
            function = _mapping(definition.get("function"))
            if function is not None:
                name = function.get("name")
            else:
                name = definition.get("name") or definition.get("type")
            if not isinstance(name, str) or not name:
                continue
            effective_alias, definition_locator = self._framework_match("tool", name)
            self._capture(
                endpoint=endpoint,
                kind="tool",
                local_id=name,
                name=name,
                content=definition,
                semantic_type=Semantic.TOOL_VERSION,
                span_id=span_id,
                aliases=(() if effective_alias is None else (effective_alias,)),
                definition_locator=definition_locator,
            )

    def _output(self, endpoint: str, output: Any, *, span_id: str) -> None:
        effective_alias, definition_locator = self._framework_match("output_schema", None)
        self._capture(
            endpoint=endpoint,
            kind="output_schema",
            local_id="output",
            name="output",
            content=output,
            semantic_type=Semantic.OUTPUT_SCHEMA_VERSION,
            span_id=span_id,
            python_target=output if isinstance(output, type) else None,
            aliases=(() if effective_alias is None else (effective_alias,)),
            definition_locator=definition_locator,
        )

    def _framework_match(
        self,
        kind: str,
        name: str | None,
    ) -> tuple[str | None, str | None]:
        context = active_run_context()
        if context is None:
            return None, None
        effective_matches = []
        definition_matches = []
        for use in context.asset_uses:
            try:
                asset = self.registry.asset_by_id(use.asset_id)
            except KeyError:
                continue
            if asset.kind != kind or (name is not None and asset.name != name):
                continue
            if use.provenance.system == "openai":
                continue
            if use.representation is AssetRepresentation.EFFECTIVE:
                effective_matches.append(use)
            else:
                definition_matches.append(use)
        unique = {use.asset_id: use for use in effective_matches}
        if not unique:
            definitions = {use.asset_id: use for use in definition_matches}
            if len(definitions) == 1:
                definition = next(iter(definitions.values()))
                return None, definition.source_locator
            if len(definitions) > 1:
                self.runtime.diagnose(
                    self.info,
                    "asset_correlation_ambiguous",
                    f"multiple framework definitions match OpenAI {kind} {name or 'output'}",
                )
        if len(unique) != 1:
            if len(unique) > 1:
                self.runtime.diagnose(
                    self.info,
                    "asset_correlation_ambiguous",
                    f"multiple effective framework assets match OpenAI {kind} {name or 'output'}",
                )
            return None, None
        match = next(iter(unique.values()))
        definition_locator: str | None = None
        if match.definition_asset_id is not None:
            definition = self.registry.asset_by_id(match.definition_asset_id)
            if isinstance(definition, AssetDefinition) and definition.source_locators:
                definition_locator = definition.source_locators[0]
            else:
                definition_locator = definition.id
        return match.source_locator, definition_locator

    def _capture(
        self,
        *,
        endpoint: str,
        kind: str,
        local_id: str,
        name: str,
        content: Any,
        semantic_type: str,
        span_id: str,
        python_target: Any = None,
        aliases: tuple[str, ...] = (),
        definition_locator: str | None = None,
        sensitivity: AssetSensitivity = AssetSensitivity.INTERNAL,
    ) -> None:
        if not self.settings.allows(kind, AssetRepresentation.EFFECTIVE):
            return
        self.runtime.asset(
            self.info,
            AssetCandidate(
                kind=kind,
                local_id=local_id,
                name=name,
                source_locator=f"openai:{endpoint}:{kind}:{local_id}",
                canonical_content=canonical_asset_content(content),
                representation=AssetRepresentation.EFFECTIVE,
                semantic_type=semantic_type,
                scope=endpoint,
                python_target=python_target,
                definition_locator=definition_locator,
                provenance=AssetProvenance(
                    system="openai",
                    key=local_id,
                    instrumentor=self.info.id,
                    instrumented_library_version=self.target_version,
                ),
                aliases=aliases,
                sensitivity=sensitivity,
            ),
            span_id=span_id,
            registry=self.registry,
        )


def _mapping(value: Any) -> Mapping[str, Any] | None:
    if isinstance(value, Mapping):
        return value
    if isinstance(value, BaseModel):
        return value.model_dump(mode="python", exclude_none=True)
    return None


def _sequence(value: Any) -> Sequence[Any]:
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return value
    return ()


def _missing(value: Any) -> bool:
    return value is None or isinstance(value, (NotGiven, Omit))


__all__ = ("AssetDiscovery",)
