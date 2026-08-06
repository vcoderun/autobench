from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from pydantic_ai import RunContext as AgentRunContext
from pydantic_ai.capabilities import AbstractCapability
from pydantic_ai.messages import InstructionPart, ModelRequest, SystemPromptPart
from pydantic_ai.models import ModelRequestContext
from pydantic_ai.output import OutputContext
from pydantic_ai.tools import Tool, ToolDefinition
from pydantic_ai.toolsets import AbstractToolset
from pydantic_ai.toolsets.function import FunctionToolset

from autobench.instrumentation.config import AssetDiscoverySettings
from autobench.instrumentation.manager import InstrumentationRuntime
from autobench.instrumentation.models import InstrumentorInfo
from autobench.metrics.semantics import Semantic
from autobench.tracking import (
    AssetCandidate,
    AssetProvenance,
    AssetRepresentation,
    AssetSensitivity,
    TrackingRegistry,
    canonical_asset_content,
)


class AssetDiscovery:
    """Normalize Pydantic AI definitions and resolved requests into Autobench assets."""

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

    def agent(
        self,
        ctx: AgentRunContext[Any],
        *,
        span_id: str,
        overrides: Mapping[str, Any],
    ) -> None:
        agent = ctx.agent
        if agent is None:
            return
        scope = _agent_scope(ctx)
        child_locators: list[str] = []

        description = agent.description
        if description is not None:
            child_locators.append(
                self._capture(
                    scope=scope,
                    kind="prompt",
                    local_id="description",
                    content=description,
                    semantic_type=Semantic.PROMPT_VERSION,
                    span_id=span_id,
                    sensitivity=AssetSensitivity.SENSITIVE,
                )
            )

        child_locators.append(
            self._output_type(
                scope,
                "output",
                agent.output_type,
                span_id=span_id,
            )
        )
        for toolset in agent.toolsets:
            child_locators.extend(self._toolset(scope, toolset, span_id=span_id))

        for runtime_id, capability in sorted(ctx.capabilities.items()):
            if capability.id == "autobench":
                continue
            child_locators.extend(self._capability(runtime_id, capability, span_id=span_id))

        child_locators.extend(self._runtime_overrides(scope, overrides, span_id=span_id))
        self._capture(
            scope=scope,
            kind="agent",
            local_id="self",
            name=agent.name or "agent",
            content={
                "name": agent.name,
                "description": description,
                "children": sorted(set(child_locators)),
            },
            semantic_type=Semantic.AGENT_VERSION,
            span_id=span_id,
        )

    def request(
        self,
        ctx: AgentRunContext[Any],
        request_context: ModelRequestContext,
        *,
        span_id: str,
    ) -> None:
        scope = _agent_scope(ctx)
        parameters = request_context.model_request_parameters
        system_parts = [
            part
            for message in request_context.messages
            if isinstance(message, ModelRequest)
            for part in message.parts
            if isinstance(part, SystemPromptPart)
        ]
        self._resolved_prompts(
            scope,
            system_parts,
            parameters.instruction_parts or [],
            span_id=span_id,
        )
        for tool in parameters.function_tools:
            owner_scope = tool.capability_id or scope
            self._effective_tool(owner_scope, tool, span_id=span_id)
        for tool in parameters.output_tools:
            owner_scope = tool.capability_id or scope
            self._effective_output_tool(owner_scope, tool, span_id=span_id)
        for native_tool in parameters.native_tools:
            local_id = type(native_tool).__name__
            self._capture(
                scope=scope,
                kind="tool",
                local_id=f"native:{local_id}",
                name=local_id,
                content=native_tool,
                semantic_type=Semantic.TOOL_VERSION,
                representation=AssetRepresentation.EFFECTIVE,
                span_id=span_id,
            )
        if parameters.output_object is not None:
            self._capture(
                scope=scope,
                kind="output_schema",
                local_id="output",
                content=parameters.output_object,
                semantic_type=Semantic.OUTPUT_SCHEMA_VERSION,
                representation=AssetRepresentation.EFFECTIVE,
                definition_locator=_locator(scope, "output_schema", "output"),
                span_id=span_id,
            )
        if parameters.prompted_output_template not in (None, False):
            self._capture(
                scope=scope,
                kind="prompt",
                local_id="prompted_output",
                content=parameters.prompted_output_template,
                semantic_type=Semantic.PROMPT_VERSION,
                representation=AssetRepresentation.EFFECTIVE,
                span_id=span_id,
                sensitivity=AssetSensitivity.SENSITIVE,
            )

    def output(self, ctx: AgentRunContext[Any], output: OutputContext, *, span_id: str) -> None:
        if output.output_type is None:
            return
        scope = _agent_scope(ctx)
        self._capture(
            scope=scope,
            kind="output_schema",
            local_id="validated_output",
            content=output.output_type,
            semantic_type=Semantic.OUTPUT_SCHEMA_VERSION,
            python_target=output.output_type if isinstance(output.output_type, type) else None,
            representation=AssetRepresentation.EFFECTIVE,
            definition_locator=_locator(scope, "output_schema", "output"),
            span_id=span_id,
        )

    def _runtime_overrides(
        self,
        scope: str,
        overrides: Mapping[str, Any],
        *,
        span_id: str,
    ) -> list[str]:
        locators: list[str] = []
        instructions = overrides.get("instructions")
        if instructions is not None:
            locators.append(
                self._capture(
                    scope=scope,
                    kind="prompt",
                    local_id="runtime_instructions",
                    content=instructions,
                    semantic_type=Semantic.PROMPT_VERSION,
                    python_target=instructions if callable(instructions) else None,
                    span_id=span_id,
                    sensitivity=AssetSensitivity.SENSITIVE,
                )
            )
        output_type = overrides.get("output_type")
        if output_type is not None:
            locators.append(
                self._output_type(scope, "runtime_output", output_type, span_id=span_id)
            )
        toolsets = overrides.get("toolsets")
        if isinstance(toolsets, Sequence) and not isinstance(toolsets, (str, bytes, bytearray)):
            for toolset in toolsets:
                if isinstance(toolset, AbstractToolset):
                    locators.extend(self._toolset(scope, toolset, span_id=span_id))
        spec = overrides.get("spec")
        if spec is not None:
            locators.append(
                self._capture(
                    scope=scope,
                    kind="policy",
                    local_id="runtime_spec",
                    content=spec,
                    semantic_type=Semantic.POLICY_VERSION,
                    span_id=span_id,
                )
            )
        return locators

    def _toolset(
        self,
        scope: str,
        toolset: AbstractToolset[Any],
        *,
        span_id: str,
    ) -> list[str]:
        child_locators: list[str] = []
        toolset_id = toolset.id or f"{type(toolset).__module__}.{type(toolset).__qualname__}"
        if isinstance(toolset, FunctionToolset):
            for tool_name, tool in sorted(toolset.tools.items()):
                child_locators.append(self._source_tool(scope, tool_name, tool, span_id=span_id))
        locator = self._capture(
            scope=scope,
            kind="toolset",
            local_id=toolset_id,
            name=toolset.id or type(toolset).__name__,
            content={
                "class": f"{type(toolset).__module__}.{type(toolset).__qualname__}",
                "id": toolset.id,
                "children": child_locators,
            },
            semantic_type=Semantic.TOOLSET_VERSION,
            span_id=span_id,
        )
        return [*child_locators, locator]

    def _source_tool(
        self,
        scope: str,
        tool_name: str,
        tool: Tool[Any],
        *,
        span_id: str,
    ) -> str:
        return self._capture(
            scope=scope,
            kind="tool",
            local_id=tool_name,
            name=tool.name,
            content={
                "name": tool.name,
                "description": tool.description,
                "parameters": tool.function_schema.json_schema,
                "returns": tool.function_schema.return_schema,
                "strict": tool.strict,
                "sequential": tool.sequential,
                "requires_approval": tool.requires_approval,
                "defer_loading": tool.defer_loading,
                "include_return_schema": tool.include_return_schema,
                "prepare": canonical_asset_content(tool.prepare),
                "validator": canonical_asset_content(tool.args_validator),
            },
            semantic_type=Semantic.TOOL_VERSION,
            python_target=tool.function,
            span_id=span_id,
        )

    def _capability(
        self,
        runtime_id: str,
        capability: AbstractCapability[Any],
        *,
        span_id: str,
    ) -> list[str]:
        scope = _capability_name(capability)
        components: list[str] = []
        contribution_methods = (
            ("get_instructions", "prompt", "instructions", Semantic.PROMPT_VERSION),
            ("get_description", "prompt", "description", Semantic.PROMPT_VERSION),
            ("get_toolset", "toolset", "tools", Semantic.TOOLSET_VERSION),
            ("get_native_tools", "toolset", "native_tools", Semantic.TOOLSET_VERSION),
            ("get_wrapper_toolset", "policy", "tool_wrapper", Semantic.POLICY_VERSION),
            ("prepare_tools", "policy", "prepare_tools", Semantic.POLICY_VERSION),
            ("prepare_output_tools", "policy", "prepare_output", Semantic.POLICY_VERSION),
            ("wrap_output_validate", "policy", "output_validator", Semantic.POLICY_VERSION),
            ("wrap_output_process", "policy", "output_processor", Semantic.POLICY_VERSION),
        )
        capability_type = type(capability)
        for method_name, kind, local_id, semantic_type in contribution_methods:
            method = getattr(capability_type, method_name)
            if method is getattr(AbstractCapability, method_name):
                continue
            components.append(
                self._capture(
                    scope=scope,
                    kind=kind,
                    local_id=local_id,
                    content=method,
                    semantic_type=semantic_type,
                    python_target=method,
                    span_id=span_id,
                    aliases=(f"{scope}:{kind}:{local_id}",),
                    sensitivity=(
                        AssetSensitivity.SENSITIVE
                        if kind == "prompt"
                        else AssetSensitivity.INTERNAL
                    ),
                )
            )
        locator = self._capture(
            scope=scope,
            kind="capability",
            local_id="self",
            name=scope,
            content={
                "runtime_id": runtime_id,
                "class": f"{capability_type.__module__}.{capability_type.__qualname__}",
                "id": capability.id,
                "description": capability.description,
                "defer_loading": capability.defer_loading,
                "components": components,
            },
            semantic_type=Semantic.CAPABILITY_VERSION,
            span_id=span_id,
            aliases=(f"{scope}:capability:self",),
        )
        return [*components, locator]

    def _resolved_prompts(
        self,
        scope: str,
        system_parts: Sequence[SystemPromptPart],
        instruction_parts: Sequence[InstructionPart],
        *,
        span_id: str,
    ) -> None:
        if system_parts:
            source = {
                "static": [part.content for part in system_parts if part.dynamic_ref is None],
                "dynamic_refs": sorted(
                    part.dynamic_ref for part in system_parts if part.dynamic_ref is not None
                ),
            }
            definition = self._capture(
                scope=scope,
                kind="prompt",
                local_id="system_prompt",
                content=source,
                semantic_type=Semantic.PROMPT_VERSION,
                span_id=span_id,
                sensitivity=AssetSensitivity.SENSITIVE,
            )
            self._capture(
                scope=scope,
                kind="prompt",
                local_id="system_prompt:effective",
                content=[part.content for part in system_parts],
                semantic_type=Semantic.ASSET_RENDERING_VERSION,
                representation=AssetRepresentation.EFFECTIVE,
                definition_locator=definition,
                span_id=span_id,
                sensitivity=AssetSensitivity.SENSITIVE,
            )
        if instruction_parts:
            static = [part.content for part in instruction_parts if not part.dynamic]
            definition: str | None = None
            if static:
                definition = self._capture(
                    scope=scope,
                    kind="prompt",
                    local_id="instructions",
                    content=static,
                    semantic_type=Semantic.PROMPT_VERSION,
                    span_id=span_id,
                    sensitivity=AssetSensitivity.SENSITIVE,
                )
            self._capture(
                scope=scope,
                kind="prompt",
                local_id="instructions:effective",
                content=[part.content for part in instruction_parts],
                semantic_type=Semantic.ASSET_RENDERING_VERSION,
                representation=AssetRepresentation.EFFECTIVE,
                definition_locator=(
                    definition
                    if definition is not None
                    and all(not part.dynamic for part in instruction_parts)
                    else None
                ),
                span_id=span_id,
                sensitivity=AssetSensitivity.SENSITIVE,
            )

    def _effective_tool(self, scope: str, tool: ToolDefinition, *, span_id: str) -> str:
        return self._capture(
            scope=scope,
            kind="tool",
            local_id=f"{tool.name}:effective",
            name=tool.name,
            content=tool,
            semantic_type=Semantic.TOOL_VERSION,
            representation=AssetRepresentation.EFFECTIVE,
            definition_locator=_locator(scope, "tool", tool.name),
            span_id=span_id,
        )

    def _effective_output_tool(
        self,
        scope: str,
        tool: ToolDefinition,
        *,
        span_id: str,
    ) -> str:
        return self._capture(
            scope=scope,
            kind="output_schema",
            local_id=f"tool:{tool.name}:effective",
            name=tool.name,
            content=tool,
            semantic_type=Semantic.OUTPUT_SCHEMA_VERSION,
            representation=AssetRepresentation.EFFECTIVE,
            definition_locator=_locator(scope, "output_schema", "output"),
            span_id=span_id,
        )

    def _output_type(
        self,
        scope: str,
        local_id: str,
        output_type: Any,
        *,
        span_id: str,
    ) -> str:
        return self._capture(
            scope=scope,
            kind="output_schema",
            local_id=local_id,
            content=output_type,
            semantic_type=Semantic.OUTPUT_SCHEMA_VERSION,
            python_target=output_type if isinstance(output_type, type) else None,
            span_id=span_id,
        )

    def _capture(
        self,
        *,
        scope: str,
        kind: str,
        local_id: str,
        content: Any,
        semantic_type: str,
        span_id: str,
        name: str | None = None,
        python_target: Any = None,
        representation: AssetRepresentation = AssetRepresentation.DEFINITION,
        definition_locator: str | None = None,
        aliases: tuple[str, ...] = (),
        sensitivity: AssetSensitivity = AssetSensitivity.INTERNAL,
    ) -> str:
        locator = _locator(scope, kind, local_id)
        if not self.settings.allows(kind, representation):
            return locator
        self.runtime.asset(
            self.info,
            AssetCandidate(
                kind=kind,
                local_id=local_id,
                name=name or local_id,
                source_locator=locator,
                canonical_content=canonical_asset_content(content),
                representation=representation,
                semantic_type=semantic_type,
                scope=scope,
                python_target=python_target,
                definition_locator=definition_locator,
                provenance=AssetProvenance(
                    system="pydantic_ai",
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
        return locator


def _agent_scope(ctx: AgentRunContext[Any]) -> str:
    if ctx.agent is None:
        return "agent"
    name = ctx.agent.name
    if name is not None and name.strip():
        return f"agent:{name.strip()}"
    agent_type = type(ctx.agent)
    return f"agent:{agent_type.__module__}.{agent_type.__qualname__}"


def _capability_name(capability: AbstractCapability[Any]) -> str:
    if capability.id is not None and capability.id.strip():
        return capability.id.strip()
    serialization_name = type(capability).get_serialization_name()
    if serialization_name is not None and serialization_name.strip():
        return serialization_name.strip()
    capability_type = type(capability)
    return f"{capability_type.__module__}.{capability_type.__qualname__}"


def _locator(scope: str, kind: str, local_id: str) -> str:
    return f"pydantic_ai:{scope}:{kind}:{local_id}"


__all__ = ("AssetDiscovery",)
