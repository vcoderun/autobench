from __future__ import annotations

from dataclasses import fields, is_dataclass
from typing import Any

from agents import Agent, FunctionTool, Handoff, InputGuardrail, OutputGuardrail

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
    """Discover OpenAI Agents definitions without evaluating dynamic callbacks."""

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

    def agent(self, agent: Agent[Any], *, span_id: str | None = None) -> None:
        self._agent(agent, span_id=span_id, visited=set())

    def _agent(
        self,
        agent: Agent[Any],
        *,
        span_id: str | None,
        visited: set[int],
    ) -> None:
        if id(agent) in visited:
            return
        visited.add(id(agent))
        scope = f"agent:{agent.name}"
        children: list[str] = []
        if agent.instructions is not None:
            children.append(
                self._capture(
                    scope=scope,
                    kind="prompt",
                    local_id="instructions",
                    content=agent.instructions,
                    semantic_type=Semantic.PROMPT_VERSION,
                    python_target=agent.instructions if callable(agent.instructions) else None,
                    span_id=span_id,
                    sensitivity=AssetSensitivity.SENSITIVE,
                )
            )
        if agent.prompt is not None:
            children.append(
                self._capture(
                    scope=scope,
                    kind="prompt",
                    local_id="prompt",
                    content=agent.prompt,
                    semantic_type=Semantic.PROMPT_VERSION,
                    python_target=agent.prompt if callable(agent.prompt) else None,
                    span_id=span_id,
                    sensitivity=AssetSensitivity.SENSITIVE,
                )
            )
        for tool in agent.tools:
            children.append(self._tool(scope, tool, span_id=span_id))
        for server in agent.mcp_servers:
            name = type(server).__name__
            children.append(
                self._capture(
                    scope=scope,
                    kind="toolset",
                    local_id=f"mcp:{name}",
                    name=name,
                    content=_public_dataclass(server),
                    semantic_type=Semantic.TOOLSET_VERSION,
                    span_id=span_id,
                )
            )
        if agent.output_type is not None:
            children.append(
                self._capture(
                    scope=scope,
                    kind="output_schema",
                    local_id="output",
                    content=agent.output_type,
                    semantic_type=Semantic.OUTPUT_SCHEMA_VERSION,
                    python_target=agent.output_type
                    if isinstance(agent.output_type, type)
                    else None,
                    span_id=span_id,
                )
            )
        for guardrail in agent.input_guardrails:
            children.append(self._guardrail(scope, "input", guardrail, span_id=span_id))
        for guardrail in agent.output_guardrails:
            children.append(self._guardrail(scope, "output", guardrail, span_id=span_id))
        for handoff in agent.handoffs:
            if isinstance(handoff, Agent):
                self._agent(handoff, span_id=span_id, visited=visited)
                content: Any = {
                    "agent": handoff.name,
                    "description": handoff.handoff_description,
                }
                local_id = handoff.name
            else:
                content = _handoff_content(handoff)
                local_id = handoff.tool_name
            children.append(
                self._capture(
                    scope=scope,
                    kind="handoff",
                    local_id=local_id,
                    content=content,
                    semantic_type=Semantic.HANDOFF_VERSION,
                    span_id=span_id,
                )
            )
        policy = agent.tool_use_behavior
        children.append(
            self._capture(
                scope=scope,
                kind="policy",
                local_id="tool_use",
                content=policy,
                semantic_type=Semantic.POLICY_VERSION,
                python_target=policy if callable(policy) else None,
                span_id=span_id,
            )
        )
        self._capture(
            scope=scope,
            kind="agent",
            local_id="self",
            name=agent.name,
            content={
                "name": agent.name,
                "handoff_description": agent.handoff_description,
                "children": sorted(set(children)),
                "reset_tool_choice": agent.reset_tool_choice,
            },
            semantic_type=Semantic.AGENT_VERSION,
            span_id=span_id,
        )

    def _tool(self, scope: str, tool: Any, *, span_id: str | None) -> str:
        if isinstance(tool, FunctionTool):
            name = tool.qualified_name
            content: Any = {
                "name": tool.name,
                "qualified_name": tool.qualified_name,
                "description": tool.description,
                "parameters": tool.params_json_schema,
                "output": tool.output_json_schema,
                "strict": tool.strict_json_schema,
                "approval": canonical_asset_content(tool.needs_approval),
                "enabled": canonical_asset_content(tool.is_enabled),
                "timeout_seconds": tool.timeout_seconds,
                "timeout_behavior": tool.timeout_behavior,
                "defer_loading": tool.defer_loading,
                "input_guardrails": canonical_asset_content(tool.tool_input_guardrails),
                "output_guardrails": canonical_asset_content(tool.tool_output_guardrails),
            }
        else:
            name = _tool_name(tool)
            content = _public_dataclass(tool)
        return self._capture(
            scope=scope,
            kind="tool",
            local_id=name,
            name=name,
            content=content,
            semantic_type=Semantic.TOOL_VERSION,
            span_id=span_id,
        )

    def _guardrail(
        self,
        scope: str,
        direction: str,
        guardrail: InputGuardrail[Any] | OutputGuardrail[Any],
        *,
        span_id: str | None,
    ) -> str:
        name = guardrail.get_name()
        return self._capture(
            scope=scope,
            kind="guardrail",
            local_id=f"{direction}:{name}",
            name=name,
            content={
                "direction": direction,
                "function": canonical_asset_content(guardrail.guardrail_function),
                "run_in_parallel": (
                    guardrail.run_in_parallel if isinstance(guardrail, InputGuardrail) else None
                ),
            },
            semantic_type=Semantic.GUARDRAIL_VERSION,
            python_target=guardrail.guardrail_function,
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
        span_id: str | None,
        name: str | None = None,
        python_target: Any = None,
        sensitivity: AssetSensitivity = AssetSensitivity.INTERNAL,
    ) -> str:
        locator = f"openai_agents:{scope}:{kind}:{local_id}"
        if not self.settings.allows(kind, AssetRepresentation.DEFINITION):
            return locator
        self.runtime.asset(
            self.info,
            AssetCandidate(
                kind=kind,
                local_id=local_id,
                name=name or local_id,
                source_locator=locator,
                canonical_content=canonical_asset_content(content),
                semantic_type=semantic_type,
                scope=scope,
                python_target=python_target,
                provenance=AssetProvenance(
                    system="openai_agents",
                    key=local_id,
                    instrumentor=self.info.id,
                    instrumented_library_version=self.target_version,
                ),
                sensitivity=sensitivity,
            ),
            span_id=span_id,
            registry=self.registry,
        )
        return locator


def _public_dataclass(value: Any) -> Any:
    if not is_dataclass(value) or isinstance(value, type):
        return canonical_asset_content(value)
    return {
        field.name: canonical_asset_content(getattr(value, field.name))
        for field in fields(value)
        if not field.name.startswith("_")
    }


def _tool_name(tool: Any) -> str:
    try:
        name: Any = tool.name
    except AttributeError:
        name = None
    if isinstance(name, str) and name:
        return name
    try:
        tool_type: Any = tool.type
    except AttributeError:
        return type(tool).__name__
    if isinstance(tool_type, str) and tool_type:
        return tool_type
    return type(tool).__name__


def _handoff_content(handoff: Handoff[Any, Any]) -> dict[str, Any]:
    return {
        "tool_name": handoff.tool_name,
        "tool_description": handoff.tool_description,
        "input_schema": handoff.input_json_schema,
        "agent_name": handoff.agent_name,
        "input_filter": canonical_asset_content(handoff.input_filter),
        "nest_history": handoff.nest_handoff_history,
        "strict": handoff.strict_json_schema,
        "enabled": canonical_asset_content(handoff.is_enabled),
    }


__all__ = ("AssetDiscovery",)
