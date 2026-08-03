from __future__ import annotations as _annotations

from typing import Annotated, Literal, TypeAlias

from pydantic import BaseModel, ConfigDict, Field, field_validator


class InstrumentationSettings(BaseModel):
    """Shared configuration for one native Autobench instrumentor."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    enabled: bool = True


InstrumentorName: TypeAlias = Literal[
    "pydantic_ai",
    "openai",
    "openai_agents",
    "httpx",
]


class AutoInstrumentation(InstrumentationSettings):
    """Discover and install every compatible built-in instrumentor."""

    kind: Literal["all"] = "all"
    exclude: tuple[InstrumentorName, ...] = ()
    strict: bool = False

    @field_validator("exclude")
    @classmethod
    def normalize_exclusions(
        cls,
        values: tuple[InstrumentorName, ...],
    ) -> tuple[InstrumentorName, ...]:
        return tuple(sorted(set(values)))


class PydanticAIInstrumentation(InstrumentationSettings):
    """Capture Pydantic AI agent, model, tool, and validation activity."""

    kind: Literal["pydantic_ai"] = "pydantic_ai"


class OpenAIInstrumentation(InstrumentationSettings):
    """Capture official OpenAI Python client calls and streams."""

    kind: Literal["openai"] = "openai"


class OpenAIAgentsInstrumentation(InstrumentationSettings):
    """Capture OpenAI Agents workflows through its native trace processor."""

    kind: Literal["openai_agents"] = "openai_agents"


class HTTPXCaptureSettings(BaseModel):
    """Privacy-first HTTP request and response capture settings."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    path: Literal["omit", "hash", "full"] = "hash"
    request_headers: tuple[str, ...] = ()
    response_headers: tuple[str, ...] = ()
    request_body: bool = False
    response_body: bool = False
    max_body_bytes: int = Field(default=65_536, ge=1)

    @field_validator("request_headers", "response_headers")
    @classmethod
    def normalize_headers(cls, values: tuple[str, ...]) -> tuple[str, ...]:
        normalized = tuple(dict.fromkeys(value.strip().lower() for value in values))
        if any(not value for value in normalized):
            raise ValueError("captured header names cannot be empty")
        return normalized


class HTTPXInstrumentation(InstrumentationSettings):
    """Capture HTTPX calls at the public transport boundary."""

    kind: Literal["httpx"] = "httpx"
    capture: HTTPXCaptureSettings = Field(default_factory=HTTPXCaptureSettings)


BuiltinInstrumentationConfig: TypeAlias = (
    PydanticAIInstrumentation
    | OpenAIInstrumentation
    | OpenAIAgentsInstrumentation
    | HTTPXInstrumentation
)

InstrumentationConfig: TypeAlias = Annotated[
    AutoInstrumentation
    | PydanticAIInstrumentation
    | OpenAIInstrumentation
    | OpenAIAgentsInstrumentation
    | HTTPXInstrumentation,
    Field(discriminator="kind"),
]


__all__ = (
    "AutoInstrumentation",
    "HTTPXCaptureSettings",
    "HTTPXInstrumentation",
    "InstrumentationConfig",
    "InstrumentationSettings",
    "InstrumentorName",
    "OpenAIAgentsInstrumentation",
    "OpenAIInstrumentation",
    "PydanticAIInstrumentation",
)
