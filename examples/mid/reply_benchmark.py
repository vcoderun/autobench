from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, TypeAdapter

from autobench import Case, Direction, ObservationRole, RunContext, Semantic

ResponseStyle = Literal["terse", "helpful"]
STYLE_ADAPTER = TypeAdapter(ResponseStyle)


class ReplyInput(BaseModel):
    question: str
    required_phrase: str


class ReplyOutput(BaseModel):
    text: str
    acceptable: bool
    success: bool = True


def run(ctx: RunContext, case: Case) -> ReplyOutput:
    sample = ReplyInput.model_validate(case.input)
    style = STYLE_ADAPTER.validate_python(ctx.factor("response_style"))
    with ctx.span(
        "generate_reply",
        kind="llm",
        input={"question": sample.question},
        duration_metric={
            "name": "request_latency",
            "semantic_type": Semantic.TIME_LATENCY,
            "unit": "s",
            "direction": Direction.MINIMIZE,
        },
    ) as span:
        text = _reply(sample, style=style)
        output = ReplyOutput(
            text=text,
            acceptable=sample.required_phrase.casefold() in text.casefold(),
        )
        input_tokens = len(sample.question.split()) + 12
        output_tokens = len(text.split())
        span.metric(
            "input_tokens",
            input_tokens,
            semantic_type=Semantic.LLM_TOKENS_INPUT,
            unit="tokens",
            role=ObservationRole.DIAGNOSTIC,
        )
        span.metric(
            "output_tokens",
            output_tokens,
            semantic_type=Semantic.LLM_TOKENS_OUTPUT,
            unit="tokens",
            role=ObservationRole.DIAGNOSTIC,
        )
        span.set_usage("input_tokens", input_tokens)
        span.set_usage("output_tokens", output_tokens)
        span.set_output(output.model_dump())
        span.artifact("reply", text, media_type="text/plain")
    return output


def _reply(sample: ReplyInput, *, style: ResponseStyle) -> str:
    if style == "terse":
        return "Please contact support."
    return f"Open the {sample.required_phrase} and follow the available instructions."


__all__ = ("run",)
