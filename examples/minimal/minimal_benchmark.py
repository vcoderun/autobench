from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, TypeAdapter

from autobench import Case, RunContext

Transform = Literal["upper", "title_upper"]
TRANSFORM_ADAPTER = TypeAdapter(Transform)


class TextInput(BaseModel):
    text: str


class TextOutput(BaseModel):
    text: str


def run(ctx: RunContext, case: Case) -> TextOutput:
    sample = TextInput.model_validate(case.input)
    transform = TRANSFORM_ADAPTER.validate_python(ctx.factor("transform"))
    with ctx.span("transform_text", kind="workflow", input=sample.model_dump()) as span:
        text = sample.text.upper() if transform == "upper" else sample.text.title().upper()
        output = TextOutput(text=text)
        span.set_output(output.model_dump())
    return output


__all__ = ("run",)
