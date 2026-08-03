from __future__ import annotations as _annotations

from pathlib import Path
from typing import Any

import httpx
from openai import OpenAI

from autobench import (
    Benchmark,
    Case,
    FactorValue,
    HTTPXInstrumentation,
    OpenAIInstrumentation,
    Variant,
    record_experiment,
)
from autobench.runtime.context import RunContext


def _response(request: httpx.Request) -> httpx.Response:
    assert request.url.path == "/chat/completions"
    content = (
        'data: {"id":"chatcmpl-demo","object":"chat.completion.chunk","created":0,'
        '"model":"demo-model","choices":[{"index":0,"delta":{"role":"assistant",'
        '"content":"hello"},"finish_reason":null}]}\n\n'
        'data: {"id":"chatcmpl-demo","object":"chat.completion.chunk","created":0,'
        '"model":"demo-model","choices":[{"index":0,"delta":{"content":" world"},'
        '"finish_reason":"stop"}]}\n\n'
        "data: [DONE]\n\n"
    )
    return httpx.Response(
        200,
        headers={"content-type": "text/event-stream"},
        content=content,
    )


_HTTP = httpx.Client(transport=httpx.MockTransport(_response), base_url="https://example.test")
_CLIENT = OpenAI(api_key="offline-example", base_url="https://example.test", http_client=_HTTP)


def run(ctx: RunContext, case: Case) -> dict[str, Any]:
    stream = _CLIENT.chat.completions.create(
        model=str(ctx.factor("model.name")),
        messages=[{"role": "user", "content": str(case.input)}],
        stream=True,
    )
    text = "".join(chunk.choices[0].delta.content or "" for chunk in stream)
    stream.close()
    return {"text": text}


def main() -> None:
    result = (
        Benchmark("openai-streaming")
        .dataset([Case(id="greeting", input="Say hello")])
        .variants(
            [
                Variant(
                    id="mock",
                    factors=[FactorValue(name="model.name", value="demo-model")],
                )
            ]
        )
        .task("run_openai_streaming:run")
        .instrument(OpenAIInstrumentation(), HTTPXInstrumentation())
        .run()
    )
    output_dir = Path(".autobench/examples/openai-streaming") / result.experiment_id
    record_experiment(result, output_dir)
    print(output_dir)


if __name__ == "__main__":
    main()
