# Pydantic AI Native Instrumentation

This example benchmarks a real Pydantic AI agent without manual `ctx.span()` or
`ctx.metric()` calls. The agent uses a tool, retries a transient tool failure,
streams a structured result, and returns a tracked Pydantic output type.

For a real OpenRouter request with automatic Pydantic AI, OpenAI client, and HTTPX
instrumentation:

```bash
uv sync --extra instrumentation
export OPENROUTER_API_KEY=...
export OPENROUTER_MODEL=openai/gpt-4.1-mini
uv run python examples/pydantic_ai/openrouter_instrument_all.py \
  --record /tmp/autobench-openrouter
```

The resulting ABP trace contains agent, model, tool validation, tool execution,
output validation, streaming, OpenAI client, HTTPX transport, usage, scoring, and
tracked-asset evidence. The example uses `Benchmark.instrument_all()` and contains
no task-level `ctx.span()` or `ctx.metric()` calls.

`agent_benchmark.py` is the provider-neutral variant. Provider cost is intentionally
not calculated by an instrumentor; use Autobench pricing derivation after token
evidence has been recorded.
