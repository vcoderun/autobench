# Pydantic-GEPA Instrumentation

This project contains four credential-free optimizer benchmarks and one optional live integration.
Every offline task uses the real pydantic-gepa runtime while Autobench records it without an
explicit observer or manual span.

| Spec | Runtime behavior |
| --- | --- |
| `standard.yaml` | standard GEPA backend and deterministic reflection proposer |
| `autobench.yaml` | Optimize Anything Omni Pipeline with BestOf and continuation |
| `multi_component.yaml` | prompt, tool schema, and output schema in one candidate |
| `resume.yaml` | staged Plan followed by a completed-checkpoint resume |

```bash
uv sync --extra dev
for name in standard autobench multi_component resume; do
  uv run autobench run "examples/pydantic_gepa/$name.yaml" \
    --record "/tmp/autobench-pydantic-gepa-$name"
  uv run autobench replay "/tmp/autobench-pydantic-gepa-$name"
  uv run autobench report "/tmp/autobench-pydantic-gepa-$name"
done
```

The records include ABP optimization/engine/evaluation/candidate spans, semantic score and
resource-specific budget observations, component versions, candidate lineage, selections,
checkpoints, and the replay-safe `autobench.pydantic_gepa/v1` extension.

The expected report contains three engine rows (`weak`, `strong`, and `continuation`), both BestOf
contenders, one final candidate, three evaluation calls, no partial spans, and no diagnostics.

## Optional Live Pydantic AI Layering

The live script nests real Pydantic AI, OpenAI-compatible OpenRouter, and HTTPX operations inside a
pydantic-gepa evaluation. It defaults to `openrouter:openai/gpt-5.6-luna` and is not part of CI.

```bash
export OPENROUTER_API_KEY=...
uv run python examples/pydantic_gepa/live_pydantic_ai.py \
  --record /tmp/autobench-pydantic-gepa-live
```

`Benchmark.instrument_all()` installs each compatible native instrumentor once. Optimizer budget
and candidate evidence remains owned by pydantic-gepa instrumentation; model usage and transport
evidence remains owned by Pydantic AI, OpenAI, and HTTPX instrumentation.
