# Pydantic-GEPA Instrumentation

This credential-free scaffold runs a real pydantic-gepa Optimize Anything pipeline. Two custom
engines explore candidates through the real evaluation server, the best candidate continues into a
second engine, and Autobench records the optimization without an explicit observer or manual span.
The repository's canonical `examples/pydantic_gepa` directory also covers standard GEPA,
multi-component candidates, checkpoint resume, and optional live Pydantic AI layering.

```bash
uv sync --extra dev
uv run autobench run examples/pydantic_gepa/autobench.yaml \
  --record /tmp/autobench-pydantic-gepa
uv run autobench replay /tmp/autobench-pydantic-gepa
uv run autobench report /tmp/autobench-pydantic-gepa
```

The record includes ABP optimization/engine/evaluation/candidate spans, semantic score and budget
observations, prompt component versions, candidate lineage, and the replay-safe
`autobench.pydantic_gepa/v1` extension.

The expected report contains three engine rows (`weak`, `strong`, and `continuation`), both BestOf
contenders, one final candidate, three evaluation calls, no partial spans, and no diagnostics.
