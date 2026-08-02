# CodeMode Benchmark

This example is the real Autobench migration boundary for the reference CodeMode benchmark. The
benchmark spec, dataset, variants, semantic metrics, artifacts, comparisons, recording, and replay
belong to Autobench. The task remains an integration with `vowel.codemode` and requires that runtime,
an `OPENROUTER_API_KEY`, and network access.

```bash
uv run python examples/codemode/run_benchmark.py --only parse_cron
uv run autobench report examples/codemode/.autobench/<experiment-id>
```

The four offline examples under `examples/minimal`, `examples/basic`, `examples/mid`, and
`examples/advanced` are the deterministic CI suite. This CodeMode integration is an explicit live
dogfood example and is not replaced by a mock runtime.
