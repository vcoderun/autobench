# Autobench

When an application changes, you need to know whether the new version is more correct, faster,
cheaper, or more reliable. Teams usually answer that question with a custom benchmark script: a
small program that runs representative inputs, checks the outputs, collects measurements, and
prints a comparison. As the application grows, these scripts are repeatedly rewritten and their
results become difficult to reproduce or compare.

We built Autobench to solve that problem. Define the cases, variants, task, scores, and reports
once in YAML or Python. Autobench runs the experiment, collects the evidence, records exactly what
happened, and lets you replay, report, compare, or export the result without rebuilding that
infrastructure for every application.

It is a YAML-first Python framework for AI and non-AI systems:

- deterministic dataset x variant execution
- sync and async application tasks
- semantic observations, checks, measurements, artifacts, and ABP traces
- built-in and custom scoring, cost derivation, policies, and paired baselines
- native Pydantic AI, OpenAI, OpenAI Agents, and HTTPX instrumentation
- explicit and automatic prompt/tool/schema/agent asset lineage
- immutable YAML records, replay, Rich reports, comparisons, and exports
- optional immutable-record export to OTLP-compatible telemetry backends

## Install

```bash
uv add autobench
```

For native SDK instrumentation:

```bash
uv add 'autobench[instrumentation]'
```

For outbound OTLP HTTP/protobuf export:

```bash
uv add 'autobench[otlp]'
```

## First Run

```bash
autobench validate examples/minimal/autobench.yaml
autobench run examples/minimal/autobench.yaml --record /tmp/autobench-minimal
autobench replay /tmp/autobench-minimal
autobench report /tmp/autobench-minimal
```

A task is a normal sync or async callable:

```python
from autobench import Case, RunContext


def run(ctx: RunContext, case: Case) -> Result:
    mode = ctx.factor("mode")
    with ctx.span("subject", kind="workflow") as span:
        result = application(case.input, mode=mode)
        span.set_output(result)
        return result
```

The YAML spec owns reusable benchmark infrastructure: cases, variants, scoring, derivation,
policies, instrumentation, and reports.

## Examples

| Directory | Demonstrates |
| --- | --- |
| `examples/minimal` | Inline cases, variants, exact scoring, report and comparison |
| `examples/basic` | File dataset, spans, checks, artifacts and failure visibility |
| `examples/mid` | Semantic usage, pricing, cost, policies and distributions |
| `examples/advanced` | Repeated measurement and paired-baseline speedup |
| `examples/pydantic_ai` | Live layered instrumentation and automatic asset discovery |
| `examples/automatic_assets` | Offline Pydantic AI and custom SDK behavioral lineage |
| `examples/abp_*` | Manual, concurrent, streaming, Agents and replay protocol flows |
| `examples/otlp_export` | Offline immutable ABP record to OTLP mapping |
| `examples/codemode` | Migration of a real external benchmark runner |

Run the offline release matrix:

```bash
make examples
```

## Documentation

Full documentation: [vcoderun.github.io/autobench](https://vcoderun.github.io/autobench/)

- [Installation](https://vcoderun.github.io/autobench/installation/)
- [First Benchmark](https://vcoderun.github.io/autobench/getting-started/)
- [Use Cases](https://vcoderun.github.io/autobench/use-cases/)
- [Architecture](https://vcoderun.github.io/autobench/architecture/)
- [YAML Spec](https://vcoderun.github.io/autobench/yaml-spec/)
- [Python API](https://vcoderun.github.io/autobench/python-api/)
- [Autobench Protocol](https://vcoderun.github.io/autobench/instrumentation-and-traces/)
- [OTLP Export](https://vcoderun.github.io/autobench/otlp-export/)

LLM-readable indexes are available at
[`llms.txt`](https://vcoderun.github.io/autobench/llms.txt) and
[`llms-full.txt`](https://vcoderun.github.io/autobench/llms-full.txt).

## Development

```bash
uv sync --extra dev --extra instrumentation --extra openai-agents --extra otlp
make prod
make pre-commit
```

The release gate enforces Python 3.11-3.14, strict lint and typing, strict documentation builds,
offline examples, and 100% source line and branch coverage.
