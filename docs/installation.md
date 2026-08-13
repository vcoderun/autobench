# Installation

Autobench supports Python 3.11 through 3.14. The base package includes the benchmark DSL, runtime,
semantic evidence models, evaluation, recording, replay, reports, CLI, and manual ABP spans.

## Base Package

=== "uv"

    ```bash
    uv add autobench
    ```

=== "pip"

    ```bash
    python -m pip install autobench
    ```

Verify the installation:

```bash
autobench --help
python -c "import autobench; print(autobench.__version__)"
```

## Optional SDK Integrations

Native ABP instrumentors are optional so a generic benchmark does not install AI SDKs.

```bash
uv add 'autobench[instrumentation]'
```

The instrumentation extra supplies the supported Pydantic AI, OpenAI Python, and HTTPX integration
environment. OpenAI Agents support has its own extra:

```bash
uv add 'autobench[openai-agents]'
```

Install only the native pydantic-gepa optimizer integration with:

```bash
uv add 'autobench[pydantic-gepa]'
```

It records optimizer lifecycle, evaluation evidence, budgets, candidate lineage, and component
asset versions. See [Pydantic-GEPA Instrumentation](pydantic-gepa-instrumentation.md).

Inspect what the current environment can instrument:

```bash
autobench instrumentation doctor
```

The command reports compatibility rather than failing because an optional SDK is absent.

To export immutable ABP records to an OTLP HTTP/protobuf backend, install the independent exporter
extra:

```bash
uv add 'autobench[otlp]'
```

Collection still uses ABP. The extra is needed only on the process that performs
`autobench telemetry export`; see [OTLP Export](otlp-export.md).

## Development Checkout

From the repository root:

```bash
uv sync --extra dev --extra instrumentation --extra openai-agents --extra otlp
make prod
```

Useful targets:

| Command | Purpose |
| --- | --- |
| `make tests` | Test suite with source line and branch coverage |
| `make check` | Ruff, ty, and basedpyright |
| `make docs` | LLM bundles and strict Zensical build |
| `make examples` | Offline end-to-end example matrix |
| `make prod` | Full supported-Python and release quality gates |
| `make pre-commit` | Repository-wide hooks |

## Editor Setup For YAML

Every exported Autobench YAML document starts with a `yaml-language-server` schema directive.
Versioned schemas are shipped under `schemas/<autobench-version>/` and installed to the user schema
directory when the schema helpers run.

For a repository-local benchmark:

```yaml
# yaml-language-server: $schema=./schemas/0.3.0/benchmark_schema.json
benchmark:
  smoke-test:
    cases: []
```

Use the schema matching the Autobench version that validates and executes the file. This provides
completion for scorer variants, policy operators, instrumentation settings, report configuration,
and semantic registry entries.

## Credential Handling

Autobench itself does not require model credentials. Live examples read provider configuration from
the relevant SDK environment. For example:

```bash
export OPENROUTER_API_KEY=...
export OPENROUTER_MODEL=openrouter:openai/gpt-5.6-luna
```

Do not place credentials in benchmark specs, cases, artifacts, or capture policies. Use the
[capture policy](automatic-asset-discovery.md#privacy-and-capture-policy) to prevent sensitive SDK
inputs from being retained. Runtime evidence defaults to metadata, but versioned behavioral assets
default to full content and are stored in `artifacts/asset-content.sqlite3`; set
`asset_default_level: hash` when that local registry must not retain prompt, tool, or schema bodies.

## Next Step

Continue with [First Benchmark](getting-started.md), which creates a task, dataset, variant matrix,
score, record, report, comparison, and export.
