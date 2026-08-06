# Troubleshooting

Start with the narrowest command that can identify the failing layer.

## Task Module Cannot Be Imported

```text
Could not import task module 'benchmarks.tasks'
```

Task targets must use `module:function`, not a file path:

```yaml
run:
  python: benchmark_task:run
```

Autobench first uses normal Python imports, then searches relative to the benchmark spec. Common
fixes:

- place `benchmark_task.py` next to `autobench.yaml` and use `benchmark_task:run`;
- for a package, ensure package directories have the expected Python import structure;
- do not include `.py` in the target;
- run `autobench validate path/to/autobench.yaml` from any directory to test resolution.

The function must accept `(ctx, case)` in that order.

## YAML Validates In One Editor But Not In Autobench

The schema directive improves editor completion; Autobench's installed Pydantic models remain the
runtime authority. Match the schema version to the installed package:

```bash
python -c "import autobench; print(autobench.__version__)"
```

```yaml
# yaml-language-server: $schema=./schemas/0.3.0/benchmark_schema.json
```

Run `autobench validate` and use its file/line diagnostics. Unknown scorer, policy,
instrumentation, and capture fields are rejected intentionally.

## Dataset File Is Not Found

`file://` references and glob patterns resolve relative to the benchmark YAML, not the shell's
current directory:

```yaml
dataset:
  source: file://datasets/cases.yaml
```

For a glob:

```yaml
dataset:
  source: file://datasets/cases/*.yaml
```

An unmatched glob is an error. Dataset files can contain a dataset DSL document, a case list, or a
single case mapping.

## Record Directory Already Exists

Autobench records are immutable. `record_experiment()` and `autobench run --record` refuse to
overwrite an existing experiment:

```text
Experiment record already exists
```

Use a new directory or remove/archive the old directory explicitly outside Autobench. Do not merge
unrelated experiments by copying run files together.

## A Metric Is Missing From Reports

Reports query semantic types, not only local names. Check:

1. the task/scorer/deriver emitted the observation;
2. `semantic_type` matches the report metric;
3. the value is numeric or boolean for the selected aggregate;
4. the selected span/query is not filtering it out;
5. source precedence did not intentionally select a score or derived observation instead.

Inspect the per-run YAML or use Python:

```python
from autobench import ObservationQuery

query = ObservationQuery(observations=run.task_result.observations)
matches = query.exact("money.cost")
```

Missing cost is not converted to zero. Ensure token, model, provider, and pricing inputs are all
available to the token-cost deriver.

## Paired Baseline Does Not Produce A Value

The baseline and candidate must match on the configured key, normally `case_id`, and both must have
the source metric. Check:

- `baseline_variant` exactly matches a variant ID;
- both runs emit the same semantic metric;
- the metric unit is compatible;
- `match_on` identifies a unique counterpart;
- the configured missing policy is appropriate.

Use a case matrix for the source metric before debugging the formula.

## `instrument_all()` Records Skipped Integrations

This is normal when optional SDKs are not installed. Automatic discovery records
`instrumentation.skipped` diagnostics and continues by default.

```bash
autobench instrumentation doctor
```

Install the relevant extra, remove the integration from `exclude`, or use `strict=True` when absence
must fail the benchmark.

Explicit `enabled: false` wins over automatic discovery. A custom runtime instrumentor with the
same ID also prevents a duplicate built-in installation.

## No Automatic Assets Appear

Automatic discovery only observes values that cross a supported instrumented SDK boundary while a
benchmark run is active. Check:

- the corresponding instrumentor is compatible and installed;
- asset discovery is enabled;
- `include` contains the expected family;
- `representations` includes `definition` or `effective` as needed;
- the SDK call occurs inside the task;
- capture policy does not reduce the asset below the expected content level.

Run the offline `examples/automatic_assets/` programs to separate environment issues from
application behavior.

## Duplicate Or Conflicting Instrumentation

Autobench prevents unsafe double patching. Do not install two instrumentors with the same ID or
instrument the same owner/method with incompatible specs. Prefer one of:

- automatic discovery only;
- explicit typed settings only;
- a custom runtime instrumentor that owns the same ID.

`InstrumentationConflictError` and patch diagnostics identify the owner and method involved.

## Trace Is Partial

A partial trace can be valid evidence. It may result from cancellation, an interrupted stream, a
task exception, or unmatched start/end signals. Inspect:

```bash
autobench instrumentation trace runs/example
```

ABP materialization keeps completed spans and diagnostics instead of dropping the trace. Accounting
extractors avoid double counting aggregate and leaf usage even when evidence is incomplete.

## Captured Content Is Redacted Or Hashed

Capture is privacy-first. A value's retained representation is controlled by the benchmark
`CapturePolicy`, semantic defaults, path rules, and SDK-specific HTTP settings.

Use `CapturePolicy.full()` only for controlled local evidence. Prefer targeted semantic overrides:

```python
from autobench import CaptureLevel, CapturePolicy

policy = CapturePolicy.hashed(
    semantic_overrides={"output_schema": CaptureLevel.FULL},
)
```

Secret names, denied paths, truncation limits, and binary rules still apply.

## Replay Needs An Optional SDK

It should not. `replay`, `report`, `compare`, `export`, and `instrumentation trace` are designed to
load records without benchmark or provider imports. If replay fails, verify that:

- `experiment.yaml` and every path in `runs.paths` exist;
- trace artifact paths remain inside the experiment directory;
- referenced artifacts were copied with the records;
- the record version is supported.

Do not solve a missing artifact by re-executing the benchmark implicitly.

## Python Type Errors In Tasks

Autobench keeps case input and factor values generic because applications define their schemas.
Validate at the task boundary:

```python
from pydantic import BaseModel, TypeAdapter

request = Request.model_validate(case.input)
mode = TypeAdapter(Mode).validate_python(ctx.factor("mode"))
```

This gives application-specific errors without weakening Autobench's public types.

## Get A Reproducible Diagnostic Bundle

For a bug report, include:

```bash
autobench --help
autobench instrumentation doctor
autobench validate path/to/autobench.yaml
```

Also include the package version, Python version, failing record directory when it contains no
sensitive data, and the smallest benchmark/task that reproduces the problem. Review capture policy
before sharing records.
