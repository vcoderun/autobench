# CLI And Operations

## Contents

- [Command map](#command-map)
- [Standard workflow](#standard-workflow)
- [Exit behavior](#exit-behavior)
- [Schemas and YAML](#schemas-and-yaml)
- [Troubleshooting](#troubleshooting)
- [Development gates](#development-gates)
- [Release hygiene](#release-hygiene)

## Command Map

| Command | Executes subject? | Input | Purpose |
| --- | --- | --- | --- |
| `validate` | no | benchmark YAML | Resolve and validate the matrix |
| `dataset generate` | yes | generator target + request YAML | Prepare a frozen dataset before matrix planning |
| `run` | yes | benchmark YAML | Execute, record, and render |
| `replay` | no | record directory | Reconstruct recorded results |
| `report` | no | record directory | Render configured analysis |
| `compare` | no | record directory | Compare two variants |
| `export` | no | record directory | Write YAML, CSV, or Markdown projection |
| `instrumentation doctor` | no | environment | Inspect integration compatibility |
| `instrumentation trace` | no | record directory | Summarize ABP evidence |
| `telemetry export` | no | record directory | Replay immutable ABP evidence to OTLP |

Use `autobench <command> --help` for exact flags in the installed version.

## Standard Workflow

```bash
autobench dataset generate generation:generate_cases \
  --request generation-request.yaml \
  --output datasets/generated.yaml \
  --id generated-v1
autobench validate autobench.yaml
autobench run autobench.yaml --record runs/latest
autobench replay runs/latest
autobench report runs/latest
autobench compare runs/latest --baseline current --candidate proposed
autobench export runs/latest --format csv --path analysis/runs.csv
autobench telemetry export runs/latest --endpoint https://collector.example/v1/traces
```

For cross-invocation analysis, `run` accepts `--group-id`, `--attempt`, `--phase`,
`--parent-experiment-id`, `--resumed-from-experiment-id`, and repeatable
`--correlation-label KEY VALUE`. Explicit flags override YAML `execution.correlation` one field at a
time; omitted values remain unchanged.

Validate before any expensive subject call. Keep record directories outside committed source unless
the project intentionally checks in fixtures.

The CLI renders Rich tables and panels for humans. Machine consumers should read records or exports,
not scrape terminal formatting.

`dataset generate` resolves a sync or async `CaseGenerator`, records request/provider/model/review/
usage/cost provenance, and publishes normal dataset YAML plus a generation manifest. Explicit
incomplete generation exits `2` and writes only an incomplete sidecar; it never mutates a benchmark
dataset.

## Exit Behavior

- malformed YAML, spec validation, missing task target, and record corruption are non-zero errors;
- a completed experiment may contain failed or errored runs and still produce a record/report;
- strict instrumentation incompatibility fails before subject execution;
- optional/skipped instrumentation is diagnostic under non-strict discovery;
- replay/report/export never hide a missing artifact by rerunning the subject.

Read the Rich error panel and chained cause. Autobench errors should name the failing module, path,
case, variant, scorer, or integration without dumping secrets.

## Schemas And YAML

The repository ships versioned schemas for benchmark, dataset, generation request/manifest,
records, semantics, pricing, reports, artifacts, assets, and traces. All generated Autobench YAML should include a
`yaml-language-server` schema header.

When changing YAML behavior:

1. change the typed model;
2. update the authoring/serialization transform;
3. regenerate or edit the correct versioned schema;
4. add round-trip and invalid-shape tests;
5. update examples, user docs, and the skill reference;
6. ensure older record fixtures still load when compatibility requires it.

## Troubleshooting

### Task import failure

- Confirm the spec target is `module:callable`.
- Confirm the module file is next to the spec or importable from the project environment.
- Run from the project environment with Autobench installed.
- Avoid relying on an `examples` package that is not installed.
- Do not fix the issue with permanent global `sys.path` mutation.

### Missing or zero cost

- Confirm input/output token observations have canonical semantic types.
- Confirm provider/model factor semantics exist.
- Confirm model normalization resolves to a pricing entry.
- Confirm pricing units and tiers match the token count.
- Treat missing prices as diagnostics; do not silently report valid-looking zero cost.

### Missing trace or metrics

- Run `autobench instrumentation doctor`.
- Confirm the optional extra and supported SDK version.
- Inspect `instrumentation.skipped` diagnostics.
- Inspect `autobench instrumentation trace <record>`.
- Check capture policy and semantic extraction before adding manual metrics.

### Missing historical asset content

- Check `assets/*.yaml` for the content reference.
- Resolve `artifacts/asset-content.sqlite3` with `load_asset_content()`.
- Confirm `asset_default_level` was not hash, metadata, redacted, or none.
- A hash-only historical asset cannot be reconstructed after the process exits.

### Replay failure

- Keep `experiment.yaml`, run YAML, assets, and artifacts together.
- Confirm record paths remain inside the record root.
- Confirm the record version is supported.
- Never rerun automatically as a replay fallback.

## Development Gates

For the Autobench repository:

```bash
uv sync --extra dev --extra instrumentation --extra openai-agents --extra otlp
make prod
make pre-commit
```

`make prod` includes:

- full pytest suite;
- 100% targeted source line and branch coverage;
- Ruff format and lint;
- ty and basedpyright;
- strict Zensical docs and LLM bundle freshness;
- Python 3.11, 3.12, 3.13, and 3.14 matrix;
- offline examples.

`make release` additionally exercises pre-commit, package build, and built-wheel/no-extras smoke
behavior. Run targeted tests during iteration, then the complete gates before finishing.

For downstream benchmark projects, at minimum validate the spec and run/replay/report one real
record. Add project tests for task behavior and custom scorers.

## Release Hygiene

- Keep package version, changelog, schemas, docs, examples, and optional dependency ranges aligned.
- Exclude plans, references, local runs, caches, agent instructions, and generated site output from
  distributions unless they are intentional package resources.
- Build wheel and sdist, inspect contents, and smoke-test the wheel without optional SDKs.
- Do not claim production readiness only because unit tests pass; verify install, CLI, docs,
  records, replay, and compatibility surfaces.
- Commit generated `llms-full.txt` when the repository treats it as a release artifact.
