# CLI

The CLI is human-first: validation, runs, replay, reports, and comparisons render Rich panels and
tables. YAML, CSV, and Markdown are explicit file exports rather than raw terminal dumps.

## Command Summary

| Command | Executes tasks? | Requires records? | Purpose |
| --- | --- | --- | --- |
| `validate` | No | No | Parse, validate, resolve sources, and show the planned matrix |
| `run` | Yes | No | Execute a benchmark, optionally persist it, and render results |
| `replay` | No | Yes | Reconstruct and display the recorded experiment |
| `report` | No | Yes | Render configured analysis views from records |
| `export` | No | Yes | Write YAML, CSV, or Markdown and preview it |
| `compare` | No | Yes | Compare two recorded variants without claiming causality |

References inside a benchmark spec resolve relative to the spec file.

## Commands

### Validate

```bash
uv run autobench validate path/to/spec.yaml
```

Validation loads external datasets and referenced configuration, checks duplicate IDs and task
requirements, resolves source files, and renders case, variant, and planned-run counts. It does not
execute the task target.

### Run

```bash
uv run autobench run path/to/spec.yaml --record runs/example
```

Options:

- `--concurrency INTEGER`: maximum active runs, default `1`, minimum `1`.
- `--record DIRECTORY`: explicit immutable record directory.
- `--no-record`: execute and report without persistence.

Without either recording flag, Autobench writes under
`.autobench/<spec-name>/<experiment-id>/`.

### Replay

```bash
uv run autobench replay runs/example
```

Replay does not import tasks, scorers, or application modules. It reconstructs the experiment from
the record directory and renders the recorded report configuration.

### Report

```bash
uv run autobench report runs/example
```

`report` emphasizes status, variant configuration, leaderboards, per-run metrics, case matrices,
comparisons, and distributions.

### Export

```bash
uv run autobench export runs/example --format yaml --path runs/example/report.yaml
uv run autobench export runs/example --format csv --path runs/example/runs.csv
uv run autobench export runs/example --format markdown --path runs/example/report.md
```

### Compare

```bash
uv run autobench compare runs/example --baseline baseline --candidate optimized
```

Both IDs must exist in the recorded experiment. The command shows paired-run count, changed
factors, aggregate metric deltas, and a confounding flag.

## CLI Behavior

- `run` executes the benchmark matrix, optionally records it, and renders Rich summary tables.
- `replay`, `report`, `export`, and `compare` operate on recorded evidence.
- `report` and `compare` render Rich terminal views instead of dumping Markdown or YAML.
- `export` always writes a file and then shows a Rich preview of the exported projection.
- default recording paths are placed under `.autobench/<spec-name>/<experiment-id>/`.

## Exit And Error Behavior

- Invalid YAML, schema errors, unresolved tasks, recording collisions, and missing records return a
  nonzero exit code.
- User-facing errors include the relevant file and YAML location when available.
- A process may complete while individual runs are failed, errored, or skipped; the status table
  makes those states explicit.
- Replay and reporting never fall back to live benchmark execution.

## Typical Workflow

```bash
autobench validate autobench.yaml
autobench run autobench.yaml --concurrency 4 --record runs/candidate-42
autobench report runs/candidate-42
autobench compare runs/candidate-42 --baseline baseline --candidate candidate
autobench export runs/candidate-42 --format csv --path analysis/runs.csv
```
