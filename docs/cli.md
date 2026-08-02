# CLI

## Commands

### Validate

```bash
uv run autobench validate path/to/spec.yaml
```

### Run

```bash
uv run autobench run path/to/spec.yaml --record runs/example
```

### Replay

```bash
uv run autobench replay runs/example
```

### Report

```bash
uv run autobench report runs/example
```

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

## CLI Behavior

- `run` executes the benchmark matrix, optionally records it, and renders Rich summary tables.
- `replay`, `report`, `export`, and `compare` operate on recorded evidence.
- `report` and `compare` render Rich terminal views instead of dumping Markdown or YAML.
- `export` always writes a file and then shows a Rich preview of the exported projection.
- default recording paths are placed under `.autobench/<spec-name>/<experiment-id>/`.
