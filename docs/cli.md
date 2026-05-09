# CLI

## Commands

### Validate

```bash
uv run autobench validate path/to/spec.yaml
```

### Run

```bash
uv run autobench run path/to/spec.yaml --record runs/example
uv run autobench run path/to/spec.yaml --record runs/example --save-png runs/example/report.png
uv run autobench run path/to/spec.yaml --record runs/example --save-png-dir runs/example/figures
```

### Replay

```bash
uv run autobench replay runs/example
uv run autobench replay runs/example --save-png runs/example/report.png
uv run autobench replay runs/example --save-png-dir runs/example/figures
```

### Report

```bash
uv run autobench report runs/example
uv run autobench report runs/example --save-png runs/example/report.png
uv run autobench report runs/example --save-png-dir runs/example/figures
```

### Export

```bash
uv run autobench export runs/example --format yaml --path runs/example/report.yaml
uv run autobench export runs/example --format csv --path runs/example/runs.csv
uv run autobench export runs/example --format markdown --path runs/example/report.md
uv run autobench export runs/example --format png --path runs/example/report.png
uv run autobench export runs/example --format png-set --path runs/example/figures
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
- `export --format png` renders the configured report visuals with Matplotlib and saves a composite image.
- `export --format png-set` writes one PNG per configured/default visualization panel.
- default recording paths are placed under `.autobench/<spec-name>/<experiment-id>/`.
