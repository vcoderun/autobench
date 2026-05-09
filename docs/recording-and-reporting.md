# Recording And Reporting

## Recording

`autobench run --record ...` writes immutable experiment evidence.

The important properties are:

- append-only recording
- stable run paths
- artifact payload materialization
- spec and source file hashing
- replay without task imports

## Replay

Replay reconstructs `ExperimentResult` from YAML records and recorded metadata. This makes reports and comparisons safe to regenerate later.

## Reports

Built-in report views:

- leaderboard
- case matrix
- baseline/candidate comparisons
- metric distributions

Leaderboards and comparisons are semantic-type driven, not hard-wired to one app domain.

## Exports

Built-in exports:

- Markdown
- YAML summary
- CSV run table
- PNG visual report

These are generated from replayed evidence, not from re-running the benchmark.
The CLI writes them to explicit file paths and renders Rich terminal previews
instead of printing raw Markdown, YAML, or CSV to stdout.

## Visual Reports

`reports.visuals` lets a YAML spec describe Matplotlib-backed report images.

```yaml
reports:
  visuals:
    - kind: variant_config
      render_as: table
    - kind: status
      render_as: pie
    - kind: leaderboard
      render_as: bar
      metric: avg_quality
    - kind: leaderboard
      render_as: grouped_bar
    - kind: case_matrix
      render_as: heatmap
    - kind: case_matrix
      render_as: grouped_bar
    - kind: comparison
      baseline: baseline
      candidate: candidate
      render_as: delta_bar
    - kind: distribution
      name: latency_distribution
      render_as: boxplot
```

Then export it with:

```bash
uv run autobench export runs/example --format png --path runs/example/report.png
uv run autobench export runs/example --format png-set --path runs/example/figures
```
