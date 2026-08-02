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

These are generated from replayed evidence, not from re-running the benchmark.
The CLI writes them to explicit file paths and renders Rich terminal previews
instead of printing raw Markdown, YAML, or CSV to stdout.
