# Markdown Reports

Autobench Markdown reports explain benchmark outcomes to the people deciding whether a system is
good enough to ship. They are designed for release reviews, pull requests, model or pipeline
comparisons, incident follow-up, and long-lived experiment archives. The default report starts with
the quality gate, score range, case-level failures, meaningful charts, and evaluator explanations.
Run IDs, hashes, traces, asset internals, and other engineering evidence remain available in the
explicit `audit` profile instead of overwhelming the normal report.

Markdown is not the source of truth. Immutable experiment and run records remain authoritative;
the report is a deterministic projection that can be regenerated without running the application.

## What The Report Contains

A full report can include:

- a plain-language executive summary and benchmark verdict;
- quality-gate pass rate, average, median, best, and lowest case score;
- deterministic inline charts for quality-gate composition, case ranking, and normalized dimensions;
- a case table that separates benchmark quality from execution success;
- issue totals such as omissions, leaks, failures, and policy violations;
- priority evaluator feedback for the lowest-scoring failed cases;
- variant configuration, direction-aware leaderboards, and paired comparisons;
- absolute and relative deltas, win/tie/loss counts, and explicit confounding warnings;
- configured matrices and distributions when they add decision value;
- a concise benchmark setup and optimization outcome.

The `audit` profile adds lifecycle status, metric coverage, run IDs, failures, ABP traces, asset
lineage, artifacts, hashes, provenance, and explicitly permitted captured content.

Autobench uses fixed deterministic finding rules. It does not call an LLM to write the summary,
claim statistical significance, or infer causality from a confounded comparison.

## Terminal Versus Document

The default command remains the interactive Rich report:

```bash
autobench report runs/support-routing
```

Generate a Markdown document explicitly:

```bash
autobench report runs/support-routing \
  --format markdown \
  --profile full \
  --layout single \
  --output analysis/support-routing.md
```

The CLI prints a Rich publication summary containing the selected layout, file count, byte count,
run count, section count, and notices. It never dumps the Markdown document into the terminal.

The compatibility export command uses the same single-file writer:

```bash
autobench export runs/support-routing \
  --format markdown \
  --path analysis/support-routing.md
```

Both commands replay stored evidence. They do not import the task module, contact a model provider,
or execute the benchmark subject.

## Profiles

| Profile | Purpose | Included detail |
| --- | --- | --- |
| `summary` | fast decision review | verdict, quality KPIs, charts, bounded case outcomes, comparisons, policies, and material limitations |
| `full` | stakeholder benchmark report | summary plus benchmark setup, configured analysis, and priority evaluator feedback |
| `audit` | engineering evidence inspection | full report plus lifecycle, metric coverage, runs, failures, traces, assets, artifacts, hashes, provenance, and permission-gated captured content |

`full` is the default. A run whose task executed successfully can still fail the benchmark quality
gate; reports present these as separate facts. `audit` does not automatically reveal captured content. The command must
also receive `--include-captured-content`, and the original capture/redaction policy must have
retained that content. Reporting can never recover data that was not recorded or weaken a sensitive
asset policy.

`limits.value_excerpt_chars` bounds each captured value or diff excerpt independently. Autobench
serializes JSON-compatible values deterministically and marks unsupported values unavailable rather
than calling application code. `assets.diffs` controls asset history independently:

- `none`: omit changed fields and diff detail;
- `summary`: show changed field paths and whether stored diff evidence exists;
- `full`: allow a bounded stored diff excerpt, but only in an explicitly content-enabled audit.

Sensitive assets remain omitted in every profile, including an audit with content permission.

```bash
autobench report runs/support-routing \
  --format markdown \
  --profile audit \
  --include-captured-content \
  --output analysis/support-routing-audit.md
```

## Quality Outcome Convention

Autobench does not equate a completed task call with a successful benchmark outcome. A task may
return a mapping or Pydantic model with a report-ready evaluation envelope:

```python
return {
    "hard_pass": result.meets_release_gate,
    "score": result.overall_score,
    "metrics": {
        "semantic_score": result.semantic_score,
        "critical_omissions": result.critical_omissions,
        "forbidden_leaks": result.forbidden_leaks,
    },
    "feedback": result.feedback,
}
```

The same fields may live under an `evaluation` key. `hard_pass`, `passed`, and `pass` are accepted
quality-gate names. `feedback` may be one string or a list of strings. Numeric score records remain
valid evidence and provide a fallback objective score when the output has no `score` field.

Normalized score-like metrics become quality dimensions. Boolean quality metrics are summarized as
rates. Counts are shown as quality issues only when their names describe failures, omissions, leaks,
violations, or similar problems; neutral measurements such as output length remain out of the issue
table. If no recognizable quality evidence exists, Autobench does not invent a verdict or chart.

## Layouts

### Single

`single` writes one portable `.md` file. Use it for small and medium experiments, pull requests,
and attachments.

### Bundle

`bundle` publishes a directory containing:

```text
benchmark-report/
  index.md
  cases/<stable-id>.md
  variants/<stable-id>.md
  runs/<stable-id>.md       # audit only
  assets/<stable-id>.md     # audit only
```

The index contains the decision summary and links to user-facing case and variant pages. Technical
run and asset pages are created only for `audit`. Page names use normalized identifiers plus a
stable hash suffix, so human-readable IDs do not silently collide.

```bash
autobench report runs/support-routing \
  --format markdown \
  --layout bundle \
  --output analysis/support-routing-report
```

### Auto

`auto` chooses from report size, never terminal width. For normal reports, case evidence determines
detail size. For audits, run, failure, asset-version, and artifact inventories are included. It
selects a bundle when any of these deterministic limits is exceeded:

- more than 50 runs;
- more than 100 relevant detail rows;
- more than 1,000 case-matrix cells.

The selected layout is returned in `MarkdownReportPublication` and shown by the CLI.

## YAML Configuration

Configure Markdown alongside leaderboard, matrix, comparison, and distribution views:

```yaml
report:
  leaderboard:
    show:
      quality:
        metric: quality.score
        aggregate: mean
      total_cost:
        metric: money.cost
        aggregate: sum
      p95_latency:
        metric: time.latency
        aggregate: p95
  compare:
    baseline -> candidate:
      show:
        quality:
          metric: quality.score
          aggregate: mean
        p95_latency:
          metric: time.latency
          aggregate: p95
  markdown:
    profile: full
    layout: single
    output: reports/benchmark.md
    limits:
      table_rows: 200
      run_details: 100
      failure_details: 100
      value_excerpt_chars: 2000
    traces:
      top_slowest: 20
    assets:
      diffs: summary
    content:
      include_captured: false
```

`output` must be a portable relative path without `..`. When `autobench run` records through the
CLI, a configured output is generated in the staging record before final metadata and manifest
hashes are sealed. A failed publication prevents the record from being presented as successfully
finalized.

Omit `output` when reports should only be generated post-hoc.

## Python API

Build, render, or publish the same typed report model:

```python
from pathlib import Path

from autobench import (
    build_report,
    load_experiment_record,
    replay_experiment,
    render_markdown_report,
    write_markdown_report,
)

record_dir = Path("runs/support-routing")
result = replay_experiment(record_dir)
record = load_experiment_record(record_dir)
report = build_report(
    result,
    experiment_record=record,
    experiment_root=record_dir,
)

markdown = render_markdown_report(report)
publication = write_markdown_report(
    report,
    Path("analysis/support-routing"),
    layout="bundle",
    immutable_root=record_dir,
)

print(publication.layout)
print([(file.path, file.byte_count, file.sha256) for file in publication.files])
```

`render_markdown_report()` returns text and performs no write. `write_markdown_report()` owns
atomic publication and returns `MarkdownReportPublication` with every output path, byte count, and
SHA-256 hash. Existing `export_markdown_report(result, path)` remains supported and delegates to the
same single-file writer.

When `immutable_root` is supplied, the writer rebases every validated artifact link relative to the
Markdown document that contains it. Index and nested run pages therefore resolve to the same
recorded payload even when a bundle is published beside the immutable record. Direct rendering
accepts `record_link_prefix` for callers that own publication themselves.

For a custom recorder-level experiment publication, pass an `ExperimentPublisher` to
`FileRecorder`. The generic seam receives the current result, the not-yet-sealed experiment record,
and the staging root. Treat the root as read-only and return typed `ExperimentFile` values:

```python
from collections.abc import Sequence
from pathlib import Path

from autobench import ExperimentFile, ExperimentRecord, ExperimentResult


def publish_analysis(
    result: ExperimentResult,
    record: ExperimentRecord,
    experiment_root: Path,
) -> Sequence[ExperimentFile]:
    # Inspect staged record evidence through experiment_root; do not write to it directly.
    return (
        ExperimentFile(
            path="reports/custom.txt",
            content=f"runs={record.run_count}\n".encode(),
            identity=f"custom-report:{result.experiment_id}",
        ),
    )
```

The recorder validates collisions and writes returned files before sealing final metadata. The CLI
uses `MarkdownExperimentPublisher` to honor configured report output. Supplying the staging root is
important: configured reports can inspect the same recorded asset content and materialized artifact
paths as post-hoc replay reports.

## Comparison Integrity

Comparisons are paired by case. Each metric reports:

- baseline and candidate aggregate;
- absolute and relative delta;
- metric direction and resulting `improved`, `regressed`, `unchanged`, or `indeterminate` outcome;
- paired and missing pair counts;
- paired win/tie/loss count.

A lower latency or cost can therefore improve while a higher quality score improves. If direction
metadata conflicts across observations, Autobench does not mark a global winner. If multiple
factors changed, the report remains explicitly confounded and uses association language.

## Evidence And Missingness

Reports keep missing values distinct from numeric zero. Every leaderboard metric includes observed
and missing sample counts, and case matrices leave absent cells empty. Non-zero micro-costs retain
enough precision to avoid appearing as zero.

Findings link to typed experiment, run, metric, comparison, trace, asset, artifact, or optimization
identities. They describe the evidence available; they do not manufacture data for a cleaner story.

Artifact links come from the materialized path stored in the recorded `ArtifactRef.value`, never
from a user-facing filename. Autobench rejects absolute, escaping, missing, symlink-escaped, or
hash/size-mismatched payloads and reports a notice instead of emitting a broken or unsafe link.
Malformed optional run or asset detail cannot erase valid aggregate evidence.

## Safety And Immutability

Markdown values are treated as untrusted application data:

- HTML is escaped;
- table separators and line breaks are neutralized;
- links are relative and containment-checked;
- binary artifacts are referenced rather than embedded;
- remote images and active HTML are never injected;
- captured content is bounded and permission-gated;
- report generation performs no network I/O.

Post-hoc publication refuses destinations inside a finalized experiment root. Write the report next
to the record or into a separate analysis directory. Configured in-run publication is safe because
the files are staged before `experiment.yaml` and `manifest.yaml` are finalized.

Single files use a sibling temporary file and atomic replacement. Bundles are fully rendered in a
sibling staging directory before publication; an overwrite replaces the old bundle only after the
new bundle is complete.

## Charts And Optional PDF

Autobench emits deterministic inline SVG only when the evidence has a defined visual meaning:

- a stacked quality-gate pass/fail bar;
- a case-score ranking with pass/fail color encoding;
- normalized quality-dimension averages with sample counts.

Charts escape labels, use no remote assets, perform no network I/O, and remain readable as vector
graphics in Markdown viewers and PDF. Autobench intentionally does not guess arbitrary plots from
unclassified metrics. Matplotlib, Mermaid, dashboarding, and custom chart DSLs remain outside the
core report dependency graph.

PDF is an optional downstream projection rather than an Autobench runtime dependency. One supported
conversion uses `markdown-pdf>=1.13.2`:

```python
from pathlib import Path

from markdown_pdf import MarkdownPdf, Section

source = Path("analysis/support-routing.md")
pdf = MarkdownPdf(toc_level=2, optimize=True)
pdf.add_section(
    Section(
        source.read_text(encoding="utf-8"),
        root=str(source.parent),
        paper_size="A4",
    )
)
pdf.save("analysis/support-routing.pdf")
```

Keep PDF conversion outside the immutable benchmark record unless it is installed as an explicit
recorder publisher. The Markdown report remains the portable, deterministic document source.
