# Data And Runtime

## Contents

- [Cases](#cases)
- [Datasets](#datasets)
- [Variants and factors](#variants-and-factors)
- [Generated and production cases](#generated-and-production-cases)
- [Matrix planning](#matrix-planning)
- [Execution and concurrency](#execution-and-concurrency)
- [Failure behavior](#failure-behavior)
- [Reproducibility](#reproducibility)

## Cases

`Case` is one representative input and expected outcome. It carries:

- stable `id`;
- arbitrary `input`;
- optional `expected`;
- metadata and tags;
- artifact or attachment references when payloads should not be inline.

Keep case IDs stable across benchmark versions. Do not encode the variant or run number into the
case ID. Expected values describe evaluation targets; they are not task configuration.

Validate domain payloads inside the task with Pydantic models, dataclasses, or another structured
adapter. Autobench intentionally keeps case payloads generic.

## Datasets

Datasets can be:

- inline in the benchmark;
- loaded from one YAML file;
- loaded from glob-backed case files;
- assembled in Python;
- derived from reviewed production or generated samples.

`CaseDefaults` and dataset defaults can provide shared metadata, tags, input fragments, or expected
fragments. Merge defaults structurally and preserve case overrides. Record dataset IDs, versions,
and content hashes.

Use attachments or artifact references for images, binaries, large text, and other payloads that
should not be duplicated in YAML. Preserve media type, content hash, and portable relative paths.

## Variants And Factors

A `Variant` is one complete configuration in the intervention space. A `FactorValue` can declare:

- name and value;
- semantic type;
- label or metadata;
- whether the factor is controllable by an optimizer.

Examples:

```python
from autobench import FactorValue, Variant

current = Variant(
    id="current",
    factors=[
        FactorValue(name="model", value="demo-small", semantic_type="llm.model.name"),
        FactorValue(name="prompt", value="v3", optimize=True),
    ],
)
```

Use meaningful IDs such as `current`, `candidate`, or `indexed_search`; avoid `case_1` because case
already has a distinct meaning.

## Generated And Production Cases

Production ingestion primitives retain provenance, reason, review status, sampling policy, and
source metadata. Generated case batches retain generator identity and review state. Convert only
reviewed or intentionally accepted samples into regression cases.

Relevant public primitives include:

- `ProductionSample`, `SamplingPolicy`, `SampleReason`, `ReviewStatus`;
- `sample_to_case()`, `samples_to_cases()`;
- `CaseGeneratorInput`, `GeneratedCaseBatch`;
- `mark_generated_case()`, `generated_batch_from_cases()`.

Do not silently turn raw production traffic into permanent expected outcomes. Redact sensitive
fields, record provenance, and require an explicit review policy.

## Matrix Planning

The basic plan is deterministic `Dataset[Case] x Variant[Factor]`. Planning should expose run
count, source warnings, and duplicate or invalid IDs before the subject executes.

Controlled experiment matrices may include:

- baseline;
- prompt-only candidate;
- tool-only candidate;
- combined candidate.

This permits later analysis of main and interaction effects. Autobench stores the evidence and
confounding flags; an optimizer or experiment planner chooses follow-up interventions.

## Execution And Concurrency

Tasks may be sync or async. `run_benchmark_spec()` is async; `run_benchmark_spec_sync()` and
`Benchmark.run()` are sync entrypoints. Use a bounded concurrency limit for remote or expensive
subjects.

Concurrency invariants:

- each run has an isolated `RunContext`;
- ABP parentage is task-local;
- sibling runs do not leak factors, spans, assets, or capture settings;
- result order is deterministic even when execution completes out of order;
- cancellation propagates while preserving partial evidence;
- instrumentors install once per benchmark lifecycle and close reliably.

Do not add batch-level candidate injection without defining its concurrency and isolation model.

## Failure Behavior

Run statuses distinguish passed, failed, errored, and skipped outcomes:

- a failed constraint or check produces a failed run;
- a task exception produces an errored run with structured error identity;
- unsupported or absent execution may be skipped;
- later runs continue when failure isolation permits;
- cancellation remains cancellation rather than being converted to a generic error.

Task errors, scorer errors, instrumentation diagnostics, and policy failures are separate evidence.
Never replace the subject's exception or return value with an instrumentation failure.

## Reproducibility

Record:

- benchmark and dataset source hashes;
- task and custom scorer module hashes;
- case snapshot and factor values;
- environment and package metadata;
- behavioral asset versions;
- trace/instrumentor/source-map versions;
- pricing source identity when cost is derived.

Portable records use paths relative to the experiment directory. Do not infer reproducibility from
a run ID alone; the source, factors, assets, environment, and artifacts form the actual lineage.

