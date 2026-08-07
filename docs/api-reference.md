# API Reference

This reference is generated from the installed public `autobench` package. The root package is the
supported import surface; subpackages organize implementation and extension areas.

## Public Package

::: autobench
    options:
      members: true
      members_order: source
      show_root_heading: true
      show_source: true
      show_signature_annotations: true
      separate_signature: true
      heading_level: 3

## Progress Runtime

- `ProgressEvent` carries a monotonic `sequence`, stable benchmark/run identity, event-specific
  `data`, and typed `run_status` or `experiment_status` terminal fields.
- `ProgressEventKind` defines benchmark start/finish, run start/finish, and actual policy violation
  events. Candidate decisions belong to Autoptimize rather than the benchmark lifecycle.
- `ProgressHandler` accepts synchronous and asynchronous observers.
- `ProgressErrorPolicy` selects strict library delivery or explicit best-effort delivery.
- `ProgressHandlerFailure` identifies the handler index, event sequence/kind, and original error.
- `ProgressDispatchError` is raised after strict delivery failure, terminal notification attempts,
  and durable recorder cleanup.

`run_benchmark_spec()`, `run_benchmark_path()`, `Benchmark.run()`, and `Benchmark.run_async()` all
accept `progress_handlers`, `progress_error_policy`, and `progress_error_handler`.

## Public Areas

| Area | Representative symbols |
| --- | --- |
| Definition | `Benchmark`, `BenchmarkSpec`, `TaskSpec`, `load_benchmark_spec` |
| Data | `Case`, `DatasetSpec`, `Variant`, `FactorValue`, production helpers, typed generated-dataset preparation and provenance |
| Runtime | `RunContext`, `RunPhase`, `Span`, `ExperimentResult`, `run_benchmark_spec`; durable `await ctx.checkpoint(name)` and cooperative cancellation |
| Semantics | `Observation`, `Semantic`, `SemanticRegistry`, queries and projection |
| Evaluation | scorers, derivers, policies, expected actions, measurement, feedback |
| Protocol | ABP signals, traces, capture, emitter, collector and context |
| Instrumentation | settings, manager, instrumentors, method instrumentation, diagnostics, `CurrentSpan`, and `InstrumentationRuntime.span()` / `metric()` / `current_span()` for external backend composition |
| Tracking | `track`, asset models, discovery candidates, registry and history views |
| Records | `RunRecord`, `ExperimentRecord`, `ExperimentTermination`, `RecordManifest`, `FileRecorder`, frozen staging snapshots, inspection/recovery, atomic/synced publication, and replay helpers |
| Reports | report models, builders, Rich renderers and exporters |
| Telemetry export | `OTLPSettings`, `OTLPExportResult`, `export_otlp`, and `export_record_otlp` |

`ExecutionCorrelation` is the public invocation metadata model. It is accepted by the full spec,
fluent builder, run functions, and CLI; persisted on results and records; and consumed by
`correlation_matches()`, `filter_experiments()`, and `build_grouped_reports()`. It remains separate
from replay lineage and application workflow state.

Prefer root imports for application code:

```python
from autobench import Benchmark, Case, RunContext, Semantic
```

Import a submodule when implementing an extension against that subsystem, such as a custom native
instrumentor or source-map adapter.
