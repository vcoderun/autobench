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

## Public Areas

| Area | Representative symbols |
| --- | --- |
| Definition | `Benchmark`, `BenchmarkSpec`, `TaskSpec`, `load_benchmark_spec` |
| Data | `Case`, `DatasetSpec`, `Variant`, `FactorValue`, production/generated helpers |
| Runtime | `RunContext`, `Span`, `ExperimentResult`, `run_benchmark_spec` |
| Semantics | `Observation`, `Semantic`, `SemanticRegistry`, queries and projection |
| Evaluation | scorers, derivers, policies, expected actions, measurement, feedback |
| Protocol | ABP signals, traces, capture, emitter, collector and context |
| Instrumentation | settings, manager, instrumentors, method instrumentation, diagnostics |
| Tracking | `track`, asset models, discovery candidates, registry and history views |
| Records | `RunRecord`, `ExperimentRecord`, recording and replay helpers |
| Reports | report models, builders, Rich renderers and exporters |

Prefer root imports for application code:

```python
from autobench import Benchmark, Case, RunContext, Semantic
```

Import a submodule when implementing an extension against that subsystem, such as a custom native
instrumentor or source-map adapter.
