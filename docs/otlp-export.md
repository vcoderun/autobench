# OTLP Export

Autobench can replay an immutable experiment record into OTLP HTTP/protobuf traces for systems
such as Logfire, Datadog, or a vendor-neutral OpenTelemetry Collector. This is an outbound adapter,
not Autobench's collection protocol:

```text
application -> ABP instrumentation -> immutable Autobench record -> optional OTLP export
```

ABP remains the complete source of truth for semantic observations, benchmark identity, partial
execution, behavioral assets, and replay. Export never converts a record back into application
calls and never mutates the source directory.

## Install The Exporter

The base package does not import or install OpenTelemetry. Add the dedicated extra only on a worker
that exports records:

=== "uv"

    ```bash
    uv add 'autobench[otlp]'
    ```

=== "pip"

    ```bash
    python -m pip install 'autobench[otlp]'
    ```

This installs the OpenTelemetry SDK and HTTP/protobuf trace exporter. It does not replace ABP
instrumentors and does not enable OTel-to-ABP ingestion.

## Export From The CLI

Point the command at a completed or explicitly partial Autobench record:

```bash
autobench telemetry export runs/routing-42 \
  --endpoint https://collector.example/v1/traces \
  --header authorization 'Bearer ...' \
  --service-name routing-benchmark \
  --service-namespace evaluation
```

When `--endpoint` and `--header` are omitted, the underlying exporter can use standard
`OTEL_EXPORTER_OTLP_TRACES_ENDPOINT` and `OTEL_EXPORTER_OTLP_HEADERS` configuration. Credentials
belong in environment variables or CLI secret injection, never in a benchmark YAML file.

Useful options:

| Option | Meaning |
| --- | --- |
| `--endpoint` | OTLP HTTP/protobuf trace endpoint |
| `--header NAME VALUE` | Repeatable request header |
| `--timeout SECONDS` | Positive exporter timeout |
| `--service-name` | OTLP `service.name`; default is `autobench` |
| `--service-namespace` | Optional OTLP `service.namespace` |
| `--certificate-file` | Custom CA certificate path |
| `--include-captured-content` | Export captured bodies and other content-bearing evidence |

The command prints a Rich summary containing experiment and benchmark identity, record version,
run/trace/span counts, partial counts, and the selected endpoint. Export failures return a nonzero
exit status and leave the record unchanged.

## Export From Python

```python
from pathlib import Path

from autobench import OTLPSettings, export_record_otlp

result = export_record_otlp(
    Path("runs/routing-42"),
    settings=OTLPSettings(
        endpoint="https://collector.example/v1/traces",
        headers={"authorization": "Bearer ..."},
        timeout_seconds=10,
        service_name="routing-benchmark",
        service_namespace="evaluation",
        resource_attributes={"deployment.environment": "staging"},
    ),
)

print(result.exported_span_count)
```

`export_otlp(experiment, runs, ...)` accepts already loaded `ExperimentRecord` and `RunRecord`
models. It validates run count, unique run IDs, benchmark and experiment ownership, and execution
correlation before mapping anything. Tests and custom delivery layers may inject an OTel
`SpanExporter`; Autobench does not shut down an exporter it does not own.

## Evidence Mapping

One export creates this hierarchy:

```text
autobench.experiment <benchmark_id>
  autobench.run <case_id> / <variant_id>
    autobench.trace <trace_id-prefix>
      <ABP root span>
        <ABP child span>
```

| Autobench evidence | OTLP representation |
| --- | --- |
| Experiment lifecycle | Root span, status, identity attributes, termination event |
| Run lifecycle | Child span with run/case/variant, record path, status, and partial state |
| ABP trace envelope | Child span with protocol version, trace ID, partial state, diagnostics |
| ABP span | Nested span with original operation, timestamps, scope, source convention, usage, and stream attributes |
| Factor, observation, score | Timestamped span event on the run |
| Measurement | Timestamped event on its ABP span |
| Asset use and source snapshot | Run events preserving version/provenance identity |
| ABP diagnostic/reference | Trace or span events |
| Trace-and-span link target | Native OTLP span link plus the lossless ABP link event |
| Error | Error status and structured exception/termination event |

Historical benchmark measurements are exported as span events, not fabricated live OTLP metric
streams. Their canonical semantic type, unit, direction, role, and source remain in the event, so a
consumer can project them deliberately without confusing replay time with measurement time.

The exporter preserves experiment, benchmark, run, case, variant, record-version, ABP trace/span,
dataset, spec hash, manifest, correlation, instrumentation scope, package version, source-map, and
source-convention identities where available. Reserved Autobench resource identities cannot be
overridden through custom resource attributes.

## Partial And Cancelled Records

Completed, cancelled, aborted, and recovered partial records are all exportable. Experiment and run
spans retain terminal status, `partial`, end reason, planned/recorded/missing run IDs, and whether
cross-run derivation and policy evaluation completed. An open or partial ABP trace is marked as
partial rather than being represented as a successful complete trace.

OTLP export is reporting, not recovery. Use `autobench recording inspect` and
`autobench recording finalize --allow-partial` to publish recoverable staging evidence first.

## Content And Privacy

By default, export keeps structural and semantic evidence while omitting captured content:

- event bodies and ABP span outputs;
- score actual/expected values;
- non-metric observation values;
- retained source-fact values;
- tracebacks.

References, hashes, counts, semantic types, status, usage, asset IDs/versions, and provenance remain
available. `--include-captured-content` or `OTLPSettings(include_captured_content=True)` includes
content that ABP already captured. It cannot recover content excluded or redacted by the original
capture policy. Treat this switch as a deliberate data-export decision.

## Failure Boundary

Mapping, network export, exporter rejection, and owned-exporter shutdown failures raise
`OTLPExportError`. The original Pydantic record models and files are never changed. An injected
exporter remains caller-owned and is not shut down. This allows export retries or delivery to
several telemetry systems without changing benchmark evidence or replay lineage.

## What This Adapter Does Not Do

- It does not make OTel an Autobench runtime dependency.
- It does not ingest arbitrary OTel traces into ABP.
- It does not patch application SDKs through OTel instrumentations.
- It does not use OTLP as benchmark persistence.
- It does not add vendor configuration to `BenchmarkSpec`.

Use [Native Instrumentation](native-instrumentation.md) to collect automatic application evidence
into ABP, then use this adapter when the resulting record also needs to appear in an OTLP backend.
