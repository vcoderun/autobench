# Offline OTLP Export

Run any recorded example, then map its immutable evidence through the public OTLP exporter without
network access:

```bash
uv run autobench run examples/abp_manual/autobench.yaml --record /tmp/abp-manual
uv run python examples/otlp_export/export_record.py /tmp/abp-manual
```

`CapturingExporter` is a test delivery target. Production code normally omits the injected
exporter and configures an OTLP HTTP/protobuf endpoint through `OTLPSettings`, CLI flags, or OTel
exporter environment variables.
