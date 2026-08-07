from __future__ import annotations as _annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

from opentelemetry.sdk.trace import ReadableSpan
from opentelemetry.sdk.trace.export import SpanExporter, SpanExportResult

from autobench import OTLPSettings, export_record_otlp


class CapturingExporter(SpanExporter):
    """Offline delivery target used to demonstrate the public exporter boundary."""

    def __init__(self) -> None:
        self.spans: list[ReadableSpan] = []

    def export(self, spans: Sequence[ReadableSpan]) -> SpanExportResult:
        self.spans.extend(spans)
        return SpanExportResult.SUCCESS

    def shutdown(self) -> None:
        return None


def main() -> None:
    parser = argparse.ArgumentParser(description="Map one Autobench record to OTLP spans offline.")
    parser.add_argument("record", type=Path)
    args = parser.parse_args()
    exporter = CapturingExporter()
    result = export_record_otlp(
        args.record,
        settings=OTLPSettings(
            service_name="autobench-offline-example",
            resource_attributes={"deployment.environment": "example"},
        ),
        exporter=exporter,
    )
    if len(exporter.spans) != result.exported_span_count:
        raise RuntimeError("OTLP result count does not match the delivered spans")
    print(
        f"Exported {result.exported_span_count} spans from "
        f"{result.run_count} runs ({result.partial_run_count} partial)."
    )


if __name__ == "__main__":
    main()
