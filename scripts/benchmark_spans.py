from __future__ import annotations

import argparse
import json
import tracemalloc
from collections.abc import Iterator
from dataclasses import asdict, dataclass
from statistics import median
from timeit import repeat

import httpx

from autobench import Case, RunContext, Variant
from autobench.instrumentation.httpx import HTTPX
from autobench.instrumentation.manager import InstrumentationManager
from autobench.runtime.instrumentation import reset_active_run_context, set_active_run_context


@dataclass(frozen=True, slots=True)
class SpanBenchmarkResult:
    iterations: int
    repeats: int
    minimum_ns: float
    median_ns: float
    samples_ns: tuple[float, ...]


@dataclass(frozen=True, slots=True)
class HTTPXBenchmarkResult:
    iterations: int
    repeats: int
    baseline_median_ns: float
    instrumented_median_ns: float
    overhead_ns: float
    overhead_ratio: float
    instrumented_peak_bytes: int


@dataclass(frozen=True, slots=True)
class HTTPXStreamBenchmarkResult:
    chunks: int
    chunk_size_bytes: int
    repeats: int
    median_ns_per_chunk: float
    peak_bytes: int
    peak_bytes_per_chunk: float


def benchmark_manual_spans(*, iterations: int, repeats: int) -> SpanBenchmarkResult:
    if iterations < 1:
        raise ValueError("iterations must be at least 1")
    if repeats < 1:
        raise ValueError("repeats must be at least 1")

    def run_batch() -> None:
        ctx = RunContext(
            benchmark_id="abp-span-baseline",
            case=Case(id="case"),
            variant=Variant(id="variant"),
        )
        for _ in range(iterations):
            with ctx.span("manual"):
                pass

    elapsed = repeat(run_batch, number=1, repeat=repeats)
    samples_ns = tuple(sample * 1_000_000_000 / iterations for sample in elapsed)
    return SpanBenchmarkResult(
        iterations=iterations,
        repeats=repeats,
        minimum_ns=min(samples_ns),
        median_ns=median(samples_ns),
        samples_ns=samples_ns,
    )


def benchmark_httpx_transport(*, iterations: int, repeats: int) -> HTTPXBenchmarkResult:
    if iterations < 1:
        raise ValueError("iterations must be at least 1")
    if repeats < 1:
        raise ValueError("repeats must be at least 1")

    client = httpx.Client(
        transport=httpx.MockTransport(lambda request: httpx.Response(204, request=request))
    )

    def baseline_batch() -> None:
        for _ in range(iterations):
            client.get("https://benchmark.test/ping")

    def instrumented_batch() -> None:
        ctx = RunContext(
            benchmark_id="abp-httpx-overhead",
            case=Case(id="case"),
            variant=Variant(id="variant"),
        )
        manager = InstrumentationManager()
        manager.install(HTTPX())
        token = set_active_run_context(ctx)
        try:
            for _ in range(iterations):
                client.get("https://benchmark.test/ping")
        finally:
            reset_active_run_context(token)
            manager.close()

    baseline_samples = repeat(baseline_batch, number=1, repeat=repeats)
    instrumented_samples = repeat(instrumented_batch, number=1, repeat=repeats)
    baseline_median_ns = median(baseline_samples) * 1_000_000_000 / iterations
    instrumented_median_ns = median(instrumented_samples) * 1_000_000_000 / iterations

    tracemalloc.start()
    instrumented_batch()
    _, instrumented_peak_bytes = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    client.close()

    overhead_ns = instrumented_median_ns - baseline_median_ns
    return HTTPXBenchmarkResult(
        iterations=iterations,
        repeats=repeats,
        baseline_median_ns=baseline_median_ns,
        instrumented_median_ns=instrumented_median_ns,
        overhead_ns=overhead_ns,
        overhead_ratio=instrumented_median_ns / baseline_median_ns,
        instrumented_peak_bytes=instrumented_peak_bytes,
    )


def benchmark_httpx_stream(
    *,
    chunks: int,
    chunk_size_bytes: int,
    repeats: int,
) -> HTTPXStreamBenchmarkResult:
    if chunks < 1:
        raise ValueError("chunks must be at least 1")
    if chunk_size_bytes < 1:
        raise ValueError("chunk_size_bytes must be at least 1")
    if repeats < 1:
        raise ValueError("repeats must be at least 1")

    class ChunkStream(httpx.SyncByteStream):
        def __iter__(self) -> Iterator[bytes]:
            chunk = b"x" * chunk_size_bytes
            for _ in range(chunks):
                yield chunk

    client = httpx.Client(
        transport=httpx.MockTransport(
            lambda request: httpx.Response(200, request=request, stream=ChunkStream())
        )
    )
    manager = InstrumentationManager()
    manager.install(HTTPX())

    def consume_stream() -> None:
        ctx = RunContext(
            benchmark_id="abp-httpx-stream-memory",
            case=Case(id="case"),
            variant=Variant(id="variant"),
        )
        token = set_active_run_context(ctx)
        try:
            with client.stream("GET", "https://benchmark.test/stream") as response:
                for _ in response.iter_bytes():
                    pass
        finally:
            reset_active_run_context(token)

    try:
        elapsed = repeat(consume_stream, number=1, repeat=repeats)
        tracemalloc.start()
        consume_stream()
        _, peak_bytes = tracemalloc.get_traced_memory()
        tracemalloc.stop()
    finally:
        manager.close()
        client.close()

    median_ns_per_chunk = median(elapsed) * 1_000_000_000 / chunks
    return HTTPXStreamBenchmarkResult(
        chunks=chunks,
        chunk_size_bytes=chunk_size_bytes,
        repeats=repeats,
        median_ns_per_chunk=median_ns_per_chunk,
        peak_bytes=peak_bytes,
        peak_bytes_per_chunk=peak_bytes / chunks,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Measure Autobench manual span overhead.")
    parser.add_argument("--iterations", type=int, default=10_000)
    parser.add_argument("--repeats", type=int, default=7)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--httpx", action="store_true")
    mode.add_argument("--httpx-stream", action="store_true")
    parser.add_argument("--chunks", type=int, default=10_000)
    parser.add_argument("--chunk-size", type=int, default=32)
    args = parser.parse_args()
    if args.httpx_stream:
        result = benchmark_httpx_stream(
            chunks=args.chunks,
            chunk_size_bytes=args.chunk_size,
            repeats=args.repeats,
        )
    elif args.httpx:
        result = benchmark_httpx_transport(iterations=args.iterations, repeats=args.repeats)
    else:
        result = benchmark_manual_spans(iterations=args.iterations, repeats=args.repeats)
    print(
        json.dumps(
            asdict(result),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
