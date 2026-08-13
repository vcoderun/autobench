from __future__ import annotations

import asyncio
import hashlib
import os
import sys
import threading
import tracemalloc
from collections.abc import AsyncIterator, Iterator
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

import autobench.records.staging as staging_module
import autobench.runtime.context as context_module
from autobench import (
    ArtifactOverflow,
    ArtifactRef,
    ArtifactSinkRequiredError,
    ArtifactSource,
    ArtifactState,
    ArtifactTransferError,
    BenchmarkInfo,
    BenchmarkSpec,
    Case,
    DatasetSpec,
    ExperimentStatus,
    ExperimentTermination,
    FileRecorder,
    PartialRunSnapshot,
    RecordDurability,
    RecordingError,
    ReplayError,
    RunContext,
    RunPhase,
    SymlinkPolicy,
    TaskSpec,
    Variant,
    build_benchmark_plan,
    capture_environment,
    discard_staging,
    expand_matrix,
    inspect_staging,
    load_run_record,
    run_benchmark_spec,
)
from autobench.io import dump_yaml
from autobench.records.recording import artifact_payload_path, record_artifact
from autobench.records.staging import ExperimentStart, FileRecordSession, load_staging_manifest
from autobench.records.views import run_record_to_yaml_view


async def open_artifact_context(
    tmp_path: Path,
    *,
    durability: RecordDurability = "atomic",
) -> tuple[RunContext, FileRecordSession, Path]:
    case = Case(id="case")
    variant = Variant(id="baseline")
    spec = BenchmarkSpec(
        benchmark=BenchmarkInfo(id="artifacts"),
        dataset=DatasetSpec(id="dataset", cases=[case]),
        variants=[variant],
        task=TaskSpec(kind="python", target="unused:run"),
    )
    run_spec = expand_matrix(spec, experiment_id="exp_artifacts")[0]
    recorder = FileRecorder(tmp_path / "record", durability=durability)
    session = await recorder.open(
        ExperimentStart(
            experiment_id="exp_artifacts",
            benchmark_id="artifacts",
            plan=build_benchmark_plan(spec),
            runs=(run_spec,),
            environment=capture_environment(),
            semantic_registry=spec.semantic_registry,
        )
    )
    ctx = RunContext(
        benchmark_id="artifacts",
        experiment_id="exp_artifacts",
        run_id=run_spec.run_id,
        case=case,
        variant=variant,
    )
    ctx.bind_artifact_sink(session)
    return ctx, session, recorder.staging_dir


def test_file_and_stream_artifacts_require_a_sink_before_touching_sources(tmp_path: Path) -> None:
    ctx = RunContext(
        benchmark_id="artifacts",
        case=Case(id="case"),
        variant=Variant(id="baseline"),
    )
    consumed = False

    def chunks() -> Iterator[bytes]:
        nonlocal consumed
        consumed = True
        yield b"payload"

    with pytest.raises(ArtifactSinkRequiredError):
        ctx.artifact_stream("stream", chunks())
    with pytest.raises(ArtifactSinkRequiredError):
        ctx.artifact_file("file", tmp_path / "missing")
    assert consumed is False

    with pytest.raises(ValidationError, match="provided together"):
        ArtifactRef(id="invalid", name="invalid", sha256="0" * 64)
    with pytest.raises(ValidationError, match="must be complete"):
        ArtifactRef(id="invalid", name="invalid", state=ArtifactState.PARTIAL)


def test_record_artifact_validates_prepared_payload_metadata(tmp_path: Path) -> None:
    artifacts_dir = tmp_path / "artifacts"
    content = b"prepared"
    digest = hashlib.sha256(content).hexdigest()

    unavailable = ArtifactRef(
        id="unavailable",
        name="unavailable",
        source=ArtifactSource.STREAM,
        sha256=digest,
        byte_count=len(content),
    )
    with pytest.raises(RecordingError, match="unavailable"):
        record_artifact(
            unavailable,
            artifacts_dir=artifacts_dir,
            root_dir=tmp_path,
            run_id="run",
            durability="atomic",
            prepared_path=tmp_path / "wrong.bin",
        )

    prepared = ArtifactRef(
        id="prepared",
        name="prepared",
        media_type="text/plain",
        source=ArtifactSource.STREAM,
        sha256=digest,
        byte_count=len(content),
    )
    prepared_path = artifact_payload_path(artifacts_dir, run_id="run", artifact=prepared)
    prepared_path.parent.mkdir(parents=True)
    prepared_path.write_bytes(content)
    recorded = record_artifact(
        prepared,
        artifacts_dir=artifacts_dir,
        root_dir=tmp_path,
        run_id="run",
        durability="atomic",
        prepared_path=prepared_path,
    )
    assert recorded.value == "artifacts/run/prepared.txt"
    assert (
        record_artifact(
            prepared,
            artifacts_dir=artifacts_dir,
            root_dir=tmp_path,
            run_id="run",
            durability="atomic",
            prepared_path=prepared_path,
        ).value
        == recorded.value
    )

    prepared_path.write_bytes(b"changed")
    with pytest.raises(RecordingError, match="changed after capture"):
        record_artifact(
            prepared,
            artifacts_dir=artifacts_dir,
            root_dir=tmp_path,
            run_id="run",
            durability="atomic",
            prepared_path=prepared_path,
        )

    for artifact_id, artifact_hash, byte_count, message in (
        ("bad-hash", "0" * 64, len(content), "hash mismatch"),
        ("bad-size", digest, len(content) + 1, "byte count mismatch"),
    ):
        invalid = prepared.model_copy(
            update={"id": artifact_id, "sha256": artifact_hash, "byte_count": byte_count}
        )
        path = artifact_payload_path(artifacts_dir, run_id="run", artifact=invalid)
        path.write_bytes(content)
        with pytest.raises(RecordingError, match=message):
            record_artifact(
                invalid,
                artifacts_dir=artifacts_dir,
                root_dir=tmp_path,
                run_id="run",
                durability="atomic",
                prepared_path=path,
            )


async def test_file_snapshot_is_portable_and_replay_detects_tampering(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "source.bin"
    source.write_bytes(b"original payload")
    (tmp_path / "stream_task.py").write_text(
        """
from pathlib import Path

def run(ctx, case):
    source = Path(case.input["source"])
    artifact = ctx.artifact_file("snapshot", source, media_type="application/octet-stream")
    source.write_bytes(b"mutated after capture")
    return {"artifact_id": artifact.id}
""".lstrip(),
        encoding="utf-8",
    )
    monkeypatch.syspath_prepend(str(tmp_path))
    spec = BenchmarkSpec(
        benchmark=BenchmarkInfo(id="snapshot"),
        dataset=DatasetSpec(
            id="dataset",
            cases=[Case(id="case", input={"source": str(source)})],
        ),
        variants=[Variant(id="baseline")],
        task=TaskSpec(kind="python", target="stream_task:run"),
    )
    output = tmp_path / "record"

    result = await run_benchmark_spec(
        spec,
        experiment_id="exp_snapshot",
        recorder=FileRecorder(output),
    )
    assert result.runs[0].error is None, result.runs[0].error

    run_path = output / "cases" / "case" / "baseline" / "run.yaml"
    record = load_run_record(run_path, root_dir=output)
    artifact = record.artifacts[0]
    assert artifact.source is ArtifactSource.FILE
    assert artifact.state is ArtifactState.COMPLETE
    assert artifact.byte_count == len(b"original payload")
    assert artifact.sha256 == hashlib.sha256(b"original payload").hexdigest()
    assert isinstance(artifact.value, str)
    assert not Path(artifact.value).is_absolute()
    assert str(tmp_path) not in str(artifact.model_dump(mode="json"))
    payload = output / artifact.value
    assert payload.read_bytes() == b"original payload"

    malformed = tmp_path / "malformed-run.yaml"
    for invalid, message in (
        (artifact.model_copy(update={"value": {"not": "a path"}}), "must be a string"),
        (artifact.model_copy(update={"value": "../../escape.bin"}), "escapes the experiment"),
        (artifact.model_copy(update={"value": "artifacts/missing.bin"}), "does not exist"),
        (
            artifact.model_copy(update={"byte_count": len(b"original payload") + 1}),
            "byte count mismatch",
        ),
    ):
        malformed.write_text(
            dump_yaml(
                run_record_to_yaml_view(record.model_copy(update={"artifacts": (invalid,)})),
                schema_name="run_record",
            ),
            encoding="utf-8",
        )
        with pytest.raises(ReplayError, match=message):
            load_run_record(malformed, root_dir=output)

    payload.write_bytes(b"tampered")
    with pytest.raises(ReplayError, match="hash mismatch"):
        load_run_record(run_path, root_dir=output)


async def test_stream_limits_close_sources_and_preserve_partial_metadata(tmp_path: Path) -> None:
    ctx, session, _ = await open_artifact_context(tmp_path)
    closed = False

    def overflowing() -> Iterator[bytes]:
        nonlocal closed
        try:
            yield b"1234"
            yield b"5678"
        finally:
            closed = True

    truncated = ctx.artifact_stream(
        "truncated",
        overflowing(),
        max_bytes=5,
        overflow=ArtifactOverflow.TRUNCATE,
        filename="result.bin",
    )
    assert closed is True
    assert truncated.state is ArtifactState.TRUNCATED
    assert truncated.byte_count == 5
    assert session.prepared_artifacts[(ctx.run_id, truncated.id)][1].read_bytes() == b"12345"

    closed = False
    with pytest.raises(ArtifactTransferError) as error:
        ctx.artifact_stream("failed", overflowing(), max_bytes=5)
    assert closed is True
    assert error.value.artifact.state is ArtifactState.PARTIAL
    assert ctx.artifacts[-1].state is ArtifactState.PARTIAL
    assert ctx.artifacts[-1].byte_count == 5

    with pytest.raises(ArtifactTransferError):
        ctx.artifact_stream("exact-limit", iter((b"12345", b"6")), max_bytes=5)
    assert ctx.artifacts[-1].byte_count == 5

    def broken() -> Iterator[bytes]:
        yield b"part"
        raise RuntimeError("source failed")

    with pytest.raises(RuntimeError, match="source failed"):
        ctx.artifact_stream("broken", broken(), max_bytes=100)
    assert ctx.artifacts[-1].state is ArtifactState.PARTIAL
    assert ctx.artifacts[-1].byte_count == 4

    with pytest.raises(ValueError, match="at least 1"):
        ctx.artifact_stream("invalid-limit", iter((b"unused",)), max_bytes=0)
    invalid_source: Any = iter(("text",))
    with pytest.raises(TypeError, match="must yield bytes"):
        ctx.artifact_stream("invalid-chunk", invalid_source, max_bytes=100)

    class CloseFailure:
        def __init__(self) -> None:
            self.done = False

        def __iter__(self) -> CloseFailure:
            return self

        def __next__(self) -> bytes:
            if self.done:
                raise StopIteration
            self.done = True
            return b"value"

        def close(self) -> None:
            raise RuntimeError("close failed")

    with pytest.raises(RuntimeError, match="close failed"):
        ctx.artifact_stream("close-failure", CloseFailure(), max_bytes=100)
    assert ctx.artifacts[-1].state is ArtifactState.PARTIAL

    class FailingIterator:
        def __init__(self) -> None:
            self.closed = False

        def __iter__(self) -> FailingIterator:
            return self

        def __next__(self) -> bytes:
            raise RuntimeError("iteration failed")

        def close(self) -> None:
            self.closed = True

    class FailingSource:
        def __init__(self) -> None:
            self.iterator = FailingIterator()
            self.closed = False

        def __iter__(self) -> FailingIterator:
            return self.iterator

        def close(self) -> None:
            self.closed = True
            raise RuntimeError("outer close failed")

    failing_source = FailingSource()
    with pytest.raises(RuntimeError, match="iteration failed"):
        ctx.artifact_stream("outer-close", failing_source, max_bytes=100)
    assert failing_source.iterator.closed is True
    assert failing_source.closed is True

    with pytest.raises(RecordingError, match="already exists"):
        session.prepare_stream(
            run_id=ctx.run_id,
            artifact_id=truncated.id,
            name="duplicate",
            source=iter((b"duplicate",)),
            media_type=None,
            max_bytes=100,
            overflow=ArtifactOverflow.FAIL,
            filename=None,
            span_id=None,
            tags={},
        )
    with pytest.raises(RecordingError, match="not part of the experiment"):
        session.prepare_stream(
            run_id="unknown",
            artifact_id="artifact",
            name="unknown",
            source=iter((b"unused",)),
            media_type=None,
            max_bytes=100,
            overflow=ArtifactOverflow.FAIL,
            filename=None,
            span_id=None,
            tags={},
        )


async def test_async_stream_cancellation_closes_source_and_keeps_partial_payload(
    tmp_path: Path,
) -> None:
    ctx, session, _ = await open_artifact_context(tmp_path)
    first_chunk = asyncio.Event()
    release = asyncio.Event()

    class SlowStream:
        def __init__(self) -> None:
            self.index = 0
            self.closed = False

        def __aiter__(self) -> SlowStream:
            return self

        async def __anext__(self) -> bytes:
            if self.index == 0:
                self.index += 1
                first_chunk.set()
                return b"partial"
            await release.wait()
            raise StopAsyncIteration

        async def aclose(self) -> None:
            self.closed = True

    source = SlowStream()
    transfer = asyncio.create_task(ctx.artifact_stream_async("async", source, max_bytes=100))
    await first_chunk.wait()
    await asyncio.sleep(0)
    transfer.cancel()
    with pytest.raises(asyncio.CancelledError):
        await transfer

    assert source.closed is True
    artifact = ctx.artifacts[0]
    assert artifact.state is ArtifactState.PARTIAL
    assert artifact.byte_count == len(b"partial")
    assert session.prepared_artifacts[(ctx.run_id, artifact.id)][1].read_bytes() == b"partial"

    async def overflowing() -> AsyncIterator[bytes]:
        yield b"12345"
        yield b"6"

    with pytest.raises(ArtifactTransferError):
        await ctx.artifact_stream_async("async-overflow", overflowing(), max_bytes=5)
    assert ctx.artifacts[-1].state is ArtifactState.PARTIAL

    async def truncated_chunks() -> AsyncIterator[bytes]:
        yield b"123456"

    truncated = await ctx.artifact_stream_async(
        "async-truncated",
        truncated_chunks(),
        max_bytes=5,
        overflow=ArtifactOverflow.TRUNCATE,
        filename="truncated.bin",
    )
    assert truncated.state is ArtifactState.TRUNCATED

    class AsyncCloseFailure:
        def __init__(self) -> None:
            self.done = False

        def __aiter__(self) -> AsyncCloseFailure:
            return self

        async def __anext__(self) -> bytes:
            if self.done:
                raise StopAsyncIteration
            self.done = True
            return b"value"

        async def aclose(self) -> None:
            raise RuntimeError("async close failed")

    with pytest.raises(RuntimeError, match="async close failed"):
        await ctx.artifact_stream_async("async-close-failure", AsyncCloseFailure())
    assert ctx.artifacts[-1].state is ArtifactState.PARTIAL

    class FailingAsyncIterator:
        def __init__(self) -> None:
            self.closed = False

        def __aiter__(self) -> FailingAsyncIterator:
            return self

        async def __anext__(self) -> bytes:
            raise RuntimeError("async iteration failed")

        async def aclose(self) -> None:
            self.closed = True

    class FailingAsyncSource:
        def __init__(self) -> None:
            self.iterator = FailingAsyncIterator()
            self.closed = False

        def __aiter__(self) -> FailingAsyncIterator:
            return self.iterator

        async def aclose(self) -> None:
            self.closed = True
            raise RuntimeError("outer async close failed")

    failing_source = FailingAsyncSource()
    with pytest.raises(RuntimeError, match="async iteration failed"):
        await ctx.artifact_stream_async("async-outer-close", failing_source)
    assert failing_source.iterator.closed is True
    assert failing_source.closed is True

    class PlainAsyncIterator:
        def __init__(self) -> None:
            self.done = False

        def __aiter__(self) -> PlainAsyncIterator:
            return self

        async def __anext__(self) -> bytes:
            if self.done:
                raise StopAsyncIteration
            self.done = True
            return b"plain"

    class ClosableAsyncSource:
        def __init__(self) -> None:
            self.iterator = PlainAsyncIterator()
            self.closed = False

        def __aiter__(self) -> PlainAsyncIterator:
            return self.iterator

        async def aclose(self) -> None:
            self.closed = True

    closable_source = ClosableAsyncSource()
    complete = await ctx.artifact_stream_async("outer-only-close", closable_source)
    assert complete.state is ArtifactState.COMPLETE
    assert closable_source.closed is True


async def test_file_policies_async_api_and_span_forwarding(tmp_path: Path) -> None:
    ctx, session, staging = await open_artifact_context(tmp_path)
    source = tmp_path / "source.txt"
    source.write_text("hello", encoding="utf-8")
    directory = tmp_path / "directory"
    directory.mkdir()
    symlink = tmp_path / "link.txt"
    symlink.symlink_to(source)

    with pytest.raises(RecordingError, match="symlink"):
        ctx.artifact_file("link", symlink)
    with pytest.raises(RecordingError, match="does not exist"):
        await ctx.artifact_file_async("missing", tmp_path / "missing.txt")
    with pytest.raises(RecordingError, match="directories"):
        ctx.artifact_file("directory", directory)
    with pytest.raises(RecordingError, match="path traversal"):
        ctx.artifact_file("traversal", source, filename="../escape.txt")
    with pytest.raises(RecordingError, match="path traversal"):
        ctx.artifact_file("dot", source, filename=".")

    pipe_path = tmp_path / "named-pipe"
    os.mkfifo(pipe_path)
    with pytest.raises(RecordingError, match="regular file"):
        ctx.artifact_file("pipe", pipe_path)

    followed = await ctx.artifact_file_async(
        "followed",
        symlink,
        symlinks=SymlinkPolicy.FOLLOW,
    )
    assert followed.symlink_followed is True
    with ctx.span("artifact-span") as span:
        file_ref = span.artifact_file("file", source)
        stream_ref = span.artifact_stream("stream", iter((b"one", b"two")))
        async_file_ref = await span.artifact_file_async("async-file", source)

        async def chunks() -> AsyncIterator[bytes]:
            yield b"async"

        async_stream_ref = await span.artifact_stream_async("async-stream", chunks())
    assert {
        file_ref.span_id,
        stream_ref.span_id,
        async_file_ref.span_id,
        async_stream_ref.span_id,
    } == {span.id}

    orphan = ctx.artifact_file("orphan", source, span_id="missing-span")
    assert orphan.span_id == "missing-span"
    text_stream = ctx.artifact_stream(
        "text-stream",
        iter((b"text",)),
        media_type="text/plain",
    )
    assert session.prepared_artifacts[(ctx.run_id, text_stream.id)][1].suffix == ".txt"
    with pytest.raises(RuntimeError, match="already bound"):
        ctx.bind_artifact_sink(session)

    stale = ArtifactRef(
        id="stale",
        name="stale",
        source=ArtifactSource.STREAM,
        filename="stale.bin",
    )
    stale_path = artifact_payload_path(staging / "artifacts", run_id=ctx.run_id, artifact=stale)
    stale_path.parent.mkdir(parents=True, exist_ok=True)
    stale_path.write_bytes(b"stale")
    with pytest.raises(RecordingError, match="already exists"):
        session.prepare_stream(
            run_id=ctx.run_id,
            artifact_id=stale.id,
            name=stale.name,
            source=iter((b"new",)),
            media_type=None,
            max_bytes=100,
            overflow=ArtifactOverflow.FAIL,
            filename=stale.filename,
            span_id=None,
            tags={},
        )


async def test_async_file_cancellation_waits_for_snapshot_and_retains_evidence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ctx, session, _ = await open_artifact_context(tmp_path)
    source = tmp_path / "slow.bin"
    source.write_bytes(b"snapshot")
    started = threading.Event()
    release = threading.Event()
    original = FileRecordSession.prepare_file

    def slow_prepare_file(
        active_session: FileRecordSession,
        *,
        run_id: str,
        artifact_id: str,
        name: str,
        source: Path,
        media_type: str | None,
        max_bytes: int,
        overflow: ArtifactOverflow,
        symlinks: SymlinkPolicy,
        filename: str | None,
        span_id: str | None,
        tags: dict[str, Any],
    ) -> ArtifactRef:
        started.set()
        release.wait()
        return original(
            active_session,
            run_id=run_id,
            artifact_id=artifact_id,
            name=name,
            source=source,
            media_type=media_type,
            max_bytes=max_bytes,
            overflow=overflow,
            symlinks=symlinks,
            filename=filename,
            span_id=span_id,
            tags=tags,
        )

    monkeypatch.setattr(FileRecordSession, "prepare_file", slow_prepare_file)
    transfer = asyncio.create_task(ctx.artifact_file_async("slow", source))
    assert await asyncio.to_thread(started.wait, 2)
    transfer.cancel()
    release.set()
    with pytest.raises(asyncio.CancelledError):
        await transfer

    assert len(ctx.artifacts) == 1
    assert ctx.artifacts[0].state is ArtifactState.COMPLETE
    assert (
        session.prepared_artifact(
            run_id=ctx.run_id,
            artifact_id=ctx.artifacts[0].id,
        )
        == ctx.artifacts[0]
    )


async def test_async_file_cancellation_is_bounded_while_session_owns_transfer(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ctx, session, _ = await open_artifact_context(tmp_path)
    source = tmp_path / "blocked.bin"
    source.write_bytes(b"snapshot")
    started = threading.Event()
    release = threading.Event()
    original = FileRecordSession.prepare_file

    def blocked_prepare_file(
        active_session: FileRecordSession,
        *,
        run_id: str,
        artifact_id: str,
        name: str,
        source: Path,
        media_type: str | None,
        max_bytes: int,
        overflow: ArtifactOverflow,
        symlinks: SymlinkPolicy,
        filename: str | None,
        span_id: str | None,
        tags: dict[str, Any],
    ) -> ArtifactRef:
        started.set()
        release.wait()
        return original(
            active_session,
            run_id=run_id,
            artifact_id=artifact_id,
            name=name,
            source=source,
            media_type=media_type,
            max_bytes=max_bytes,
            overflow=overflow,
            symlinks=symlinks,
            filename=filename,
            span_id=span_id,
            tags=tags,
        )

    monkeypatch.setattr(FileRecordSession, "prepare_file", blocked_prepare_file)
    monkeypatch.setattr(context_module, "_ARTIFACT_TRANSFER_SETTLE_SECONDS", 0.001)
    capture = asyncio.create_task(ctx.artifact_file_async("blocked", source))
    assert await asyncio.to_thread(started.wait, 2)

    capture.cancel()
    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(capture, timeout=1)

    assert len(ctx.artifacts) == 1
    assert ctx.artifacts[0].state is ArtifactState.PARTIAL
    close = asyncio.create_task(session.close())
    await asyncio.sleep(0)
    assert close.done() is False

    release.set()
    await asyncio.wait_for(close, timeout=1)
    assert session.closed is True
    prepared = session.prepared_artifact(
        run_id=ctx.run_id,
        artifact_id=ctx.artifacts[0].id,
    )
    assert prepared is not None
    assert prepared.state is ArtifactState.COMPLETE


async def test_direct_session_checkpoint_cancellation_has_bounded_ownership(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, session, _ = await open_artifact_context(tmp_path)
    run_spec = session.start.runs[0]
    first_started = asyncio.Event()
    first_release = asyncio.Event()
    second_started = asyncio.Event()
    second_release = asyncio.Event()
    original = FileRecordSession._checkpoint

    async def controlled_checkpoint(
        active_session: FileRecordSession,
        snapshot: PartialRunSnapshot,
    ) -> None:
        if snapshot.name == "settled-after-cancel":
            first_started.set()
            await first_release.wait()
        if snapshot.name == "owned-after-timeout":
            second_started.set()
            await second_release.wait()
        await original(active_session, snapshot)

    monkeypatch.setattr(FileRecordSession, "_checkpoint", controlled_checkpoint)
    first_snapshot = PartialRunSnapshot(
        run_id=run_spec.run_id,
        experiment_id=run_spec.experiment_id,
        benchmark_id=run_spec.benchmark_id,
        case_id=run_spec.case.id,
        variant_id=run_spec.variant.id,
        name="settled-after-cancel",
        phase=RunPhase.EXECUTING,
    )
    first = asyncio.create_task(session.checkpoint(first_snapshot))
    await first_started.wait()
    first.cancel()
    await asyncio.sleep(0)
    first_release.set()
    with pytest.raises(asyncio.CancelledError):
        await first
    assert {item.name for item in session.manifest.checkpoints} == {"settled-after-cancel"}

    monkeypatch.setattr(staging_module, "_SESSION_OPERATION_SETTLE_SECONDS", 0.001)
    second_snapshot = first_snapshot.model_copy(update={"name": "owned-after-timeout"})
    second = asyncio.create_task(session.checkpoint(second_snapshot))
    await second_started.wait()
    second.cancel()
    with pytest.raises(asyncio.CancelledError) as raised:
        await asyncio.wait_for(second, timeout=1)
    assert any("checkpoint" in note for note in raised.value.__notes__)

    close = asyncio.create_task(session.close())
    await asyncio.sleep(0)
    assert close.done() is False
    second_release.set()
    await asyncio.wait_for(close, timeout=1)
    assert {item.name for item in session.manifest.checkpoints} == {
        "settled-after-cancel",
        "owned-after-timeout",
    }


async def test_abort_rechecks_lifecycle_after_owned_operation_finishes(tmp_path: Path) -> None:
    _, session, _ = await open_artifact_context(tmp_path)
    operation_started = asyncio.Event()
    release_operation = asyncio.Event()

    async def finish_session() -> None:
        operation_started.set()
        await release_operation.wait()
        session.finished = True

    session._start_operation(finish_session())
    abort = asyncio.create_task(
        session.abort(
            ExperimentTermination(
                status=ExperimentStatus.ABORTED,
                partial=True,
            )
        )
    )
    await operation_started.wait()
    await asyncio.sleep(0)
    assert abort.done() is False

    release_operation.set()
    await abort

    assert session.state.status.value == "active"
    await session.close()


async def test_cancelled_file_worker_is_removed_from_session_ownership(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ctx, session, _ = await open_artifact_context(tmp_path)
    source = tmp_path / "cancelled.bin"
    source.write_bytes(b"payload")

    def cancel_prepare_file(
        active_session: FileRecordSession,
        *,
        run_id: str,
        artifact_id: str,
        name: str,
        source: Path,
        media_type: str | None,
        max_bytes: int,
        overflow: ArtifactOverflow,
        symlinks: SymlinkPolicy,
        filename: str | None,
        span_id: str | None,
        tags: dict[str, Any],
    ) -> ArtifactRef:
        raise asyncio.CancelledError

    monkeypatch.setattr(FileRecordSession, "prepare_file", cancel_prepare_file)
    with pytest.raises(asyncio.CancelledError):
        await ctx.artifact_file_async("cancelled", source)
    await asyncio.sleep(0)

    assert session.artifact_transfers == set()
    assert ctx.artifacts[0].state is ArtifactState.PARTIAL
    await session.close()


async def test_stream_publish_honors_synced_durability_and_cleans_failed_commit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ctx, _, staging = await open_artifact_context(tmp_path, durability="synced")
    synced = ctx.artifact_stream("synced", iter((b"durable",)))
    assert synced.state is ArtifactState.COMPLETE

    failed_ctx, _, failed_staging = await open_artifact_context(tmp_path / "failed")

    def fail_replace(source: Path, destination: Path) -> None:
        raise OSError(f"cannot publish {source.name} to {destination.name}")

    monkeypatch.setattr(staging_module.os, "replace", fail_replace)
    with pytest.raises(OSError, match="cannot publish"):
        failed_ctx.artifact_stream("failed", iter((b"payload",)))

    assert not tuple((failed_staging / "artifacts").rglob("*.tmp"))
    assert any(path.is_file() for path in (staging / "artifacts").rglob("*"))


async def test_checkpoint_commits_prepared_payload_and_detects_corruption(tmp_path: Path) -> None:
    ctx, session, staging = await open_artifact_context(tmp_path)
    inline = ctx.artifact("inline", {"status": "ready"})
    artifact = ctx.artifact_stream("checkpointed", iter((b"payload",)), max_bytes=100)
    snapshot = PartialRunSnapshot(
        run_id=ctx.run_id,
        experiment_id=ctx.experiment_id,
        benchmark_id=ctx.benchmark_id,
        case_id=ctx.case.id,
        variant_id=ctx.variant.id,
        name="payload-ready",
        phase=RunPhase.EXECUTING,
        artifacts=tuple(ctx.artifacts),
    )

    await session.checkpoint(snapshot)

    manifest = load_staging_manifest(staging)
    assert len(manifest.payloads) == 2
    checkpoint_path = staging / manifest.checkpoints[0].path
    checkpoint_text = checkpoint_path.read_text(encoding="utf-8")
    assert f"artifacts/run_0001_0001_case__baseline/{artifact.id}.bin" in checkpoint_text
    assert inspect_staging(staging).recoverable is True

    await session.checkpoint(snapshot)
    assert len(session.manifest.checkpoints) == 1

    second = snapshot.model_copy(update={"name": "payload-ready-again"})
    await session.checkpoint(second)
    assert len(session.manifest.checkpoints) == 2
    assert inline.source is ArtifactSource.VALUE

    unavailable = ArtifactRef(
        id="unavailable",
        name="unavailable",
        source=ArtifactSource.STREAM,
        sha256=hashlib.sha256(b"missing").hexdigest(),
        byte_count=len(b"missing"),
    )
    with pytest.raises(RecordingError, match="unavailable for checkpoint"):
        await session.checkpoint(
            snapshot.model_copy(
                update={
                    "name": "missing-payload",
                    "artifacts": (*snapshot.artifacts, unavailable),
                }
            )
        )

    payload_path = session.prepared_artifacts[(ctx.run_id, artifact.id)][1]
    payload_path.write_bytes(b"corrupt")
    with pytest.raises(RecordingError, match="changed after capture"):
        await session.checkpoint(snapshot.model_copy(update={"name": "corrupt-payload"}))
    inspection = inspect_staging(staging)
    assert inspection.recoverable is False
    assert ctx.run_id in inspection.corrupt_run_ids


@pytest.mark.large_artifact
async def test_generated_500_mib_stream_has_bounded_python_memory(tmp_path: Path) -> None:
    ctx, session, staging = await open_artifact_context(tmp_path)
    chunk = b"x" * (1024 * 1024)

    def chunks() -> Iterator[bytes]:
        for _ in range(500):
            yield chunk

    tracemalloc.start()
    artifact = ctx.artifact_stream(
        "large",
        chunks(),
        max_bytes=500 * 1024 * 1024,
        filename="large.bin",
    )
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    assert artifact.byte_count == 500 * 1024 * 1024
    assert peak < 16 * 1024 * 1024
    await session.close()
    discard_staging(staging)
    assert not staging.exists()


@pytest.fixture(autouse=True)
def clear_dynamic_modules() -> Iterator[None]:
    yield
    sys.modules.pop("stream_task", None)
