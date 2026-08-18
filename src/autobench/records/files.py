from __future__ import annotations as _annotations

import hashlib
import os
import shutil
import tempfile
from enum import StrEnum
from pathlib import Path, PurePosixPath
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from autobench.records.models import RecordingError

RecordDurability = Literal["atomic", "synced"]


class RecordFileKind(StrEnum):
    EXPERIMENT = "experiment"
    SUMMARY = "summary"
    RUN = "run"
    TRACE = "trace"
    ARTIFACT = "artifact"
    ASSET = "asset"
    SOURCE = "source"
    OTHER = "other"


class LogicalRecordTarget(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    path: str = Field(min_length=1)
    kind: RecordFileKind
    identity: str = Field(min_length=1)


class ExperimentFile(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    path: str = Field(min_length=1)
    content: bytes
    kind: RecordFileKind = RecordFileKind.OTHER
    identity: str = Field(min_length=1)

    @field_validator("path")
    @classmethod
    def validate_path(cls, path: str) -> str:
        return normalize_logical_path(path)


class ManifestEntry(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    path: str = Field(min_length=1)
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    byte_count: int = Field(ge=0)
    kind: RecordFileKind
    identity: str = Field(min_length=1)

    @field_validator("path")
    @classmethod
    def validate_path(cls, value: str) -> str:
        try:
            return normalize_logical_path(value)
        except RecordingError as exc:
            raise ValueError(str(exc)) from exc


class RecordManifest(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    version: Literal[1] = 1
    experiment_id: str = Field(min_length=1)
    files: tuple[ManifestEntry, ...]


def validate_logical_targets(
    targets: tuple[LogicalRecordTarget, ...],
) -> dict[str, LogicalRecordTarget]:
    mapped: dict[str, LogicalRecordTarget] = {}
    for target in targets:
        normalized = normalize_logical_path(target.path)
        previous = mapped.get(normalized)
        if previous is not None:
            raise RecordingError(
                "Normalized record path collision: "
                f"{previous.path!r} ({previous.identity}) and "
                f"{target.path!r} ({target.identity}) both map to {normalized!r}"
            )
        mapped[normalized] = target.model_copy(update={"path": normalized})
    return mapped


def normalize_logical_path(path: str) -> str:
    candidate = PurePosixPath(path)
    if candidate.is_absolute() or any(part in {"", ".", ".."} for part in candidate.parts):
        raise RecordingError(f"Record path must be a normalized relative path: {path!r}")
    return candidate.as_posix()


def atomic_write_text(
    path: Path,
    content: str,
    *,
    durability: RecordDurability = "atomic",
) -> None:
    atomic_write_bytes(path, content.encode("utf-8"), durability=durability)


def atomic_write_bytes(
    path: Path,
    content: bytes,
    *,
    durability: RecordDurability = "atomic",
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(content)
            stream.flush()
            if durability == "synced":
                os.fsync(stream.fileno())
        os.replace(temporary_path, path)
        if durability == "synced":
            sync_directory(path.parent)
    except BaseException:
        temporary_path.unlink(missing_ok=True)
        raise


def create_temporary_record_directory(destination: Path) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    return Path(
        tempfile.mkdtemp(
            prefix=f".{destination.name}.finalizing-",
            dir=destination.parent,
        )
    )


def publish_record_directory(
    staging: Path,
    destination: Path,
    *,
    durability: RecordDurability = "atomic",
) -> None:
    if destination.is_symlink():
        raise RecordingError(f"Experiment record already exists: {destination}")
    if destination.exists():
        if not destination.is_dir() or any(destination.iterdir()):
            raise RecordingError(f"Experiment record already exists: {destination}")
        destination.rmdir()
    if durability == "synced":
        sync_tree(staging)
    os.replace(staging, destination)
    if durability == "synced":
        sync_directory(destination.parent)


def remove_temporary_record_directory(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)


def sync_tree(root: Path) -> None:
    if os.name != "posix":
        raise RecordingError("synced recording requires POSIX directory fsync support")
    for path in sorted(root.rglob("*")):
        if path.is_file():
            descriptor = os.open(path, os.O_RDONLY)
            try:
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
    directories = [path for path in root.rglob("*") if path.is_dir()]
    for directory in sorted(directories, key=lambda item: len(item.parts), reverse=True):
        sync_directory(directory)
    sync_directory(root)


def sync_directory(path: Path) -> None:
    if os.name != "posix":
        raise RecordingError("synced recording requires POSIX directory fsync support")
    try:
        descriptor = os.open(path, os.O_RDONLY)
    except OSError as exc:
        raise RecordingError(f"Could not open directory for fsync: {path}") from exc
    try:
        os.fsync(descriptor)
    except OSError as exc:
        raise RecordingError(f"Could not fsync directory: {path}") from exc
    finally:
        os.close(descriptor)


def build_manifest(
    root: Path,
    *,
    experiment_id: str,
    targets: dict[str, LogicalRecordTarget],
) -> RecordManifest:
    entries = tuple(
        _manifest_entry(root, path, targets=targets)
        for path in sorted(item for item in root.rglob("*") if item.is_file())
        if path.relative_to(root).as_posix() != "manifest.yaml"
    )
    return RecordManifest(experiment_id=experiment_id, files=entries)


def validate_manifest(root: Path, manifest: RecordManifest) -> None:
    actual_paths = {
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file() and path.relative_to(root).as_posix() != "manifest.yaml"
    }
    manifest_paths = {entry.path for entry in manifest.files}
    if actual_paths != manifest_paths:
        missing = sorted(manifest_paths - actual_paths)
        unexpected = sorted(actual_paths - manifest_paths)
        raise RecordingError(
            f"Manifest file set mismatch; missing={missing}, unexpected={unexpected}"
        )
    for entry in manifest.files:
        path = (root / entry.path).resolve()
        if not path.is_relative_to(root.resolve()):
            raise RecordingError(f"Manifest path escapes experiment directory: {entry.path}")
        digest, byte_count = hash_and_size(path)
        if byte_count != entry.byte_count:
            raise RecordingError(f"Manifest byte count mismatch: {entry.path}")
        if digest != entry.sha256:
            raise RecordingError(f"Manifest hash mismatch: {entry.path}")


def _manifest_entry(
    root: Path,
    path: Path,
    *,
    targets: dict[str, LogicalRecordTarget],
) -> ManifestEntry:
    relative_path = path.relative_to(root).as_posix()
    target = targets.get(relative_path)
    digest, byte_count = hash_and_size(path)
    return ManifestEntry(
        path=relative_path,
        sha256=digest,
        byte_count=byte_count,
        kind=_inferred_kind(relative_path) if target is None else target.kind,
        identity=relative_path if target is None else target.identity,
    )


def hash_and_size(path: Path) -> tuple[str, int]:
    digest = hashlib.sha256()
    byte_count = 0
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
            byte_count += len(chunk)
    return digest.hexdigest(), byte_count


def _inferred_kind(path: str) -> RecordFileKind:
    if path.startswith("assets/"):
        return RecordFileKind.ASSET
    if path.startswith("artifacts/"):
        return RecordFileKind.ARTIFACT
    return RecordFileKind.OTHER


__all__ = (
    "ExperimentFile",
    "LogicalRecordTarget",
    "ManifestEntry",
    "RecordDurability",
    "RecordFileKind",
    "RecordManifest",
    "atomic_write_bytes",
    "atomic_write_text",
    "build_manifest",
    "hash_and_size",
    "create_temporary_record_directory",
    "normalize_logical_path",
    "publish_record_directory",
    "remove_temporary_record_directory",
    "sync_directory",
    "sync_tree",
    "validate_logical_targets",
    "validate_manifest",
)
