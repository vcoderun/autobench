from __future__ import annotations as _annotations

import platform
import sys
from hashlib import sha256
from pathlib import Path

from pydantic import BaseModel


class EnvironmentMetadata(BaseModel):
    python_version: str
    platform: str
    cwd: str


class ResolvedFileHash(BaseModel):
    path: str
    sha256: str


def capture_environment(*, cwd: Path | None = None) -> EnvironmentMetadata:
    return EnvironmentMetadata(
        python_version=sys.version.split()[0],
        platform=platform.platform(),
        cwd=str(cwd or Path.cwd()),
    )


def hash_file(path: Path) -> ResolvedFileHash:
    digest = sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return ResolvedFileHash(path=str(path.resolve()), sha256=digest.hexdigest())


__all__ = (
    "EnvironmentMetadata",
    "ResolvedFileHash",
    "capture_environment",
    "hash_file",
)
