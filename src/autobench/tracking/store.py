from __future__ import annotations as _annotations

import hashlib
import json
import sqlite3
from pathlib import Path
from types import TracebackType

from pydantic import TypeAdapter

from .models import SerializedValue

_SNAPSHOT_ADAPTER = TypeAdapter(dict[str, SerializedValue])

_SCHEMA = """
CREATE TABLE IF NOT EXISTS content_blobs (
    blob_hash TEXT PRIMARY KEY,
    payload TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS asset_versions (
    asset_id TEXT NOT NULL,
    version TEXT NOT NULL,
    content_hash TEXT NOT NULL,
    blob_hash TEXT NOT NULL REFERENCES content_blobs(blob_hash),
    PRIMARY KEY (asset_id, version)
);
CREATE INDEX IF NOT EXISTS asset_versions_content_hash
    ON asset_versions(content_hash);
CREATE TABLE IF NOT EXISTS diff_blobs (
    blob_hash TEXT PRIMARY KEY,
    payload TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS asset_diffs (
    asset_id TEXT NOT NULL,
    version TEXT NOT NULL,
    parent_version TEXT NOT NULL,
    blob_hash TEXT NOT NULL REFERENCES diff_blobs(blob_hash),
    PRIMARY KEY (asset_id, version, parent_version)
);
"""


class AssetContentStore:
    def __init__(self, path: Path, *, read_only: bool = False) -> None:
        self.path = path
        if read_only:
            self._connection = sqlite3.connect(
                f"file:{path}?mode=ro",
                timeout=30,
                uri=True,
            )
            self._connection.execute("PRAGMA foreign_keys = ON")
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        created = not path.exists()
        self._connection = sqlite3.connect(path, timeout=30)
        self._connection.execute("PRAGMA foreign_keys = ON")
        self._connection.executescript(_SCHEMA)
        if created:
            path.chmod(0o600)

    def __enter__(self) -> AssetContentStore:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self._connection.close()

    def write_content(
        self,
        *,
        asset_id: str,
        version: str,
        content_hash: str,
        snapshot: dict[str, SerializedValue],
    ) -> None:
        normalized = _SNAPSHOT_ADAPTER.validate_python(snapshot)
        payload = json.dumps(normalized, sort_keys=True, separators=(",", ":"))
        blob_hash = hashlib.sha256(payload.encode()).hexdigest()
        with self._connection:
            self._connection.execute(
                "INSERT OR IGNORE INTO content_blobs(blob_hash, payload) VALUES (?, ?)",
                (blob_hash, payload),
            )
            self._connection.execute(
                """
                INSERT INTO asset_versions(asset_id, version, content_hash, blob_hash)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(asset_id, version) DO NOTHING
                """,
                (asset_id, version, content_hash, blob_hash),
            )
            stored = self._connection.execute(
                """
                SELECT content_hash, blob_hash
                FROM asset_versions
                WHERE asset_id = ? AND version = ?
                """,
                (asset_id, version),
            ).fetchone()
            if stored != (content_hash, blob_hash):
                raise ValueError(f"Conflicting Autobench asset content: {asset_id}@{version}")

    def write_diff(
        self,
        *,
        asset_id: str,
        version: str,
        parent_version: str,
        diff: str,
    ) -> None:
        blob_hash = hashlib.sha256(diff.encode()).hexdigest()
        with self._connection:
            self._connection.execute(
                "INSERT OR IGNORE INTO diff_blobs(blob_hash, payload) VALUES (?, ?)",
                (blob_hash, diff),
            )
            self._connection.execute(
                """
                INSERT INTO asset_diffs(asset_id, version, parent_version, blob_hash)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(asset_id, version, parent_version) DO NOTHING
                """,
                (asset_id, version, parent_version, blob_hash),
            )
            stored = self._connection.execute(
                """
                SELECT blob_hash
                FROM asset_diffs
                WHERE asset_id = ? AND version = ? AND parent_version = ?
                """,
                (asset_id, version, parent_version),
            ).fetchone()
            if stored != (blob_hash,):
                raise ValueError(
                    f"Conflicting Autobench asset diff: {asset_id}@{parent_version}..{version}"
                )

    def content(self, *, asset_id: str, version: str) -> dict[str, SerializedValue] | None:
        row = self._connection.execute(
            """
            SELECT content_blobs.payload
            FROM asset_versions
            JOIN content_blobs USING (blob_hash)
            WHERE asset_versions.asset_id = ? AND asset_versions.version = ?
            """,
            (asset_id, version),
        ).fetchone()
        if row is None:
            return None
        payload = row[0]
        if not isinstance(payload, str):
            raise ValueError(f"Invalid Autobench asset content: {asset_id}@{version}")
        return _SNAPSHOT_ADAPTER.validate_json(payload)

    def diff(self, *, asset_id: str, version: str, parent_version: str) -> str | None:
        row = self._connection.execute(
            """
            SELECT diff_blobs.payload
            FROM asset_diffs
            JOIN diff_blobs USING (blob_hash)
            WHERE asset_diffs.asset_id = ?
              AND asset_diffs.version = ?
              AND asset_diffs.parent_version = ?
            """,
            (asset_id, version, parent_version),
        ).fetchone()
        if row is None:
            return None
        payload = row[0]
        if not isinstance(payload, str):
            raise ValueError(
                f"Invalid Autobench asset diff: {asset_id}@{parent_version}..{version}"
            )
        return payload


def load_asset_content(
    path: Path,
    *,
    asset_id: str,
    version: str,
) -> dict[str, SerializedValue]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with AssetContentStore(path, read_only=True) as store:
        snapshot = store.content(asset_id=asset_id, version=version)
    if snapshot is None:
        raise KeyError(f"Unknown Autobench asset content: {asset_id}@{version}")
    return snapshot


def load_asset_diff(
    path: Path,
    *,
    asset_id: str,
    version: str,
    parent_version: str,
) -> str:
    if not path.is_file():
        raise FileNotFoundError(path)
    with AssetContentStore(path, read_only=True) as store:
        diff = store.diff(
            asset_id=asset_id,
            version=version,
            parent_version=parent_version,
        )
    if diff is None:
        raise KeyError(f"Unknown Autobench asset diff: {asset_id}@{parent_version}..{version}")
    return diff


__all__ = ("load_asset_content", "load_asset_diff")
