from __future__ import annotations

import pytest

from autobench.protocol import (
    CaptureLevel,
    CapturePolicy,
    CaptureSession,
    ReferenceKind,
    ReferenceStore,
)
from autobench.tracking import AssetVersion


def test_binary_values_are_referenced_deduplicated_or_explicitly_omitted() -> None:
    store = ReferenceStore()
    session = CaptureSession(CapturePolicy.metadata(), store=store)

    first = session.capture(b"binary", semantic_type="message.image")
    second = session.capture(b"binary", semantic_type="message.image")

    assert first.reference is not None
    assert first.reference == second.reference
    assert first.reference.kind is ReferenceKind.ARTIFACT
    assert isinstance(first.value, dict)
    assert first.value["size_bytes"] == 6
    assert first.diagnostics[0].code == "binary_content_referenced"
    assert len(store.artifacts) == 1
    assert store.artifacts[0].content == b"binary"
    assert store.artifacts[0].size_bytes == 6

    hashed = CaptureSession(CapturePolicy.hashed()).capture(b"binary")
    assert hashed.level is CaptureLevel.HASH
    assert hashed.reference is None
    assert isinstance(hashed.value, dict)
    assert hashed.value["sha256"]

    disabled = CaptureSession(CapturePolicy(store_binary=False)).capture(b"binary")
    assert disabled.omitted is True
    assert disabled.reference is None
    assert disabled.diagnostics[0].code == "binary_content_omitted"

    oversized = CaptureSession(CapturePolicy(max_artifact_bytes=3)).capture(b"binary")
    assert oversized.omitted is True
    assert oversized.diagnostics[0].code == "artifact_too_large"


def test_default_binary_capture_uses_references_without_persisting_secrets() -> None:
    store = ReferenceStore()
    session = CaptureSession(store=store)

    message = session.capture(b"binary", semantic_type="message.image")
    secret = session.capture(
        b"secret",
        semantic_type="message.attachment",
        path="headers.x-api-key",
    )

    assert message.level is CaptureLevel.METADATA
    assert message.reference is not None
    assert message.diagnostics[0].details == {"size_bytes": 6}
    assert secret.omitted is True
    assert secret.reference is None
    assert secret.value == {"type": "bytes", "size_bytes": 6}
    assert secret.diagnostics[0].code == "secret_redacted"
    assert len(store.artifacts) == 1


def test_artifact_identity_includes_media_type_while_content_hash_stays_stable() -> None:
    store = ReferenceStore()

    binary = store.add_artifact(b"same", media_type="application/octet-stream")
    json = store.add_artifact(b"same", media_type="application/json")

    assert binary.id != json.id
    assert binary.media_type == "application/octet-stream"
    assert json.media_type == "application/json"
    assert len(store.artifacts) == 2
    assert store.artifacts[0].content_hash == store.artifacts[1].content_hash


def test_large_normalized_values_become_deduplicated_json_artifacts() -> None:
    store = ReferenceStore()
    policy = CapturePolicy.full(
        max_inline_bytes=8,
        max_artifact_bytes=1_000,
        max_string_length=100,
    )
    session = CaptureSession(policy, store=store)
    value = {"text": "a long captured value"}

    first = session.capture(value)
    second = session.capture(value)

    assert first.reference == second.reference
    assert first.reference is not None
    assert first.reference.media_type == "application/json"
    assert first.value == {"type": "dict", "length": 1}
    assert first.diagnostics[0].code == "capture_referenced"
    size_bytes = first.diagnostics[0].details["size_bytes"]
    assert isinstance(size_bytes, int)
    assert size_bytes > 8
    assert first.diagnostics[0].details["max_inline_bytes"] == 8
    assert len(store.artifacts) == 1

    too_large = CaptureSession(
        CapturePolicy.full(
            max_inline_bytes=4,
            max_artifact_bytes=8,
            max_string_length=100,
        )
    ).capture(value)
    assert too_large.omitted is True
    assert too_large.reference is None
    assert too_large.diagnostics[0].code == "artifact_too_large"
    assert too_large.diagnostics[0].details["max_artifact_bytes"] == 8


@pytest.mark.parametrize(
    ("semantic_type", "kind"),
    [
        ("prompt.system", ReferenceKind.PROMPT),
        ("tool.definition", ReferenceKind.TOOL),
        ("output_schema.model", ReferenceKind.OUTPUT_SCHEMA),
        ("config.runtime", ReferenceKind.ASSET),
        (None, ReferenceKind.ASSET),
    ],
)
def test_tracked_asset_versions_reuse_typed_references(
    semantic_type: str | None,
    kind: ReferenceKind,
) -> None:
    store = ReferenceStore()
    session = CaptureSession(CapturePolicy.full(), store=store)
    version = AssetVersion(
        asset_id="asset-1",
        version="v2",
        content_hash="a" * 64,
    )

    first = session.capture("ignored", semantic_type=semantic_type, asset_version=version)
    second = session.capture("ignored", semantic_type=semantic_type, asset_version=version)

    assert first.reference == second.reference
    assert first.reference is not None
    assert first.reference.kind is kind
    assert first.reference.id == "asset-1"
    assert first.reference.version == "v2"
    assert first.value == {"asset_id": "asset-1", "version": "v2"}
    assert len(store.assets) == 1


def test_reference_store_rejects_non_asset_kinds_and_supports_explicit_kind() -> None:
    store = ReferenceStore()
    version = AssetVersion(asset_id="asset", version="v1", content_hash="b" * 64)

    with pytest.raises(ValueError, match="asset reference kind"):
        store.add_asset(version, kind=ReferenceKind.MESSAGE)

    result = CaptureSession(CapturePolicy.full(), store=store).capture(
        "ignored",
        asset_version=version,
        reference_kind=ReferenceKind.PROMPT,
    )
    assert result.reference is not None
    assert result.reference.kind is ReferenceKind.PROMPT
