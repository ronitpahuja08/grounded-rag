"""Tests for corpus manifest persistence and validation."""

from datetime import UTC, datetime
from pathlib import Path

import pytest
from pydantic import ValidationError

from grounded_rag import (
    CorpusManifest,
    ManifestSource,
    load_manifest,
    save_manifest,
)


def _source(name: str = "docs/guide.md") -> ManifestSource:
    return ManifestSource(
        source=name,
        source_hash="a" * 64,
        document_ids=("markdown:guide",),
        chunk_ids=("chunk:guide",),
    )


def test_manifest_round_trip_is_atomic_and_versioned(tmp_path: Path) -> None:
    destination = tmp_path / "state" / "manifest.json"
    manifest = CorpusManifest(
        generated_at=datetime(2026, 11, 14, 9, 0, tzinfo=UTC),
        sources=(_source(),),
    )

    save_manifest(manifest, destination)

    assert load_manifest(destination) == manifest
    assert '"schema_version": 1' in destination.read_text(encoding="utf-8")
    assert not (destination.parent / ".manifest.json.tmp").exists()


def test_manifest_rejects_duplicate_source_locations() -> None:
    with pytest.raises(ValidationError, match="source locations must be unique"):
        CorpusManifest(
            generated_at=datetime(2026, 11, 14, 9, 0, tzinfo=UTC),
            sources=(_source(), _source()),
        )


def test_manifest_rejects_malformed_hashes() -> None:
    with pytest.raises(ValidationError):
        ManifestSource(
            source="guide.md",
            source_hash="not-a-sha256",
            document_ids=("guide",),
            chunk_ids=("chunk:guide",),
        )
