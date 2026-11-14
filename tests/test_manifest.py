"""Tests for corpus manifest persistence and validation."""

from datetime import UTC, datetime
from pathlib import Path

import pytest
from pydantic import ValidationError

from grounded_rag import (
    CorpusManifest,
    ManifestChanges,
    ManifestSource,
    diff_manifests,
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


def test_manifest_diff_detects_added_changed_removed_and_unchanged() -> None:
    generated_at = datetime(2026, 11, 14, 9, 0, tzinfo=UTC)
    previous = CorpusManifest(
        generated_at=generated_at,
        sources=(
            _source("changed.md"),
            _source("removed.md"),
            _source("stable.md"),
        ),
    )
    current = CorpusManifest(
        generated_at=generated_at,
        sources=(
            _source("added.md"),
            _source("changed.md").model_copy(
                update={"source_hash": "b" * 64}
            ),
            _source("stable.md"),
        ),
    )

    changes = diff_manifests(previous, current)

    assert changes == ManifestChanges(
        added=("added.md",),
        changed=("changed.md",),
        removed=("removed.md",),
        unchanged=("stable.md",),
    )
    assert changes.requires_update


def test_first_manifest_marks_every_source_as_added() -> None:
    current = CorpusManifest(
        generated_at=datetime(2026, 11, 14, 9, 0, tzinfo=UTC),
        sources=(_source("b.md"), _source("a.md")),
    )

    assert diff_manifests(None, current).added == ("a.md", "b.md")
