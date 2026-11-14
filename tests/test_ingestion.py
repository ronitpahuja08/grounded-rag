"""End-to-end tests for incremental corpus ingestion."""

from datetime import UTC, datetime
from pathlib import Path

import pytest

from grounded_rag import (
    ChunkingConfig,
    LoaderRegistry,
    TextDecodingError,
    TextLoader,
    ingest_directory,
    load_manifest,
)


def test_incremental_ingestion_reuses_unchanged_sources(tmp_path: Path) -> None:
    sources = tmp_path / "sources"
    sources.mkdir()
    first = sources / "first.txt"
    second = sources / "second.txt"
    first.write_text("first source evidence", encoding="utf-8")
    second.write_text("second source evidence", encoding="utf-8")
    manifest_path = tmp_path / "state" / "manifest.json"
    registry = LoaderRegistry((TextLoader(),))
    config = ChunkingConfig(max_tokens=16, overlap_tokens=0)

    initial = ingest_directory(
        sources,
        registry,
        manifest_path,
        chunking=config,
        generated_at=datetime(2026, 11, 14, 8, 0, tzinfo=UTC),
    )
    repeated = ingest_directory(
        sources,
        registry,
        manifest_path,
        chunking=config,
        generated_at=datetime(2026, 11, 14, 9, 0, tzinfo=UTC),
    )

    assert len(initial.processed_chunks) == 2
    assert initial.changes.added == tuple(
        sorted((first.resolve().as_posix(), second.resolve().as_posix()))
    )
    assert repeated.processed_chunks == ()
    assert repeated.changes.unchanged == initial.changes.added
    assert not repeated.changes.requires_update
    assert load_manifest(manifest_path) == repeated.manifest


def test_incremental_ingestion_tracks_change_addition_and_removal(
    tmp_path: Path,
) -> None:
    sources = tmp_path / "sources"
    sources.mkdir()
    changed = sources / "changed.txt"
    removed = sources / "removed.txt"
    changed.write_text("version one", encoding="utf-8")
    removed.write_text("to be removed", encoding="utf-8")
    manifest_path = tmp_path / "manifest.json"
    registry = LoaderRegistry((TextLoader(),))
    config = ChunkingConfig(max_tokens=16, overlap_tokens=0)
    ingest_directory(sources, registry, manifest_path, chunking=config)

    changed.write_text("version two with new evidence", encoding="utf-8")
    removed.unlink()
    added = sources / "added.txt"
    added.write_text("new source", encoding="utf-8")
    result = ingest_directory(
        sources,
        registry,
        manifest_path,
        chunking=config,
    )

    assert result.changes.added == (added.resolve().as_posix(),)
    assert result.changes.changed == (changed.resolve().as_posix(),)
    assert result.changes.removed == (removed.resolve().as_posix(),)
    assert len(result.processed_chunks) == 2
    assert {entry.source for entry in result.manifest.sources} == {
        added.resolve().as_posix(),
        changed.resolve().as_posix(),
    }


def test_failed_incremental_run_does_not_replace_last_valid_manifest(
    tmp_path: Path,
) -> None:
    sources = tmp_path / "sources"
    sources.mkdir()
    source = sources / "guide.txt"
    source.write_text("valid evidence", encoding="utf-8")
    manifest_path = tmp_path / "manifest.json"
    registry = LoaderRegistry((TextLoader(),))
    config = ChunkingConfig(max_tokens=16, overlap_tokens=0)
    ingest_directory(sources, registry, manifest_path, chunking=config)
    valid_manifest = manifest_path.read_bytes()

    source.write_bytes(b"\xff\xfeinvalid")

    with pytest.raises(TextDecodingError):
        ingest_directory(sources, registry, manifest_path, chunking=config)
    assert manifest_path.read_bytes() == valid_manifest
