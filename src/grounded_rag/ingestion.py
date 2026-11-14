"""Incremental document ingestion and corpus-manifest orchestration."""

from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from grounded_rag.chunking import (
    ChunkingConfig,
    chunk_document,
    deduplicate_chunks,
)
from grounded_rag.discovery import DEFAULT_IGNORED_PATHS, discover_sources
from grounded_rag.hashing import hash_bytes
from grounded_rag.loaders import LoaderRegistry
from grounded_rag.manifest import (
    CorpusManifest,
    ManifestChanges,
    ManifestSource,
    diff_manifests,
    load_manifest,
    save_manifest,
)
from grounded_rag.models import Chunk


@dataclass(frozen=True, slots=True)
class IngestionResult:
    """The new corpus state and chunks rebuilt during this run."""

    manifest: CorpusManifest
    changes: ManifestChanges
    processed_chunks: tuple[Chunk, ...]


def ingest_directory(
    root: str | Path,
    registry: LoaderRegistry,
    manifest_path: str | Path,
    *,
    chunking: ChunkingConfig | None = None,
    ignored_paths: tuple[str, ...] = DEFAULT_IGNORED_PATHS,
    generated_at: datetime | None = None,
) -> IngestionResult:
    """Rebuild only added or changed sources and atomically save new state."""
    destination = Path(manifest_path)
    previous = load_manifest(destination) if destination.is_file() else None
    previous_by_source = {
        entry.source: entry for entry in (() if previous is None else previous.sources)
    }
    entries: list[ManifestSource] = []
    processed_chunks: list[Chunk] = []
    active_chunking = chunking or ChunkingConfig()

    for source_path in discover_sources(
        root,
        registry,
        ignored_paths=ignored_paths,
    ):
        source = source_path.as_posix()
        source_hash = hash_bytes(source_path.read_bytes())
        previous_entry = previous_by_source.get(source)
        if previous_entry is not None and previous_entry.source_hash == source_hash:
            entries.append(previous_entry)
            continue

        documents = registry.load(source_path)
        source_chunks = deduplicate_chunks(
            tuple(
                chunk
                for document in documents
                for chunk in chunk_document(document, active_chunking)
            )
        )
        entries.append(
            ManifestSource(
                source=source,
                source_hash=source_hash,
                document_ids=tuple(document.id for document in documents),
                chunk_ids=tuple(chunk.id for chunk in source_chunks),
            )
        )
        processed_chunks.extend(source_chunks)

    current = CorpusManifest(
        generated_at=generated_at or datetime.now(UTC),
        sources=tuple(sorted(entries, key=lambda entry: entry.source)),
    )
    changes = diff_manifests(previous, current)
    save_manifest(current, destination)
    return IngestionResult(
        manifest=current,
        changes=changes,
        processed_chunks=tuple(processed_chunks),
    )
