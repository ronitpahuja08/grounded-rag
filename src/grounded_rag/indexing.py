"""Incremental vector-index updates that avoid redundant embedding calls."""

from collections.abc import Sequence
from dataclasses import dataclass

from grounded_rag.models import Chunk
from grounded_rag.protocols import EmbeddingProvider
from grounded_rag.vector_index import InMemoryVectorIndex


@dataclass(frozen=True, slots=True)
class IndexUpdate:
    """A summary of changes applied to a vector index."""

    added: int
    retained: int
    removed: int


def update_index(
    index: InMemoryVectorIndex,
    chunks: Sequence[Chunk],
    provider: EmbeddingProvider,
) -> IndexUpdate:
    """Synchronize an index with chunks, embedding only unknown identities."""
    if index.dimensions != provider.dimensions:
        msg = "index and embedding provider dimensions must match"
        raise ValueError(msg)
    desired = {chunk.id: chunk for chunk in chunks}
    existing = {entry.chunk.id for entry in index.entries()}
    added_ids = tuple(sorted(set(desired) - existing))
    added_chunks = tuple(desired[chunk_id] for chunk_id in added_ids)
    added_vectors = provider.embed_documents(
        tuple(chunk.content for chunk in added_chunks)
    )
    for chunk, vector in zip(
        added_chunks,
        added_vectors,
        strict=True,
    ):
        index.upsert(chunk, vector)
    removed = 0
    for chunk_id in existing - set(desired):
        removed += int(index.remove(chunk_id))
    return IndexUpdate(
        added=len(added_ids),
        retained=len(existing & set(desired)),
        removed=removed,
    )
