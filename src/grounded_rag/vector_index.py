"""A deterministic in-memory vector index for traceable retrieval."""

from dataclasses import dataclass
from typing import Self

from grounded_rag.models import Chunk, RetrievalHit
from grounded_rag.protocols import EmbeddingVector
from grounded_rag.vectors import cosine_similarity, validate_vector


@dataclass(frozen=True, slots=True)
class VectorEntry:
    """A chunk paired with its validated dense embedding."""

    chunk: Chunk
    vector: EmbeddingVector


class InMemoryVectorIndex:
    """Stores one fixed-width vector per chunk with deterministic ranking."""

    def __init__(self, dimensions: int) -> None:
        if dimensions < 1:
            msg = "dimensions must be positive"
            raise ValueError(msg)
        self._dimensions = dimensions
        self._entries: dict[str, VectorEntry] = {}

    @property
    def dimensions(self) -> int:
        """Return the required vector width."""
        return self._dimensions

    @property
    def size(self) -> int:
        """Return the number of unique indexed chunks."""
        return len(self._entries)

    def upsert(self, chunk: Chunk, vector: EmbeddingVector) -> None:
        """Insert or replace a chunk vector after validating its width."""
        self._entries[chunk.id] = VectorEntry(
            chunk=chunk,
            vector=validate_vector(vector, dimensions=self._dimensions),
        )

    def remove(self, chunk_id: str) -> bool:
        """Remove an entry and report whether one was present."""
        return self._entries.pop(chunk_id, None) is not None

    def entries(self) -> tuple[VectorEntry, ...]:
        """Return entries in stable identifier order for persistence."""
        return tuple(sorted(self._entries.values(), key=lambda entry: entry.chunk.id))

    def search(
        self,
        query: EmbeddingVector,
        *,
        limit: int = 5,
    ) -> tuple[RetrievalHit, ...]:
        """Return top matching chunks with scores normalized to the model contract."""
        if limit < 1:
            msg = "limit must be positive"
            raise ValueError(msg)
        query_vector = validate_vector(query, dimensions=self._dimensions, name="query")
        scored = tuple(
            (entry, (cosine_similarity(query_vector, entry.vector) + 1.0) / 2.0)
            for entry in self._entries.values()
        )
        ranked = sorted(scored, key=lambda item: (-item[1], item[0].chunk.id))[:limit]
        return tuple(
            RetrievalHit(chunk=entry.chunk, score=score, rank=index, retriever="vector")
            for index, (entry, score) in enumerate(ranked, start=1)
        )

    @classmethod
    def from_entries(cls, dimensions: int, entries: tuple[VectorEntry, ...]) -> Self:
        """Construct an index by replaying validated persisted entries."""
        index = cls(dimensions)
        for entry in entries:
            index.upsert(entry.chunk, entry.vector)
        return index
