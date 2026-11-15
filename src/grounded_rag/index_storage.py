"""Atomic persistence for local vector indexes."""

import os
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from grounded_rag.models import Chunk
from grounded_rag.vector_index import InMemoryVectorIndex, VectorEntry
from grounded_rag.vectors import validate_vector


class StoredVectorEntry(BaseModel):
    """Serializable representation of one indexed chunk."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    chunk: Chunk
    vector: tuple[float, ...] = Field(min_length=1)


class StoredVectorIndex(BaseModel):
    """Versioned on-disk vector index payload."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[1] = 1
    dimensions: int = Field(ge=1)
    entries: tuple[StoredVectorEntry, ...] = ()


def save_vector_index(index: InMemoryVectorIndex, destination: str | Path) -> None:
    """Atomically persist vectors and their chunk metadata."""
    payload = StoredVectorIndex(
        dimensions=index.dimensions,
        entries=tuple(
            StoredVectorEntry(chunk=entry.chunk, vector=entry.vector)
            for entry in index.entries()
        ),
    )
    path = Path(destination)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(f"{payload.model_dump_json(indent=2)}\n", encoding="utf-8")
    os.replace(temporary, path)


def load_vector_index(source: str | Path) -> InMemoryVectorIndex:
    """Load and validate a saved index before exposing it to retrieval."""
    contents = Path(source).read_text(encoding="utf-8")
    payload = StoredVectorIndex.model_validate_json(contents)
    entries = tuple(
        VectorEntry(
            chunk=entry.chunk,
            vector=validate_vector(entry.vector, dimensions=payload.dimensions),
        )
        for entry in payload.entries
    )
    return InMemoryVectorIndex.from_entries(payload.dimensions, entries)
