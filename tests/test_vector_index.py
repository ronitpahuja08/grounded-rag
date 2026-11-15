"""Tests for deterministic vector indexing."""

from pathlib import Path

from grounded_rag.index_storage import load_vector_index, save_vector_index
from grounded_rag.models import Chunk
from grounded_rag.vector_index import InMemoryVectorIndex


def make_chunk(chunk_id: str) -> Chunk:
    return Chunk(
        id=chunk_id,
        document_id="document-1",
        content="Evidence content",
        index=0,
        start_char=0,
        end_char=16,
    )


def test_index_ranks_cosine_matches_and_replaces_existing_ids() -> None:
    index = InMemoryVectorIndex(dimensions=2)
    index.upsert(make_chunk("chunk-b"), (0.0, 1.0))
    index.upsert(make_chunk("chunk-a"), (1.0, 0.0))
    index.upsert(make_chunk("chunk-a"), (1.0, 0.0))

    hits = index.search((1.0, 0.0), limit=2)

    assert index.size == 2
    assert [hit.chunk.id for hit in hits] == ["chunk-a", "chunk-b"]
    assert [hit.rank for hit in hits] == [1, 2]
    assert hits[0].score == 1.0


def test_index_validates_search_and_removal_contracts() -> None:
    index = InMemoryVectorIndex(dimensions=2)

    assert not index.remove("missing")
    try:
        index.search((1.0,), limit=1)
    except ValueError as error:
        assert "expected 2" in str(error)
    else:
        raise AssertionError("mismatched query dimensions must fail")


def test_vector_index_round_trips_atomically_with_chunk_metadata(
    tmp_path: Path,
) -> None:
    index = InMemoryVectorIndex(dimensions=2)
    index.upsert(make_chunk("chunk-a"), (1.0, 0.0))
    destination = tmp_path / "index.json"

    save_vector_index(index, destination)
    loaded = load_vector_index(destination)

    assert loaded.dimensions == 2
    assert loaded.search((1.0, 0.0))[0].chunk.id == "chunk-a"
