"""Tests for semantic retrieval behavior."""

import pytest

from grounded_rag.embeddings import LocalHashEmbeddingProvider
from grounded_rag.models import Chunk, RetrievalQuery
from grounded_rag.retrieval import SemanticRetriever
from grounded_rag.vector_index import InMemoryVectorIndex


def make_chunk(chunk_id: str, content: str, source: str) -> Chunk:
    return Chunk(
        id=chunk_id,
        document_id="guide",
        content=content,
        index=0,
        start_char=0,
        end_char=len(content),
        metadata={"source": source},
    )


def test_semantic_retriever_ranks_indexed_evidence_and_filters_source() -> None:
    provider = LocalHashEmbeddingProvider(dimensions=32)
    index = InMemoryVectorIndex(dimensions=32)
    first = make_chunk("chunk-1", "retrieval requires supporting evidence", "guide.md")
    second = make_chunk("chunk-2", "queue workers process background jobs", "queue.md")
    index.upsert(first, provider.embed_query(first.content))
    index.upsert(second, provider.embed_query(second.content))
    retriever = SemanticRetriever(index, provider)

    query = RetrievalQuery(text="supporting retrieval evidence", limit=2)
    result = retriever.retrieve(query)
    filtered = retriever.retrieve(
        RetrievalQuery(
            text="supporting retrieval evidence",
            limit=2,
            source="guide.md",
        )
    )

    assert result.hits[0].chunk.id == "chunk-1"
    assert [hit.chunk.id for hit in filtered.hits] == ["chunk-1"]


def test_semantic_retriever_rejects_lexical_queries_and_dimension_mismatches() -> None:
    provider = LocalHashEmbeddingProvider(dimensions=32)
    index = InMemoryVectorIndex(dimensions=32)
    retriever = SemanticRetriever(index, provider)

    with pytest.raises(ValueError, match="semantic or hybrid"):
        retriever.retrieve(RetrievalQuery(text="find evidence", mode="lexical"))
    with pytest.raises(ValueError, match="dimensions must match"):
        SemanticRetriever(InMemoryVectorIndex(dimensions=16), provider)


def test_source_filter_fills_the_requested_limit_and_re_ranks_hits() -> None:
    provider = LocalHashEmbeddingProvider(dimensions=32)
    index = InMemoryVectorIndex(dimensions=32)
    chunks = (
        make_chunk("unrelated", "retrieval evidence", "other.md"),
        make_chunk("guide-one", "background context", "guide.md"),
        make_chunk("guide-two", "source specific notes", "guide.md"),
    )
    for chunk in chunks:
        index.upsert(chunk, provider.embed_query(chunk.content))
    result = SemanticRetriever(index, provider).retrieve(
        RetrievalQuery(text="retrieval evidence", source="guide.md", limit=2)
    )

    assert {hit.chunk.id for hit in result.hits} == {"guide-one", "guide-two"}
    assert [hit.rank for hit in result.hits] == [1, 2]
