"""Tests for semantic retrieval behavior."""

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
