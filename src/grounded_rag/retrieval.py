"""Provider-independent semantic retrieval over the local vector index."""

from grounded_rag.models import RetrievalQuery, RetrievalResult
from grounded_rag.protocols import EmbeddingProvider
from grounded_rag.vector_index import InMemoryVectorIndex


class SemanticRetriever:
    """Retrieves top-ranked evidence using a compatible embedding provider."""

    def __init__(
        self,
        index: InMemoryVectorIndex,
        provider: EmbeddingProvider,
    ) -> None:
        if index.dimensions != provider.dimensions:
            msg = "index and embedding provider dimensions must match"
            raise ValueError(msg)
        self._index = index
        self._provider = provider

    def retrieve(self, query: RetrievalQuery) -> RetrievalResult:
        """Embed a query and return ranked evidence, optionally from one source."""
        if query.mode not in {"semantic", "hybrid"}:
            msg = "semantic retriever only supports semantic or hybrid query modes"
            raise ValueError(msg)
        query_vector = self._provider.embed_query(query.text)
        hits = self._index.search(query_vector, limit=query.limit)
        if query.source is not None:
            hits = tuple(
                hit
                for hit in hits
                if hit.chunk.metadata.get("source") == query.source
            )
        return RetrievalResult(query=query, hits=hits)
