"""Embedding providers that keep the retrieval core independent of vendors."""

import hashlib
import math
import re
from collections.abc import Sequence

from grounded_rag.protocols import EmbeddingVector

_TOKEN = re.compile(r"[a-z0-9]+")


class LocalHashEmbeddingProvider:
    """Deterministic, offline token-hashing embedder for local development."""

    def __init__(self, dimensions: int = 256) -> None:
        if dimensions < 8:
            msg = "dimensions must be at least 8"
            raise ValueError(msg)
        self._dimensions = dimensions

    @property
    def dimensions(self) -> int:
        """Return the fixed vector width."""
        return self._dimensions

    def embed_documents(self, texts: Sequence[str]) -> Sequence[EmbeddingVector]:
        """Embed text in input order without network access."""
        return tuple(self._embed(text) for text in texts)

    def embed_query(self, text: str) -> EmbeddingVector:
        """Embed a query in the same vector space as documents."""
        return self._embed(text)

    def _embed(self, text: str) -> EmbeddingVector:
        values = [0.0] * self._dimensions
        for token in _TOKEN.findall(text.lower()):
            digest = hashlib.sha256(token.encode("utf-8")).digest()
            index = int.from_bytes(digest[:8], "big") % self._dimensions
            values[index] += 1.0 if digest[8] % 2 else -1.0
        magnitude = math.sqrt(sum(value * value for value in values))
        if magnitude == 0:
            return tuple(values)
        return tuple(value / magnitude for value in values)
