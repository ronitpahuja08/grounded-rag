"""Tests for offline deterministic embedding providers."""

import math

import pytest

from grounded_rag.embeddings import LocalHashEmbeddingProvider
from grounded_rag.protocols import EmbeddingProvider


def test_local_hash_embedder_is_deterministic_and_normalized() -> None:
    provider = LocalHashEmbeddingProvider(dimensions=16)

    first = provider.embed_query("Grounded retrieval needs evidence")
    second = provider.embed_documents(("Grounded retrieval needs evidence",))[0]

    assert isinstance(provider, EmbeddingProvider)
    assert first == second
    assert len(first) == 16
    assert math.isclose(sum(value * value for value in first), 1.0)


def test_local_hash_embedder_handles_empty_text_and_bad_dimensions() -> None:
    provider = LocalHashEmbeddingProvider(dimensions=8)

    assert provider.embed_query("") == (0.0,) * 8
    with pytest.raises(ValueError, match="at least 8"):
        LocalHashEmbeddingProvider(dimensions=7)
