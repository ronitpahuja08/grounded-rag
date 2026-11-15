"""Offline tests for OpenAI-compatible embedding provider behavior."""

from collections.abc import Mapping
from typing import Any

import pytest

from grounded_rag.openai_embeddings import OpenAICompatibleEmbeddingProvider


def test_provider_sends_ordered_inputs_and_validates_vectors() -> None:
    requests: list[Mapping[str, Any]] = []

    def transport(
        url: str,
        headers: Mapping[str, str],
        payload: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        assert url.endswith("/embeddings")
        assert headers["Authorization"] == "Bearer secret"
        requests.append(payload)
        return {"data": [{"embedding": [1, 0]}, {"embedding": [0, 1]}]}

    provider = OpenAICompatibleEmbeddingProvider(
        api_key="secret",
        model="text-embedding-3-small",
        dimensions=2,
        transport=transport,
    )

    assert provider.embed_documents(("first", "second")) == ((1.0, 0.0), (0.0, 1.0))
    assert requests == [
        {"model": "text-embedding-3-small", "input": ["first", "second"]}
    ]


def test_provider_rejects_malformed_provider_responses() -> None:
    provider = OpenAICompatibleEmbeddingProvider(
        api_key="secret",
        model="model",
        dimensions=2,
        transport=lambda *_: {"data": [{"embedding": [1, 2, 3]}]},
    )

    with pytest.raises(ValueError, match="expected 2"):
        provider.embed_query("question")
