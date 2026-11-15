"""Tests for bounded, deterministic embedding batches."""

from collections.abc import Sequence

import pytest

from grounded_rag.embedding_batches import EmbeddingBatchPolicy, embed_batched
from grounded_rag.protocols import EmbeddingVector


class FlakyEmbedder:
    dimensions = 2

    def __init__(self) -> None:
        self.calls: list[tuple[str, ...]] = []
        self.fail_first = True

    def embed_documents(self, texts: Sequence[str]) -> Sequence[EmbeddingVector]:
        self.calls.append(tuple(texts))
        if self.fail_first:
            self.fail_first = False
            raise RuntimeError("temporary provider failure")
        return tuple((float(len(text)), 1.0) for text in texts)

    def embed_query(self, text: str) -> EmbeddingVector:
        return (float(len(text)), 1.0)


def test_batching_retries_failed_batches_without_reordering() -> None:
    provider = FlakyEmbedder()
    waits: list[float] = []

    vectors = embed_batched(
        provider,
        ("a", "bb", "ccc"),
        policy=EmbeddingBatchPolicy(
            batch_size=2,
            max_attempts=2,
            retry_delay_seconds=0.5,
        ),
        sleep=waits.append,
    )

    assert vectors == ((1.0, 1.0), (2.0, 1.0), (3.0, 1.0))
    assert provider.calls == [("a", "bb"), ("a", "bb"), ("ccc",)]
    assert waits == [0.5]


def test_batch_policy_rejects_invalid_limits() -> None:
    with pytest.raises(ValueError):
        EmbeddingBatchPolicy(batch_size=0)
    with pytest.raises(ValueError):
        EmbeddingBatchPolicy(retry_delay_seconds=-1)
