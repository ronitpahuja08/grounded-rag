"""Bounded, retryable embedding batches with injected waiting."""

from collections.abc import Callable, Sequence
from dataclasses import dataclass

from grounded_rag.protocols import EmbeddingProvider, EmbeddingVector


@dataclass(frozen=True, slots=True)
class EmbeddingBatchPolicy:
    """Limits request size and retry behavior for embedding providers."""

    batch_size: int = 32
    max_attempts: int = 3
    retry_delay_seconds: float = 0.25

    def __post_init__(self) -> None:
        if self.batch_size < 1 or self.max_attempts < 1:
            msg = "batch_size and max_attempts must be positive"
            raise ValueError(msg)
        if self.retry_delay_seconds < 0:
            msg = "retry_delay_seconds must not be negative"
            raise ValueError(msg)


def embed_batched(
    provider: EmbeddingProvider,
    texts: Sequence[str],
    *,
    policy: EmbeddingBatchPolicy | None = None,
    sleep: Callable[[float], None] = lambda _: None,
) -> tuple[EmbeddingVector, ...]:
    """Embed texts in bounded batches, retrying a failed batch only."""
    active_policy = policy or EmbeddingBatchPolicy()
    vectors: list[EmbeddingVector] = []
    for start in range(0, len(texts), active_policy.batch_size):
        batch = texts[start : start + active_policy.batch_size]
        vectors.extend(_embed_with_retry(provider, batch, active_policy, sleep))
    return tuple(vectors)


def _embed_with_retry(
    provider: EmbeddingProvider,
    texts: Sequence[str],
    policy: EmbeddingBatchPolicy,
    sleep: Callable[[float], None],
) -> Sequence[EmbeddingVector]:
    for attempt in range(1, policy.max_attempts + 1):
        try:
            vectors = provider.embed_documents(texts)
            if len(vectors) != len(texts):
                msg = "embedding provider returned an unexpected vector count"
                raise ValueError(msg)
            return vectors
        except Exception:
            if attempt == policy.max_attempts:
                raise
            sleep(policy.retry_delay_seconds * attempt)
    raise AssertionError("unreachable retry state")
