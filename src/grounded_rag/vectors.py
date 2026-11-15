"""Validated numeric vector primitives used by embeddings and indexes."""

import math
from collections.abc import Sequence

from grounded_rag.protocols import EmbeddingVector


def validate_vector(
    vector: Sequence[float],
    *,
    dimensions: int | None = None,
    name: str = "vector",
) -> EmbeddingVector:
    """Return an immutable finite vector with an optional fixed dimension."""
    values = tuple(float(value) for value in vector)
    if not values:
        msg = f"{name} must not be empty"
        raise ValueError(msg)
    if dimensions is not None and len(values) != dimensions:
        msg = f"{name} has {len(values)} dimensions; expected {dimensions}"
        raise ValueError(msg)
    if not all(math.isfinite(value) for value in values):
        msg = f"{name} must contain only finite values"
        raise ValueError(msg)
    return values
