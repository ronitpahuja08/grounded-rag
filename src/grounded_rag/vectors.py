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


def cosine_similarity(left: Sequence[float], right: Sequence[float]) -> float:
    """Return cosine similarity for non-zero vectors with matching widths."""
    left_values = validate_vector(left, name="left vector")
    right_values = validate_vector(
        right,
        dimensions=len(left_values),
        name="right vector",
    )
    left_norm = math.sqrt(sum(value * value for value in left_values))
    right_norm = math.sqrt(sum(value * value for value in right_values))
    if left_norm == 0 or right_norm == 0:
        msg = "cosine similarity is undefined for a zero vector"
        raise ValueError(msg)
    dot_product = sum(
        left_value * right_value
        for left_value, right_value in zip(left_values, right_values, strict=True)
    )
    return dot_product / (left_norm * right_norm)
