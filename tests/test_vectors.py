"""Tests for vector validation."""

import math

import pytest

from grounded_rag.vectors import cosine_similarity, validate_vector


def test_validate_vector_returns_immutable_finite_values() -> None:
    vector = validate_vector([1, 2.5, -3], dimensions=3)

    assert vector == (1.0, 2.5, -3.0)


@pytest.mark.parametrize("value", [[], [math.inf], [math.nan]])
def test_validate_vector_rejects_empty_or_non_finite_values(value: list[float]) -> None:
    with pytest.raises(ValueError):
        validate_vector(value)


def test_validate_vector_rejects_dimension_mismatch() -> None:
    with pytest.raises(ValueError, match="expected 2"):
        validate_vector((1, 2, 3), dimensions=2, name="embedding")


def test_cosine_similarity_preserves_expected_geometry() -> None:
    assert cosine_similarity((1, 0), (2, 0)) == 1.0
    assert cosine_similarity((1, 0), (0, 1)) == 0.0
    assert cosine_similarity((1, 0), (-1, 0)) == -1.0


def test_cosine_similarity_rejects_zero_or_mismatched_vectors() -> None:
    with pytest.raises(ValueError, match="zero"):
        cosine_similarity((0, 0), (1, 0))
    with pytest.raises(ValueError, match="expected 2"):
        cosine_similarity((1, 0), (1, 0, 0))
