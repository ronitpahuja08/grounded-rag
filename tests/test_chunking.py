"""Tests for deterministic document chunking."""

import pytest
from pydantic import ValidationError

from grounded_rag import ChunkingConfig


def test_chunking_config_has_safe_defaults() -> None:
    config = ChunkingConfig()

    assert config.max_tokens == 256
    assert config.overlap_tokens == 32


@pytest.mark.parametrize(
    ("max_tokens", "overlap_tokens"),
    [(15, 0), (32, 32), (32, 40), (9000, 1), (128, -1)],
)
def test_chunking_config_rejects_invalid_windows(
    max_tokens: int,
    overlap_tokens: int,
) -> None:
    with pytest.raises(ValidationError):
        ChunkingConfig(max_tokens=max_tokens, overlap_tokens=overlap_tokens)
