"""Tests for deterministic content hashing utilities."""

import pytest

from grounded_rag import hash_bytes, hash_parts, hash_text, normalize_text


def test_hash_bytes_matches_known_sha256_vector() -> None:
    assert hash_bytes(b"hello") == (
        "2cf24dba5fb0a30e26e83b2ac5b9e29e"
        "1b161e5c1fa7425e73043362938b9824"
    )


def test_hash_text_normalizes_line_endings() -> None:
    assert hash_text("first\r\nsecond\r") == hash_text("first\nsecond\n")


def test_hash_text_normalizes_equivalent_unicode() -> None:
    composed = "Caf\u00e9"
    decomposed = "Cafe\u0301"

    assert normalize_text(decomposed) == composed
    assert hash_text(composed) == hash_text(decomposed)


def test_hash_text_preserves_meaningful_whitespace() -> None:
    assert hash_text("evidence") != hash_text("evidence ")


def test_hash_parts_preserves_value_boundaries_and_order() -> None:
    assert hash_parts("ab", "c") != hash_parts("a", "bc")
    assert hash_parts("document", b"chunk") != hash_parts(b"chunk", "document")


def test_hash_parts_rejects_empty_input() -> None:
    with pytest.raises(ValueError, match="at least one"):
        hash_parts()
