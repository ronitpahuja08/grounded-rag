"""Tests for loader-safe source text normalization."""

from grounded_rag import normalize_document_text


def test_normalizes_unicode_bom_line_endings_and_trailing_space() -> None:
    content = "\ufeffCafe\u0301 \t\r\n \t\rSecond line\t\r\n\r\n"

    assert normalize_document_text(content) == "Café\n\nSecond line\n"


def test_preserves_leading_lines_and_indentation() -> None:
    content = "\n\tindented code  \n    nested code\t\n"

    assert normalize_document_text(content) == (
        "\n\tindented code\n    nested code\n"
    )


def test_empty_and_whitespace_only_content_normalize_to_empty() -> None:
    assert normalize_document_text("") == ""
    assert normalize_document_text(" \t\r\n\t") == ""


def test_normalization_is_idempotent() -> None:
    normalized = normalize_document_text("Evidence  \r\n\r\n")

    assert normalize_document_text(normalized) == normalized
