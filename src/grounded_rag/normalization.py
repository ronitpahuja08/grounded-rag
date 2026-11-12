"""Stable text normalization shared by every document loader."""

from grounded_rag.hashing import normalize_text


def normalize_document_text(content: str) -> str:
    """Normalize source text while preserving meaningful indentation.

    The result uses NFC Unicode, LF line endings, no UTF-8 BOM, no trailing
    horizontal whitespace, and exactly one final newline when content exists.
    Leading and internal blank lines remain intact so source line locations do
    not shift and Markdown code indentation is not damaged.
    """
    normalized = normalize_text(content)
    if normalized.startswith("\ufeff"):
        normalized = normalized.removeprefix("\ufeff")

    lines = [line.rstrip(" \t") for line in normalized.split("\n")]
    while lines and not lines[-1]:
        lines.pop()
    if not lines:
        return ""
    return "\n".join(lines) + "\n"
