"""Tests for the core Grounded RAG domain contracts."""

import pytest
from pydantic import ValidationError

from grounded_rag import SourceDocument


def test_source_document_preserves_traceable_content() -> None:
    document = SourceDocument(
        id="guide:introduction",
        content="# Introduction\n\nGrounded answers cite their evidence.",
        source="docs/introduction.md",
        media_type="text/markdown",
        title=" Introduction ",
        metadata={"heading": "Introduction", "line": 1},
    )

    assert document.title == "Introduction"
    assert document.metadata == {"heading": "Introduction", "line": 1}


@pytest.mark.parametrize(
    "document_id",
    ["", "UPPERCASE", "-leading-symbol", "contains space", "path/segment"],
)
def test_source_document_rejects_invalid_identifiers(document_id: str) -> None:
    with pytest.raises(ValidationError):
        SourceDocument(
            id=document_id,
            content="Valid content",
            source="notes.txt",
            media_type="text/plain",
        )


def test_source_document_rejects_blank_content_and_unknown_fields() -> None:
    with pytest.raises(ValidationError):
        SourceDocument.model_validate(
            {
                "id": "notes",
                "content": "  \n\t",
                "source": "notes.txt",
                "media_type": "text/plain",
                "unexpected": True,
            }
        )


def test_source_document_is_immutable() -> None:
    document = SourceDocument(
        id="notes",
        content="Stable content",
        source="notes.txt",
        media_type="text/plain",
    )

    field_name = "id"
    with pytest.raises(ValidationError):
        setattr(document, field_name, "changed")
