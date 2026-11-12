"""Cross-loader tests for actionable errors and preserved metadata."""

from pathlib import Path

import pytest
from pypdf import PdfWriter

from grounded_rag import (
    AmbiguousLoaderError,
    BinarySourceError,
    EmptyTextSourceError,
    LoaderRegistry,
    MarkdownLoader,
    PdfLoader,
    PdfLoadingError,
    SourceDocument,
    TextLoader,
    UnsupportedSourceError,
    hash_bytes,
    hash_text,
)


@pytest.mark.parametrize(
    ("loader", "filename", "payload", "media_type"),
    [
        (TextLoader(), "notes.txt", b"First\r\nSecond\r", "text/plain"),
        (
            MarkdownLoader(),
            "guide.md",
            b"# Guide\r\n\r\n## Setup\r\n",
            "text/markdown",
        ),
    ],
)
def test_textual_loaders_preserve_required_provenance(
    tmp_path: Path,
    loader: TextLoader | MarkdownLoader,
    filename: str,
    payload: bytes,
    media_type: str,
) -> None:
    source = tmp_path / filename
    source.write_bytes(payload)

    (document,) = loader.load(source)

    assert document.media_type == media_type
    assert document.source == source.resolve().as_posix()
    assert document.metadata["location"] == document.source
    assert document.metadata["source_hash"] == hash_bytes(payload)
    assert document.metadata["content_hash"] == hash_text(document.content)
    assert str(document.metadata["source_modified_at"]).endswith("+00:00")
    assert SourceDocument.model_validate_json(document.model_dump_json()) == document


def test_markdown_outline_survives_model_serialization(tmp_path: Path) -> None:
    source = tmp_path / "guide.md"
    source.write_text("# Guide\n\n## Setup\n", encoding="utf-8")

    (document,) = MarkdownLoader().load(source)
    restored = SourceDocument.model_validate_json(document.model_dump_json())

    assert restored.metadata["headings"] == [
        {"level": 1, "line": 1, "text": "Guide"},
        {"level": 2, "line": 3, "text": "Setup"},
    ]


def test_markdown_loader_rejects_binary_payload(tmp_path: Path) -> None:
    source = tmp_path / "binary.md"
    source.write_bytes(b"# Heading\n\x00binary")

    with pytest.raises(BinarySourceError, match=r"binary\.md"):
        MarkdownLoader().load(source)


def test_pdf_loader_rejects_blank_document(tmp_path: Path) -> None:
    source = tmp_path / "blank.pdf"
    writer = PdfWriter()
    writer.add_blank_page(width=612, height=792)
    with source.open("wb") as output:
        writer.write(output)

    with pytest.raises(EmptyTextSourceError, match="no extractable text"):
        PdfLoader().load(source)


def test_pdf_loader_rejects_encrypted_document(tmp_path: Path) -> None:
    source = tmp_path / "secret.pdf"
    writer = PdfWriter()
    writer.add_blank_page(width=612, height=792)
    writer.encrypt("password")
    with source.open("wb") as output:
        writer.write(output)

    with pytest.raises(PdfLoadingError, match="encrypted PDF"):
        PdfLoader().load(source)


def test_registry_distinguishes_unsupported_and_ambiguous_sources(
    tmp_path: Path,
) -> None:
    unsupported = tmp_path / "archive.zip"
    unsupported.write_bytes(b"PK fixture")
    notes = tmp_path / "notes.txt"
    notes.write_text("Evidence", encoding="utf-8")

    with pytest.raises(UnsupportedSourceError, match=r"archive\.zip"):
        LoaderRegistry((TextLoader(), MarkdownLoader())).load(unsupported)

    with pytest.raises(AmbiguousLoaderError, match=r"notes\.txt"):
        LoaderRegistry((TextLoader(), TextLoader())).load(notes)
