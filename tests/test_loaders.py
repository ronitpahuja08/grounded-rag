"""Tests for loader registration and file dispatch."""

from collections.abc import Sequence
from pathlib import Path
from typing import Final

import pytest

from grounded_rag import (
    AmbiguousLoaderError,
    EmptyLoaderResultError,
    EmptyTextSourceError,
    LoaderRegistry,
    MarkdownLoader,
    PdfLoader,
    PdfLoadingError,
    SourceDocument,
    TextDecodingError,
    TextLoader,
    UnsupportedSourceError,
)


class SuffixLoader:
    """Small deterministic loader fake keyed by a file suffix."""

    def __init__(self, suffix: str, *, empty: bool = False) -> None:
        self.suffix = suffix
        self.empty = empty

    def supports(self, source: Path) -> bool:
        return source.suffix.lower() == self.suffix

    def load(self, source: Path) -> Sequence[SourceDocument]:
        if self.empty:
            return ()
        return (
            SourceDocument(
                id=f"fixture:{source.stem}",
                content="Synthetic loader content",
                source=source.as_posix(),
                media_type="text/plain",
            ),
        )


def test_registry_dispatches_to_single_matching_loader(tmp_path: Path) -> None:
    source = tmp_path / "notes.txt"
    source.write_text("fixture", encoding="utf-8")
    registry = LoaderRegistry((SuffixLoader(".md"), SuffixLoader(".txt")))

    documents = registry.load(source)

    assert registry.registered_count == 2
    assert documents[0].id == "fixture:notes"


def test_registry_rejects_unsupported_source(tmp_path: Path) -> None:
    source = tmp_path / "archive.bin"
    source.write_bytes(b"fixture")

    with pytest.raises(UnsupportedSourceError, match=r"archive\.bin"):
        LoaderRegistry((SuffixLoader(".txt"),)).load(source)


def test_registry_rejects_ambiguous_dispatch(tmp_path: Path) -> None:
    source = tmp_path / "notes.txt"
    source.write_text("fixture", encoding="utf-8")
    registry = LoaderRegistry((SuffixLoader(".txt"), SuffixLoader(".txt")))

    with pytest.raises(AmbiguousLoaderError, match="multiple loaders"):
        registry.load(source)


def test_registry_rejects_empty_loader_result(tmp_path: Path) -> None:
    source = tmp_path / "notes.txt"
    source.write_text("fixture", encoding="utf-8")

    with pytest.raises(EmptyLoaderResultError, match="returned no documents"):
        LoaderRegistry((SuffixLoader(".txt", empty=True),)).load(source)


def test_registry_requires_existing_file(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match=r"missing\.txt"):
        LoaderRegistry((SuffixLoader(".txt"),)).load(tmp_path / "missing.txt")


def test_registry_rejects_duplicate_loader_instance() -> None:
    loader = SuffixLoader(".txt")
    registry = LoaderRegistry((loader,))

    with pytest.raises(ValueError, match="already registered"):
        registry.register(loader)


def test_text_loader_reads_utf8_content_with_traceable_metadata(
    tmp_path: Path,
) -> None:
    source = tmp_path / "Café notes.txt"
    source.write_bytes("First line\r\nCafé evidence\r".encode())

    (document,) = LoaderRegistry((TextLoader(),)).load(source)

    assert document.content == "First line\nCafé evidence\n"
    assert document.media_type == "text/plain"
    assert document.title == "Café notes"
    assert document.source == source.resolve().as_posix()
    assert document.metadata == {
        "encoding": "utf-8",
        "location": source.resolve().as_posix(),
    }


def test_text_loader_identifier_is_stable_for_same_location(tmp_path: Path) -> None:
    source = tmp_path / "notes.txt"
    source.write_text("Version one", encoding="utf-8")
    loader = TextLoader()
    first_id = loader.load(source)[0].id

    source.write_text("Version two", encoding="utf-8")

    assert loader.load(source)[0].id == first_id


def test_text_loader_matches_txt_extension_case_insensitively() -> None:
    loader = TextLoader()

    assert loader.supports(Path("notes.txt"))
    assert loader.supports(Path("NOTES.TXT"))
    assert not loader.supports(Path("notes.md"))


def test_text_loader_rejects_non_utf8_bytes(tmp_path: Path) -> None:
    source = tmp_path / "legacy.txt"
    source.write_bytes(b"\xff\xfeinvalid")

    with pytest.raises(TextDecodingError, match="valid UTF-8"):
        TextLoader().load(source)


def test_text_loader_rejects_whitespace_only_files(tmp_path: Path) -> None:
    source = tmp_path / "empty.txt"
    source.write_text(" \n\t", encoding="utf-8")

    with pytest.raises(EmptyTextSourceError, match="no readable content"):
        TextLoader().load(source)


def test_markdown_loader_preserves_outline_and_source_metadata(
    tmp_path: Path,
) -> None:
    source = tmp_path / "guide.md"
    source.write_text(
        "# Grounded RAG\n\n"
        "## Loading\n\n"
        "```python\n# not a heading\n```\n\n"
        "Evaluation\n----------\n",
        encoding="utf-8",
    )

    (document,) = LoaderRegistry((MarkdownLoader(),)).load(source)

    assert document.media_type == "text/markdown"
    assert document.title == "Grounded RAG"
    assert document.source == source.resolve().as_posix()
    assert document.metadata["headings"] == [
        {"level": 1, "line": 1, "text": "Grounded RAG"},
        {"level": 2, "line": 3, "text": "Loading"},
        {"level": 2, "line": 9, "text": "Evaluation"},
    ]
    assert document.metadata["location"] == source.resolve().as_posix()


def test_markdown_loader_uses_setext_h1_as_title(tmp_path: Path) -> None:
    source = tmp_path / "design.markdown"
    source.write_text("Architecture\n============\n\nDetails\n", encoding="utf-8")

    (document,) = MarkdownLoader().load(source)

    assert document.title == "Architecture"
    assert document.metadata["headings"] == [
        {"level": 1, "line": 1, "text": "Architecture"}
    ]


def test_markdown_loader_falls_back_to_filename_without_h1(
    tmp_path: Path,
) -> None:
    source = tmp_path / "release-notes.md"
    source.write_text("## Version 0.1\n\nInitial release.\n", encoding="utf-8")

    (document,) = MarkdownLoader().load(source)

    assert document.title == "release-notes"


def test_markdown_loader_supports_common_extensions() -> None:
    loader = MarkdownLoader()

    assert loader.supports(Path("README.md"))
    assert loader.supports(Path("GUIDE.MARKDOWN"))
    assert not loader.supports(Path("notes.txt"))


def test_markdown_loader_rejects_invalid_utf8(tmp_path: Path) -> None:
    source = tmp_path / "legacy.md"
    source.write_bytes(b"\xff\xfeinvalid")

    with pytest.raises(TextDecodingError, match="Markdown source"):
        MarkdownLoader().load(source)


_PDF_HEADER: Final = b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n"


def _write_text_pdf(source: Path) -> None:
    """Write a minimal two-page PDF with deterministic text and metadata."""
    streams = (
        b"BT /F1 12 Tf 72 720 Td (First page evidence) Tj ET",
        b"BT /F1 12 Tf 72 720 Td (Second page findings) Tj ET",
    )
    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R 5 0 R] /Count 2 >>",
        (
            b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
            b"/Resources << /Font << /F1 4 0 R >> >> /Contents 6 0 R >>"
        ),
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
        (
            b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
            b"/Resources << /Font << /F1 4 0 R >> >> /Contents 7 0 R >>"
        ),
        b"<< /Length "
        + str(len(streams[0])).encode()
        + b" >>\nstream\n"
        + streams[0]
        + b"\nendstream",
        b"<< /Length "
        + str(len(streams[1])).encode()
        + b" >>\nstream\n"
        + streams[1]
        + b"\nendstream",
        b"<< /Title (Fixture Guide) >>",
    ]
    payload = bytearray(_PDF_HEADER)
    offsets = [0]
    for object_number, body in enumerate(objects, start=1):
        offsets.append(len(payload))
        payload.extend(f"{object_number} 0 obj\n".encode())
        payload.extend(body)
        payload.extend(b"\nendobj\n")

    xref_offset = len(payload)
    payload.extend(f"xref\n0 {len(objects) + 1}\n".encode())
    payload.extend(b"0000000000 65535 f \n")
    for offset in offsets[1:]:
        payload.extend(f"{offset:010d} 00000 n \n".encode())
    payload.extend(
        (
            f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R /Info 8 0 R >>\n"
            f"startxref\n{xref_offset}\n%%EOF\n"
        ).encode()
    )
    source.write_bytes(payload)


def test_pdf_loader_returns_page_documents_with_provenance(tmp_path: Path) -> None:
    source = tmp_path / "guide.pdf"
    _write_text_pdf(source)

    documents = LoaderRegistry((PdfLoader(),)).load(source)

    assert [document.content for document in documents] == [
        "First page evidence",
        "Second page findings",
    ]
    assert [document.id.rsplit(":", 1)[1] for document in documents] == [
        "page-1",
        "page-2",
    ]
    assert all(document.media_type == "application/pdf" for document in documents)
    assert all(document.title == "Fixture Guide" for document in documents)
    assert [document.metadata["page"] for document in documents] == [1, 2]
    assert all(document.metadata["page_count"] == 2 for document in documents)
    assert all(
        document.metadata["location"] == source.resolve().as_posix()
        for document in documents
    )


def test_pdf_loader_supports_extension_case_insensitively() -> None:
    loader = PdfLoader()

    assert loader.supports(Path("guide.pdf"))
    assert loader.supports(Path("GUIDE.PDF"))
    assert not loader.supports(Path("guide.md"))


def test_pdf_loader_rejects_invalid_pdf(tmp_path: Path) -> None:
    source = tmp_path / "broken.pdf"
    source.write_bytes(b"not a PDF")

    with pytest.raises(PdfLoadingError, match="unable to read PDF"):
        PdfLoader().load(source)
