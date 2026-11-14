"""Tests for deterministic document chunking."""

import pytest
from pydantic import ValidationError

from grounded_rag import ChunkingConfig, SourceDocument, chunk_document


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


def _document(content: str, *, document_id: str = "guide") -> SourceDocument:
    return SourceDocument(
        id=document_id,
        content=content,
        source="guide.txt",
        media_type="text/plain",
        metadata={"source_hash": "abc123"},
    )


def test_chunk_document_builds_overlapping_token_windows() -> None:
    content = " ".join(f"token-{index}" for index in range(40))
    chunks = chunk_document(
        _document(content),
        ChunkingConfig(max_tokens=16, overlap_tokens=4),
    )

    assert [chunk.metadata["token_count"] for chunk in chunks] == [16, 16, 16]
    assert chunks[0].content.split()[-4:] == chunks[1].content.split()[:4]
    assert chunks[1].content.split()[-4:] == chunks[2].content.split()[:4]
    assert chunks[0].start_char == 0
    assert chunks[-1].end_char == len(content)


def test_chunk_document_keeps_offsets_aligned_with_original_content() -> None:
    document = _document("  first\n\nsecond   third  ")

    (chunk,) = chunk_document(
        document,
        ChunkingConfig(max_tokens=16, overlap_tokens=0),
    )

    assert chunk.content == "first\n\nsecond   third"
    assert document.content[chunk.start_char : chunk.end_char] == chunk.content
    assert chunk.metadata["source_hash"] == "abc123"


def test_markdown_chunks_do_not_cross_heading_boundaries() -> None:
    first_body = " ".join(["alpha"] * 18)
    second_body = " ".join(["beta"] * 18)
    content = f"# First\n\n{first_body}\n\n## Second\n\n{second_body}\n"
    document = SourceDocument(
        id="guide",
        content=content,
        source="guide.md",
        media_type="text/markdown",
        metadata={
            "headings": [
                {"level": 1, "line": 1, "text": "First"},
                {"level": 2, "line": 5, "text": "Second"},
            ]
        },
    )

    chunks = chunk_document(
        document,
        ChunkingConfig(max_tokens=16, overlap_tokens=0),
    )

    assert len(chunks) == 4
    assert all(
        not ({"alpha", "beta"} <= set(chunk.content.split()))
        for chunk in chunks
    )
    assert [chunk.metadata["heading"] for chunk in chunks] == [
        "First",
        "First",
        "Second",
        "Second",
    ]
    assert [chunk.index for chunk in chunks] == [0, 1, 2, 3]


def test_pdf_chunks_retain_page_and_source_provenance() -> None:
    document = SourceDocument(
        id="pdf:guide:page-3",
        content="Page-specific evidence remains traceable.",
        source="documents/guide.pdf",
        media_type="application/pdf",
        title="Evidence Guide",
        metadata={"page": 3, "page_count": 8, "source_hash": "pdf-hash"},
    )

    (chunk,) = chunk_document(document)

    assert chunk.metadata == {
        "page": 3,
        "page_count": 8,
        "source_hash": "pdf-hash",
        "source": "documents/guide.pdf",
        "media_type": "application/pdf",
        "title": "Evidence Guide",
        "token_count": 4,
    }


def test_chunk_identifiers_are_deterministic_and_document_scoped() -> None:
    config = ChunkingConfig(max_tokens=16, overlap_tokens=0)
    original = _document("stable chunk content")
    same_content_elsewhere = _document(
        "stable chunk content",
        document_id="other-guide",
    )

    first = chunk_document(original, config)
    second = chunk_document(original, config)
    other = chunk_document(same_content_elsewhere, config)

    assert [chunk.id for chunk in first] == [chunk.id for chunk in second]
    assert first[0].id.startswith("chunk:")
    assert first[0].id != other[0].id
