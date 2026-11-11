"""Tests for the core Grounded RAG domain contracts."""

import pytest
from pydantic import ValidationError

from grounded_rag import Chunk, Citation, GroundedAnswer, RetrievalHit, SourceDocument


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


def make_chunk() -> Chunk:
    """Create a representative chunk for retrieval model tests."""
    return Chunk(
        id="guide:introduction:0",
        document_id="guide:introduction",
        content="Grounded answers cite their evidence.",
        index=0,
        start_char=16,
        end_char=53,
        metadata={"heading": "Introduction"},
    )


def test_chunk_preserves_document_position() -> None:
    chunk = make_chunk()

    assert chunk.document_id == "guide:introduction"
    assert (chunk.start_char, chunk.end_char) == (16, 53)


@pytest.mark.parametrize(
    ("start_char", "end_char"),
    [(5, 5), (8, 3), (-1, 4)],
)
def test_chunk_rejects_invalid_character_ranges(
    start_char: int,
    end_char: int,
) -> None:
    with pytest.raises(ValidationError):
        Chunk(
            id="notes:0",
            document_id="notes",
            content="Text",
            index=0,
            start_char=start_char,
            end_char=end_char,
        )


def test_retrieval_hit_records_ranked_evidence() -> None:
    hit = RetrievalHit(chunk=make_chunk(), score=0.87, rank=1, retriever="hybrid")

    assert hit.chunk.id == "guide:introduction:0"
    assert hit.score == 0.87
    assert hit.rank == 1


@pytest.mark.parametrize("score", [-0.01, 1.01, float("inf"), float("nan")])
def test_retrieval_hit_rejects_invalid_scores(score: float) -> None:
    with pytest.raises(ValidationError):
        RetrievalHit(chunk=make_chunk(), score=score, rank=1)


def test_retrieval_hit_rejects_non_positive_rank() -> None:
    with pytest.raises(ValidationError):
        RetrievalHit(chunk=make_chunk(), score=0.5, rank=0)


def make_citation(index: int = 1) -> Citation:
    """Create a traceable citation for answer model tests."""
    return Citation(
        index=index,
        chunk_id="guide:introduction:0",
        document_id="guide:introduction",
        source="docs/introduction.md",
        quote="Grounded answers cite their evidence.",
        start_char=16,
        end_char=53,
    )


def test_grounded_answer_requires_ordered_citations() -> None:
    answer = GroundedAnswer(
        answer="Grounded answers point to supporting passages [1].",
        status="grounded",
        citations=(make_citation(),),
    )

    assert answer.citations[0].source == "docs/introduction.md"


def test_insufficient_evidence_answer_is_explicit_and_citation_free() -> None:
    answer = GroundedAnswer(
        answer="The available sources do not contain enough evidence.",
        status="insufficient_evidence",
    )

    assert answer.citations == ()


@pytest.mark.parametrize(
    ("status", "citations"),
    [
        ("grounded", ()),
        ("insufficient_evidence", (make_citation(),)),
    ],
)
def test_grounded_answer_rejects_status_citation_mismatches(
    status: str,
    citations: tuple[Citation, ...],
) -> None:
    with pytest.raises(ValidationError):
        GroundedAnswer.model_validate(
            {
                "answer": "A response",
                "status": status,
                "citations": citations,
            }
        )


def test_grounded_answer_rejects_non_contiguous_citation_indexes() -> None:
    with pytest.raises(ValidationError):
        GroundedAnswer(
            answer="A response with a malformed citation sequence.",
            status="grounded",
            citations=(make_citation(index=2),),
        )


def test_citation_rejects_invalid_source_range() -> None:
    with pytest.raises(ValidationError):
        Citation(
            index=1,
            chunk_id="notes:0",
            document_id="notes",
            source="notes.txt",
            quote="Text",
            start_char=4,
            end_char=4,
        )
