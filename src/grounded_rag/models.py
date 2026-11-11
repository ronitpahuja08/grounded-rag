"""Validated domain models shared across the Grounded RAG pipeline."""

from typing import Annotated, Literal, Self

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    JsonValue,
    StringConstraints,
    model_validator,
)

DocumentId = Annotated[
    str,
    StringConstraints(
        strip_whitespace=True,
        min_length=1,
        max_length=128,
        pattern=r"^[a-z0-9][a-z0-9._:-]*$",
    ),
]
MetadataKey = Annotated[
    str,
    StringConstraints(strip_whitespace=True, min_length=1, max_length=64),
]
MediaType = Literal["text/plain", "text/markdown", "application/pdf"]
NonEmptyText = Annotated[str, StringConstraints(pattern=r"(?s).*\S.*")]
SourceLocation = Annotated[
    str,
    StringConstraints(strip_whitespace=True, min_length=1, max_length=2048),
]
Title = Annotated[
    str,
    StringConstraints(strip_whitespace=True, min_length=1, max_length=256),
]
ChunkIndex = Annotated[int, Field(ge=0)]
CharacterOffset = Annotated[int, Field(ge=0)]
RetrievalRank = Annotated[int, Field(ge=1)]
RetrievalScore = Annotated[float, Field(ge=0.0, le=1.0, allow_inf_nan=False)]
CitationIndex = Annotated[int, Field(ge=1)]
RetrieverName = Annotated[
    str,
    StringConstraints(strip_whitespace=True, min_length=1, max_length=64),
]
AnswerStatus = Literal["grounded", "insufficient_evidence"]


class SourceDocument(BaseModel):
    """A normalized, traceable source ready for chunking."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    id: DocumentId
    content: NonEmptyText
    source: SourceLocation
    media_type: MediaType
    title: Title | None = None
    metadata: dict[MetadataKey, JsonValue] = Field(default_factory=dict)


class Chunk(BaseModel):
    """A stable, position-aware retrieval unit derived from a document."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    id: DocumentId
    document_id: DocumentId
    content: NonEmptyText
    index: ChunkIndex
    start_char: CharacterOffset
    end_char: CharacterOffset
    metadata: dict[MetadataKey, JsonValue] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_character_range(self) -> Self:
        """Require a non-empty, forward-only source character range."""
        if self.end_char <= self.start_char:
            msg = "end_char must be greater than start_char"
            raise ValueError(msg)
        return self


class RetrievalHit(BaseModel):
    """A ranked chunk returned by a named retrieval strategy."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    chunk: Chunk
    score: RetrievalScore
    rank: RetrievalRank
    retriever: RetrieverName = "vector"


class Citation(BaseModel):
    """A numbered answer reference that resolves to an exact source span."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    index: CitationIndex
    chunk_id: DocumentId
    document_id: DocumentId
    source: SourceLocation
    quote: NonEmptyText
    start_char: CharacterOffset
    end_char: CharacterOffset

    @model_validator(mode="after")
    def validate_character_range(self) -> Self:
        """Require a non-empty, forward-only cited source range."""
        if self.end_char <= self.start_char:
            msg = "end_char must be greater than start_char"
            raise ValueError(msg)
        return self


class GroundedAnswer(BaseModel):
    """A cited answer or an explicit insufficient-evidence fallback."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    answer: NonEmptyText
    status: AnswerStatus
    citations: tuple[Citation, ...] = ()

    @model_validator(mode="after")
    def validate_citation_contract(self) -> Self:
        """Keep grounded answers cited and fallback answers citation-free."""
        if self.status == "grounded" and not self.citations:
            msg = "grounded answers require at least one citation"
            raise ValueError(msg)
        if self.status == "insufficient_evidence" and self.citations:
            msg = "insufficient-evidence answers cannot claim citations"
            raise ValueError(msg)

        indexes = [citation.index for citation in self.citations]
        expected_indexes = list(range(1, len(self.citations) + 1))
        if indexes != expected_indexes:
            msg = "citation indexes must be unique and contiguous from 1"
            raise ValueError(msg)
        return self
