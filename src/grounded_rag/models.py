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
RetrieverName = Annotated[
    str,
    StringConstraints(strip_whitespace=True, min_length=1, max_length=64),
]


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
