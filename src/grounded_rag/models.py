"""Validated domain models shared across the Grounded RAG pipeline."""

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, JsonValue, StringConstraints

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


class SourceDocument(BaseModel):
    """A normalized, traceable source ready for chunking."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    id: DocumentId
    content: NonEmptyText
    source: SourceLocation
    media_type: MediaType
    title: Title | None = None
    metadata: dict[MetadataKey, JsonValue] = Field(default_factory=dict)
