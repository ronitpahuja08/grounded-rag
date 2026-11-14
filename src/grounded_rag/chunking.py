"""Deterministic, provenance-preserving document chunking."""

import re
from typing import Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from grounded_rag.models import Chunk, SourceDocument

_TOKEN = re.compile(r"\S+")


class ChunkingConfig(BaseModel):
    """Token-window settings used by every chunking strategy."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    max_tokens: int = Field(default=256, ge=16, le=8192)
    overlap_tokens: int = Field(default=32, ge=0, le=2048)

    @model_validator(mode="after")
    def validate_overlap(self) -> Self:
        """Require each window to advance by at least one token."""
        if self.overlap_tokens >= self.max_tokens:
            msg = "overlap_tokens must be smaller than max_tokens"
            raise ValueError(msg)
        return self


def chunk_document(
    document: SourceDocument,
    config: ChunkingConfig | None = None,
) -> tuple[Chunk, ...]:
    """Split a document into overlapping token windows with exact offsets."""
    active_config = config or ChunkingConfig()
    tokens = tuple(_TOKEN.finditer(document.content))
    if not tokens:
        return ()

    chunks: list[Chunk] = []
    step = active_config.max_tokens - active_config.overlap_tokens
    for chunk_index, token_start in enumerate(range(0, len(tokens), step)):
        window = tokens[token_start : token_start + active_config.max_tokens]
        if not window:
            break
        start_char = window[0].start()
        end_char = window[-1].end()
        chunks.append(
            Chunk(
                id=f"{document.id}:chunk-{chunk_index}",
                document_id=document.id,
                content=document.content[start_char:end_char],
                index=chunk_index,
                start_char=start_char,
                end_char=end_char,
                metadata={**document.metadata, "token_count": len(window)},
            )
        )
        if token_start + active_config.max_tokens >= len(tokens):
            break
    return tuple(chunks)
