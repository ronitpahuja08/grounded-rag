"""Deterministic, provenance-preserving document chunking."""

import re
from collections.abc import Mapping
from typing import Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from grounded_rag.hashing import hash_parts, hash_text
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
    chunks: list[Chunk] = []
    for section_start, section_end, section_metadata in _section_ranges(document):
        chunks.extend(
            _chunk_range(
                document,
                active_config,
                section_start,
                section_end,
                start_index=len(chunks),
                section_metadata=section_metadata,
            )
        )
    return tuple(chunks)


def deduplicate_chunks(chunks: tuple[Chunk, ...]) -> tuple[Chunk, ...]:
    """Keep the first chunk for each normalized content digest."""
    unique: list[Chunk] = []
    seen_content: set[str] = set()
    for chunk in chunks:
        content_hash = hash_text(chunk.content)
        if content_hash in seen_content:
            continue
        seen_content.add(content_hash)
        unique.append(chunk)
    return tuple(unique)


def _chunk_range(
    document: SourceDocument,
    config: ChunkingConfig,
    start: int,
    end: int,
    *,
    start_index: int,
    section_metadata: Mapping[str, str | int],
) -> list[Chunk]:
    tokens = tuple(_TOKEN.finditer(document.content, start, end))
    if not tokens:
        return []

    chunks: list[Chunk] = []
    step = config.max_tokens - config.overlap_tokens
    for local_index, token_start in enumerate(range(0, len(tokens), step)):
        window = tokens[token_start : token_start + config.max_tokens]
        if not window:
            break
        start_char = window[0].start()
        end_char = window[-1].end()
        chunk_index = start_index + local_index
        content = document.content[start_char:end_char]
        chunk_id = hash_parts(
            document.id,
            str(start_char),
            str(end_char),
            content,
        )
        chunks.append(
            Chunk(
                id=f"chunk:{chunk_id[:40]}",
                document_id=document.id,
                content=content,
                index=chunk_index,
                start_char=start_char,
                end_char=end_char,
                metadata={
                    **document.metadata,
                    "source": document.source,
                    "media_type": document.media_type,
                    **({"title": document.title} if document.title else {}),
                    **section_metadata,
                    "token_count": len(window),
                },
            )
        )
        if token_start + config.max_tokens >= len(tokens):
            break
    return chunks


def _section_ranges(
    document: SourceDocument,
) -> tuple[tuple[int, int, Mapping[str, str | int]], ...]:
    """Return Markdown section spans or one span for other media types."""
    if document.media_type != "text/markdown":
        return ((0, len(document.content), {}),)

    raw_headings = document.metadata.get("headings")
    if not isinstance(raw_headings, list):
        return ((0, len(document.content), {}),)

    lines = document.content.splitlines(keepends=True)
    line_starts: list[int] = []
    offset = 0
    for line in lines:
        line_starts.append(offset)
        offset += len(line)

    headings: list[tuple[int, int, str]] = []
    for raw_heading in raw_headings:
        if not isinstance(raw_heading, dict):
            continue
        heading_line = raw_heading.get("line")
        level = raw_heading.get("level")
        text = raw_heading.get("text")
        if (
            isinstance(heading_line, int)
            and 1 <= heading_line <= len(line_starts)
            and isinstance(level, int)
            and isinstance(text, str)
        ):
            headings.append((line_starts[heading_line - 1], level, text))

    if not headings:
        return ((0, len(document.content), {}),)

    ranges: list[tuple[int, int, Mapping[str, str | int]]] = []
    first_start = headings[0][0]
    if document.content[:first_start].strip():
        ranges.append((0, first_start, {}))
    for index, (start, level, text) in enumerate(headings):
        end = (
            headings[index + 1][0]
            if index + 1 < len(headings)
            else len(document.content)
        )
        ranges.append((start, end, {"heading": text, "heading_level": level}))
    return tuple(ranges)
