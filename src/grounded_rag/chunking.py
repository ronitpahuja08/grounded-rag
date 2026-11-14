"""Deterministic, provenance-preserving document chunking."""

from typing import Self

from pydantic import BaseModel, ConfigDict, Field, model_validator


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
