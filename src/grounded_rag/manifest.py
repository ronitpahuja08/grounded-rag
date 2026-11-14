"""Versioned corpus manifests with atomic JSON persistence."""

import os
from pathlib import Path
from typing import Annotated, Literal, Self

from pydantic import (
    AwareDatetime,
    BaseModel,
    ConfigDict,
    Field,
    StringConstraints,
    model_validator,
)

from grounded_rag.models import DocumentId, SourceLocation

ContentHash = Annotated[
    str,
    StringConstraints(pattern=r"^[a-f0-9]{64}$"),
]


class ManifestSource(BaseModel):
    """The documents and chunks produced from one physical source."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    source: SourceLocation
    source_hash: ContentHash
    document_ids: tuple[DocumentId, ...] = Field(min_length=1)
    chunk_ids: tuple[DocumentId, ...] = Field(min_length=1)


class CorpusManifest(BaseModel):
    """A complete, versioned snapshot of ingested source identities."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[1] = 1
    generated_at: AwareDatetime
    sources: tuple[ManifestSource, ...] = ()

    @model_validator(mode="after")
    def validate_unique_sources(self) -> Self:
        """Reject ambiguous manifests with repeated source locations."""
        source_names = [entry.source for entry in self.sources]
        if len(source_names) != len(set(source_names)):
            msg = "manifest source locations must be unique"
            raise ValueError(msg)
        return self


def save_manifest(manifest: CorpusManifest, destination: str | Path) -> None:
    """Atomically replace a manifest without exposing partial JSON."""
    path = Path(destination)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        f"{manifest.model_dump_json(indent=2)}\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def load_manifest(source: str | Path) -> CorpusManifest:
    """Load and validate a persisted corpus manifest."""
    return CorpusManifest.model_validate_json(Path(source).read_text(encoding="utf-8"))
