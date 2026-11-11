"""Provider-independent interfaces for ingestion, embedding, and generation."""

from collections.abc import Sequence
from pathlib import Path
from typing import Protocol, runtime_checkable

from grounded_rag.models import GroundedAnswer, RetrievalHit, SourceDocument

EmbeddingVector = tuple[float, ...]


@runtime_checkable
class DocumentLoader(Protocol):
    """Load a supported source into one or more normalized documents."""

    def supports(self, source: Path) -> bool:
        """Return whether this loader can read the source."""
        ...

    def load(self, source: Path) -> Sequence[SourceDocument]:
        """Read and normalize the source without mutating it."""
        ...


@runtime_checkable
class EmbeddingProvider(Protocol):
    """Convert document and query text into compatible dense vectors."""

    @property
    def dimensions(self) -> int:
        """Return the fixed vector width produced by this provider."""
        ...

    def embed_documents(self, texts: Sequence[str]) -> Sequence[EmbeddingVector]:
        """Embed document text in input order."""
        ...

    def embed_query(self, text: str) -> EmbeddingVector:
        """Embed one retrieval query in the same vector space."""
        ...


@runtime_checkable
class AnswerGenerator(Protocol):
    """Generate a validated answer from ranked retrieval evidence."""

    def generate(
        self,
        question: str,
        evidence: Sequence[RetrievalHit],
    ) -> GroundedAnswer:
        """Return a cited answer or an insufficient-evidence fallback."""
        ...
