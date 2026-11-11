"""Grounded RAG public package."""

from grounded_rag.models import (
    Chunk,
    Citation,
    GroundedAnswer,
    RetrievalHit,
    SourceDocument,
)

__all__ = [
    "Chunk",
    "Citation",
    "GroundedAnswer",
    "RetrievalHit",
    "SourceDocument",
    "__version__",
]

__version__ = "0.1.0"
