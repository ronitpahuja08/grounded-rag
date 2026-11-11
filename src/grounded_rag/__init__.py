"""Grounded RAG public package."""

from grounded_rag.hashing import hash_bytes, hash_parts, hash_text, normalize_text
from grounded_rag.models import (
    Chunk,
    Citation,
    GroundedAnswer,
    RetrievalHit,
    SourceDocument,
)
from grounded_rag.settings import AppSettings

__all__ = [
    "AppSettings",
    "Chunk",
    "Citation",
    "GroundedAnswer",
    "RetrievalHit",
    "SourceDocument",
    "__version__",
    "hash_bytes",
    "hash_parts",
    "hash_text",
    "normalize_text",
]

__version__ = "0.1.0"
