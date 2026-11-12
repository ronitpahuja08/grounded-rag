"""Grounded RAG public package."""

from grounded_rag.hashing import hash_bytes, hash_parts, hash_text, normalize_text
from grounded_rag.loaders import (
    AmbiguousLoaderError,
    EmptyLoaderResultError,
    LoaderDispatchError,
    LoaderRegistry,
    UnsupportedSourceError,
)
from grounded_rag.models import (
    Chunk,
    Citation,
    GroundedAnswer,
    RetrievalHit,
    SourceDocument,
)
from grounded_rag.protocols import (
    AnswerGenerator,
    DocumentLoader,
    EmbeddingProvider,
    EmbeddingVector,
)
from grounded_rag.settings import AppSettings

__all__ = [
    "AmbiguousLoaderError",
    "AnswerGenerator",
    "AppSettings",
    "Chunk",
    "Citation",
    "DocumentLoader",
    "EmbeddingProvider",
    "EmbeddingVector",
    "EmptyLoaderResultError",
    "GroundedAnswer",
    "LoaderDispatchError",
    "LoaderRegistry",
    "RetrievalHit",
    "SourceDocument",
    "UnsupportedSourceError",
    "__version__",
    "hash_bytes",
    "hash_parts",
    "hash_text",
    "normalize_text",
]

__version__ = "0.1.0"
