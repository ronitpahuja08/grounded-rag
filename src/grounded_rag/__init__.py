"""Grounded RAG public package."""

from grounded_rag.hashing import hash_bytes, hash_parts, hash_text, normalize_text
from grounded_rag.loaders import (
    AmbiguousLoaderError,
    EmptyLoaderResultError,
    EmptyTextSourceError,
    LoaderDispatchError,
    LoaderRegistry,
    MarkdownLoader,
    PdfLoader,
    PdfLoadingError,
    TextDecodingError,
    TextLoader,
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
    "EmptyTextSourceError",
    "GroundedAnswer",
    "LoaderDispatchError",
    "LoaderRegistry",
    "MarkdownLoader",
    "PdfLoader",
    "PdfLoadingError",
    "RetrievalHit",
    "SourceDocument",
    "TextDecodingError",
    "TextLoader",
    "UnsupportedSourceError",
    "__version__",
    "hash_bytes",
    "hash_parts",
    "hash_text",
    "normalize_text",
]

__version__ = "0.1.0"
