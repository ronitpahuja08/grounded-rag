"""Grounded RAG public package."""

from grounded_rag.chunking import (
    ChunkingConfig,
    chunk_document,
    deduplicate_chunks,
)
from grounded_rag.discovery import (
    DEFAULT_IGNORED_PATHS,
    discover_sources,
    load_directory,
)
from grounded_rag.hashing import hash_bytes, hash_parts, hash_text, normalize_text
from grounded_rag.loaders import (
    AmbiguousLoaderError,
    BinarySourceError,
    EmptyLoaderResultError,
    EmptyTextSourceError,
    LoaderDispatchError,
    LoaderRegistry,
    MarkdownLoader,
    PdfLoader,
    PdfLoadingError,
    SourceTooLargeError,
    TextDecodingError,
    TextLoader,
    UnsupportedSourceError,
)
from grounded_rag.manifest import (
    CorpusManifest,
    ManifestChanges,
    ManifestSource,
    diff_manifests,
    load_manifest,
    save_manifest,
)
from grounded_rag.models import (
    Chunk,
    Citation,
    GroundedAnswer,
    RetrievalHit,
    SourceDocument,
)
from grounded_rag.normalization import normalize_document_text
from grounded_rag.protocols import (
    AnswerGenerator,
    DocumentLoader,
    EmbeddingProvider,
    EmbeddingVector,
)
from grounded_rag.provenance import source_provenance
from grounded_rag.settings import AppSettings

__all__ = [
    "DEFAULT_IGNORED_PATHS",
    "AmbiguousLoaderError",
    "AnswerGenerator",
    "AppSettings",
    "BinarySourceError",
    "Chunk",
    "ChunkingConfig",
    "Citation",
    "CorpusManifest",
    "DocumentLoader",
    "EmbeddingProvider",
    "EmbeddingVector",
    "EmptyLoaderResultError",
    "EmptyTextSourceError",
    "GroundedAnswer",
    "LoaderDispatchError",
    "LoaderRegistry",
    "ManifestChanges",
    "ManifestSource",
    "MarkdownLoader",
    "PdfLoader",
    "PdfLoadingError",
    "RetrievalHit",
    "SourceDocument",
    "SourceTooLargeError",
    "TextDecodingError",
    "TextLoader",
    "UnsupportedSourceError",
    "__version__",
    "chunk_document",
    "deduplicate_chunks",
    "diff_manifests",
    "discover_sources",
    "hash_bytes",
    "hash_parts",
    "hash_text",
    "load_directory",
    "load_manifest",
    "normalize_document_text",
    "normalize_text",
    "save_manifest",
    "source_provenance",
]

__version__ = "0.1.0"
