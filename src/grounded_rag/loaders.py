"""Document loader registration and deterministic file dispatch."""

from collections.abc import Iterable
from pathlib import Path

from grounded_rag.hashing import hash_text, normalize_text
from grounded_rag.models import SourceDocument
from grounded_rag.protocols import DocumentLoader


class LoaderDispatchError(ValueError):
    """Base error for failures that occur before document parsing."""


class UnsupportedSourceError(LoaderDispatchError):
    """Raised when no registered loader supports a source."""


class AmbiguousLoaderError(LoaderDispatchError):
    """Raised when multiple registered loaders claim the same source."""


class EmptyLoaderResultError(LoaderDispatchError):
    """Raised when a loader accepts a source but returns no documents."""


class TextDecodingError(ValueError):
    """Raised when a text source is not valid UTF-8."""


class EmptyTextSourceError(ValueError):
    """Raised when a text source contains no readable content."""


class TextLoader:
    """Load UTF-8 ``.txt`` files into one traceable source document."""

    def supports(self, source: Path) -> bool:
        """Return whether the source uses the plain-text file extension."""
        return source.suffix.lower() == ".txt"

    def load(self, source: Path) -> tuple[SourceDocument, ...]:
        """Decode a text file strictly as UTF-8 and preserve its location."""
        resolved_source = source.resolve()
        try:
            content = normalize_text(resolved_source.read_text(encoding="utf-8"))
        except UnicodeDecodeError as error:
            msg = f"text source is not valid UTF-8: {source}"
            raise TextDecodingError(msg) from error

        if not content.strip():
            msg = f"text source contains no readable content: {source}"
            raise EmptyTextSourceError(msg)

        location = resolved_source.as_posix()
        document_id = f"text:{hash_text(location)[:32]}"
        return (
            SourceDocument(
                id=document_id,
                content=content,
                source=location,
                media_type="text/plain",
                title=source.stem,
                metadata={"encoding": "utf-8", "location": location},
            ),
        )


class LoaderRegistry:
    """Resolve exactly one registered loader for each source file."""

    def __init__(self, loaders: Iterable[DocumentLoader] = ()) -> None:
        self._loaders = list(loaders)

    @property
    def registered_count(self) -> int:
        """Return the number of available loader implementations."""
        return len(self._loaders)

    def register(self, loader: DocumentLoader) -> None:
        """Append a loader to the registry's deterministic lookup order."""
        if any(existing is loader for existing in self._loaders):
            msg = "the loader instance is already registered"
            raise ValueError(msg)
        self._loaders.append(loader)

    def resolve(self, source: str | Path) -> DocumentLoader:
        """Return the sole loader that advertises support for the source."""
        path = Path(source)
        matches = [loader for loader in self._loaders if loader.supports(path)]
        if not matches:
            msg = f"no loader supports source: {path}"
            raise UnsupportedSourceError(msg)
        if len(matches) > 1:
            msg = f"multiple loaders support source: {path}"
            raise AmbiguousLoaderError(msg)
        return matches[0]

    def load(self, source: str | Path) -> tuple[SourceDocument, ...]:
        """Dispatch an existing file and return a non-empty document tuple."""
        path = Path(source)
        if not path.is_file():
            msg = f"source file does not exist: {path}"
            raise FileNotFoundError(msg)

        documents = tuple(self.resolve(path).load(path))
        if not documents:
            msg = f"loader returned no documents for source: {path}"
            raise EmptyLoaderResultError(msg)
        return documents
