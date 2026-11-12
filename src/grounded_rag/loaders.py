"""Document loader registration and deterministic file dispatch."""

from collections.abc import Iterable
from pathlib import Path

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
