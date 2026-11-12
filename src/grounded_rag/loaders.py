"""Document loader registration and deterministic file dispatch."""

import re
from collections.abc import Iterable
from pathlib import Path
from typing import TypedDict, cast

from pydantic import JsonValue
from pypdf import PdfReader
from pypdf.errors import PdfReadError

from grounded_rag.hashing import hash_text
from grounded_rag.models import SourceDocument
from grounded_rag.normalization import normalize_document_text
from grounded_rag.protocols import DocumentLoader
from grounded_rag.provenance import source_provenance


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


class PdfLoadingError(ValueError):
    """Raised when a PDF cannot be opened or its text cannot be extracted."""


class TextLoader:
    """Load UTF-8 ``.txt`` files into one traceable source document."""

    def supports(self, source: Path) -> bool:
        """Return whether the source uses the plain-text file extension."""
        return source.suffix.lower() == ".txt"

    def load(self, source: Path) -> tuple[SourceDocument, ...]:
        """Decode a text file strictly as UTF-8 and preserve its location."""
        resolved_source = source.resolve()
        try:
            content = normalize_document_text(
                resolved_source.read_text(encoding="utf-8")
            )
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
                metadata={
                    **source_provenance(resolved_source, content),
                    "encoding": "utf-8",
                    "location": location,
                },
            ),
        )


_ATX_HEADING = re.compile(
    r"^ {0,3}(?P<marks>#{1,6})[ \t]+(?P<text>.*?)[ \t]*#*[ \t]*$"
)
_SETEXT_HEADING = re.compile(r"^ {0,3}(?P<marks>=+|-+)[ \t]*$")
_FENCE = re.compile(r"^ {0,3}(?P<marks>`{3,}|~{3,})")


class HeadingMetadata(TypedDict):
    """Serializable Markdown heading location."""

    level: int
    line: int
    text: str


class MarkdownLoader:
    """Load Markdown while retaining an ordered heading outline."""

    def supports(self, source: Path) -> bool:
        """Return whether the source has a common Markdown extension."""
        return source.suffix.lower() in {".md", ".markdown"}

    def load(self, source: Path) -> tuple[SourceDocument, ...]:
        """Decode Markdown as UTF-8 and record headings with line locations."""
        resolved_source = source.resolve()
        try:
            content = normalize_document_text(
                resolved_source.read_text(encoding="utf-8")
            )
        except UnicodeDecodeError as error:
            msg = f"Markdown source is not valid UTF-8: {source}"
            raise TextDecodingError(msg) from error

        if not content.strip():
            msg = f"Markdown source contains no readable content: {source}"
            raise EmptyTextSourceError(msg)

        headings = _extract_markdown_headings(content)
        heading_metadata = cast(JsonValue, headings)
        location = resolved_source.as_posix()
        document_id = f"markdown:{hash_text(location)[:32]}"
        title = next(
            (
                str(heading["text"])
                for heading in headings
                if heading["level"] == 1
            ),
            source.stem,
        )
        return (
            SourceDocument(
                id=document_id,
                content=content,
                source=location,
                media_type="text/markdown",
                title=title,
                metadata={
                    **source_provenance(resolved_source, content),
                    "encoding": "utf-8",
                    "headings": heading_metadata,
                    "location": location,
                },
            ),
        )


def _extract_markdown_headings(content: str) -> list[HeadingMetadata]:
    """Return ATX and Setext headings outside fenced code blocks."""
    headings: list[HeadingMetadata] = []
    lines = content.splitlines()
    fence_marker: str | None = None
    fence_length = 0

    for index, line in enumerate(lines):
        fence = _FENCE.match(line)
        if fence:
            marks = fence.group("marks")
            if fence_marker is None:
                fence_marker = marks[0]
                fence_length = len(marks)
            elif marks[0] == fence_marker and len(marks) >= fence_length:
                fence_marker = None
                fence_length = 0
            continue

        if fence_marker is not None:
            continue

        atx = _ATX_HEADING.match(line)
        if atx:
            text = atx.group("text").strip()
            if text:
                headings.append(
                    {
                        "level": len(atx.group("marks")),
                        "line": index + 1,
                        "text": text,
                    }
                )
            continue

        setext = _SETEXT_HEADING.match(line)
        if setext and index > 0:
            text = lines[index - 1].strip()
            if text:
                headings.append(
                    {
                        "level": 1 if setext.group("marks")[0] == "=" else 2,
                        "line": index,
                        "text": text,
                    }
                )

    return headings


class PdfLoader:
    """Load each readable PDF page as an independently traceable document."""

    def supports(self, source: Path) -> bool:
        """Return whether the source uses the PDF extension."""
        return source.suffix.lower() == ".pdf"

    def load(self, source: Path) -> tuple[SourceDocument, ...]:
        """Extract page text and retain one-based PDF page metadata."""
        resolved_source = source.resolve()
        try:
            reader = PdfReader(resolved_source)
        except (OSError, PdfReadError) as error:
            msg = f"unable to read PDF source: {source}"
            raise PdfLoadingError(msg) from error

        if reader.is_encrypted:
            msg = f"encrypted PDF sources are not supported: {source}"
            raise PdfLoadingError(msg)

        location = resolved_source.as_posix()
        source_id = hash_text(location)[:32]
        page_count = len(reader.pages)
        raw_title = reader.metadata.title if reader.metadata is not None else None
        title = raw_title.strip() if raw_title and raw_title.strip() else source.stem
        documents: list[SourceDocument] = []

        for page_number, page in enumerate(reader.pages, start=1):
            try:
                content = normalize_document_text(page.extract_text() or "")
            except (KeyError, PdfReadError, TypeError, ValueError) as error:
                msg = f"unable to extract page {page_number} from PDF source: {source}"
                raise PdfLoadingError(msg) from error

            if not content.strip():
                continue

            documents.append(
                SourceDocument(
                    id=f"pdf:{source_id}:page-{page_number}",
                    content=content,
                    source=location,
                    media_type="application/pdf",
                    title=title,
                    metadata={
                        **source_provenance(resolved_source, content),
                        "location": location,
                        "page": page_number,
                        "page_count": page_count,
                    },
                )
            )

        if not documents:
            msg = f"PDF source contains no extractable text: {source}"
            raise EmptyTextSourceError(msg)
        return tuple(documents)


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
