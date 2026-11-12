"""Command-line interface for inspecting Grounded RAG source documents."""

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any, cast

from grounded_rag.discovery import load_directory
from grounded_rag.loaders import LoaderRegistry, MarkdownLoader, PdfLoader, TextLoader
from grounded_rag.models import SourceDocument
from grounded_rag.settings import AppSettings


def build_parser() -> argparse.ArgumentParser:
    """Build the stable public command-line parser."""
    parser = argparse.ArgumentParser(
        prog="grounded-rag",
        description="Inspect normalized RAG source documents and provenance.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    inspect_parser = subparsers.add_parser(
        "inspect",
        help="load a file or directory and print its document metadata",
    )
    inspect_parser.add_argument("source", type=Path)
    inspect_parser.add_argument(
        "--ignore",
        action="append",
        default=[],
        metavar="PATTERN",
        help="additional directory path pattern to ignore; repeat as needed",
    )
    inspect_parser.add_argument(
        "--max-source-bytes",
        type=_positive_integer,
        default=None,
        metavar="BYTES",
        help="override the configured maximum source file size",
    )
    inspect_parser.add_argument(
        "--include-content",
        action="store_true",
        help="include normalized document content in the JSON output",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the requested command and return a process exit status."""
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command != "inspect":  # pragma: no cover - argparse enforces this
        parser.error(f"unknown command: {args.command}")

    try:
        settings = AppSettings()
        configured_limit = cast(int | None, args.max_source_bytes)
        max_source_bytes = configured_limit or settings.max_source_bytes
        registry = LoaderRegistry(
            (TextLoader(), MarkdownLoader(), PdfLoader()),
            max_source_bytes=max_source_bytes,
        )
        source = cast(Path, args.source)
        extra_ignored = tuple(cast(list[str], args.ignore))
        documents = (
            load_directory(
                source,
                registry,
                ignored_paths=settings.ignored_paths + extra_ignored,
            )
            if source.is_dir()
            else registry.load(source)
        )
        report = _inspection_report(
            source,
            documents,
            include_content=bool(args.include_content),
        )
    except (OSError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2

    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


def _inspection_report(
    source: Path,
    documents: Sequence[SourceDocument],
    *,
    include_content: bool,
) -> dict[str, Any]:
    """Build a JSON-serializable, deterministic inspection report."""
    summaries: list[dict[str, Any]] = []
    for document in documents:
        summary: dict[str, Any] = {
            "characters": len(document.content),
            "id": document.id,
            "lines": len(document.content.splitlines()),
            "media_type": document.media_type,
            "metadata": document.metadata,
            "source": document.source,
            "title": document.title,
        }
        if include_content:
            summary["content"] = document.content
        summaries.append(summary)

    return {
        "document_count": len(summaries),
        "documents": summaries,
        "input": source.resolve().as_posix(),
    }


def _positive_integer(value: str) -> int:
    """Parse a strictly positive integer for argparse options."""
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be a positive integer")
    return parsed


if __name__ == "__main__":  # pragma: no cover - exercised through entry point
    raise SystemExit(main())
