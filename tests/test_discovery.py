"""Tests for deterministic directory discovery and loading."""

from pathlib import Path

import pytest

from grounded_rag import (
    LoaderRegistry,
    MarkdownLoader,
    TextLoader,
    discover_sources,
    load_directory,
)


def _relative_paths(root: Path, sources: tuple[Path, ...]) -> list[str]:
    return [source.relative_to(root).as_posix() for source in sources]


def test_discovers_supported_sources_in_deterministic_order(
    tmp_path: Path,
) -> None:
    (tmp_path / "zeta").mkdir()
    (tmp_path / "Alpha").mkdir()
    (tmp_path / "zeta" / "notes.txt").write_text("Notes", encoding="utf-8")
    (tmp_path / "Alpha" / "guide.md").write_text("# Guide", encoding="utf-8")
    (tmp_path / "README.MD").write_text("# Readme", encoding="utf-8")
    (tmp_path / "data.csv").write_text("unsupported", encoding="utf-8")
    registry = LoaderRegistry((TextLoader(), MarkdownLoader()))

    sources = discover_sources(tmp_path, registry)

    assert _relative_paths(tmp_path, sources) == [
        "Alpha/guide.md",
        "README.MD",
        "zeta/notes.txt",
    ]


def test_discovery_ignores_default_and_custom_path_patterns(
    tmp_path: Path,
) -> None:
    (tmp_path / ".git").mkdir()
    (tmp_path / "drafts").mkdir()
    (tmp_path / "published").mkdir()
    (tmp_path / ".git" / "message.txt").write_text("internal", encoding="utf-8")
    (tmp_path / "drafts" / "idea.md").write_text("# Draft", encoding="utf-8")
    (tmp_path / "published" / "guide.md").write_text("# Guide", encoding="utf-8")
    registry = LoaderRegistry((TextLoader(), MarkdownLoader()))

    sources = discover_sources(
        tmp_path,
        registry,
        ignored_paths=(".git", "drafts/**"),
    )

    assert _relative_paths(tmp_path, sources) == ["published/guide.md"]


def test_load_directory_flattens_documents_in_source_order(tmp_path: Path) -> None:
    (tmp_path / "b.txt").write_text("Second", encoding="utf-8")
    (tmp_path / "a.md").write_text("# First", encoding="utf-8")
    registry = LoaderRegistry((TextLoader(), MarkdownLoader()))

    documents = load_directory(tmp_path, registry)

    assert [document.title for document in documents] == ["First", "b"]
    assert [document.content for document in documents] == ["# First\n", "Second\n"]


def test_discovery_requires_existing_directory(tmp_path: Path) -> None:
    registry = LoaderRegistry((TextLoader(),))

    with pytest.raises(NotADirectoryError, match="source directory"):
        discover_sources(tmp_path / "missing", registry)
