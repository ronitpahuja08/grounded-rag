"""Tests for loader registration and file dispatch."""

from collections.abc import Sequence
from pathlib import Path

import pytest

from grounded_rag import (
    AmbiguousLoaderError,
    EmptyLoaderResultError,
    EmptyTextSourceError,
    LoaderRegistry,
    SourceDocument,
    TextDecodingError,
    TextLoader,
    UnsupportedSourceError,
)


class SuffixLoader:
    """Small deterministic loader fake keyed by a file suffix."""

    def __init__(self, suffix: str, *, empty: bool = False) -> None:
        self.suffix = suffix
        self.empty = empty

    def supports(self, source: Path) -> bool:
        return source.suffix.lower() == self.suffix

    def load(self, source: Path) -> Sequence[SourceDocument]:
        if self.empty:
            return ()
        return (
            SourceDocument(
                id=f"fixture:{source.stem}",
                content="Synthetic loader content",
                source=source.as_posix(),
                media_type="text/plain",
            ),
        )


def test_registry_dispatches_to_single_matching_loader(tmp_path: Path) -> None:
    source = tmp_path / "notes.txt"
    source.write_text("fixture", encoding="utf-8")
    registry = LoaderRegistry((SuffixLoader(".md"), SuffixLoader(".txt")))

    documents = registry.load(source)

    assert registry.registered_count == 2
    assert documents[0].id == "fixture:notes"


def test_registry_rejects_unsupported_source(tmp_path: Path) -> None:
    source = tmp_path / "archive.bin"
    source.write_bytes(b"fixture")

    with pytest.raises(UnsupportedSourceError, match=r"archive\.bin"):
        LoaderRegistry((SuffixLoader(".txt"),)).load(source)


def test_registry_rejects_ambiguous_dispatch(tmp_path: Path) -> None:
    source = tmp_path / "notes.txt"
    source.write_text("fixture", encoding="utf-8")
    registry = LoaderRegistry((SuffixLoader(".txt"), SuffixLoader(".txt")))

    with pytest.raises(AmbiguousLoaderError, match="multiple loaders"):
        registry.load(source)


def test_registry_rejects_empty_loader_result(tmp_path: Path) -> None:
    source = tmp_path / "notes.txt"
    source.write_text("fixture", encoding="utf-8")

    with pytest.raises(EmptyLoaderResultError, match="returned no documents"):
        LoaderRegistry((SuffixLoader(".txt", empty=True),)).load(source)


def test_registry_requires_existing_file(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match=r"missing\.txt"):
        LoaderRegistry((SuffixLoader(".txt"),)).load(tmp_path / "missing.txt")


def test_registry_rejects_duplicate_loader_instance() -> None:
    loader = SuffixLoader(".txt")
    registry = LoaderRegistry((loader,))

    with pytest.raises(ValueError, match="already registered"):
        registry.register(loader)


def test_text_loader_reads_utf8_content_with_traceable_metadata(
    tmp_path: Path,
) -> None:
    source = tmp_path / "Café notes.txt"
    source.write_bytes("First line\r\nCafé evidence\r".encode())

    (document,) = LoaderRegistry((TextLoader(),)).load(source)

    assert document.content == "First line\nCafé evidence\n"
    assert document.media_type == "text/plain"
    assert document.title == "Café notes"
    assert document.source == source.resolve().as_posix()
    assert document.metadata == {
        "encoding": "utf-8",
        "location": source.resolve().as_posix(),
    }


def test_text_loader_identifier_is_stable_for_same_location(tmp_path: Path) -> None:
    source = tmp_path / "notes.txt"
    source.write_text("Version one", encoding="utf-8")
    loader = TextLoader()
    first_id = loader.load(source)[0].id

    source.write_text("Version two", encoding="utf-8")

    assert loader.load(source)[0].id == first_id


def test_text_loader_matches_txt_extension_case_insensitively() -> None:
    loader = TextLoader()

    assert loader.supports(Path("notes.txt"))
    assert loader.supports(Path("NOTES.TXT"))
    assert not loader.supports(Path("notes.md"))


def test_text_loader_rejects_non_utf8_bytes(tmp_path: Path) -> None:
    source = tmp_path / "legacy.txt"
    source.write_bytes(b"\xff\xfeinvalid")

    with pytest.raises(TextDecodingError, match="valid UTF-8"):
        TextLoader().load(source)


def test_text_loader_rejects_whitespace_only_files(tmp_path: Path) -> None:
    source = tmp_path / "empty.txt"
    source.write_text(" \n\t", encoding="utf-8")

    with pytest.raises(EmptyTextSourceError, match="no readable content"):
        TextLoader().load(source)
