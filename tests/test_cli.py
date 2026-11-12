"""Tests for the document inspection command."""

import json
from pathlib import Path

import pytest

from grounded_rag.cli import main


def test_inspect_file_prints_metadata_without_content_by_default(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    source = tmp_path / "notes.txt"
    source.write_text("Grounded evidence", encoding="utf-8")

    exit_code = main(("inspect", str(source)))
    captured = capsys.readouterr()
    report = json.loads(captured.out)

    assert exit_code == 0
    assert captured.err == ""
    assert report["document_count"] == 1
    assert report["documents"][0]["media_type"] == "text/plain"
    assert "content_hash" in report["documents"][0]["metadata"]
    assert "content" not in report["documents"][0]


def test_inspect_directory_is_ordered_and_honors_extra_ignores(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    (tmp_path / "drafts").mkdir()
    (tmp_path / "z.txt").write_text("Last", encoding="utf-8")
    (tmp_path / "a.md").write_text("# First", encoding="utf-8")
    (tmp_path / "drafts" / "hidden.md").write_text("# Hidden", encoding="utf-8")

    exit_code = main(("inspect", str(tmp_path), "--ignore", "drafts"))
    captured = capsys.readouterr()
    report = json.loads(captured.out)

    assert exit_code == 0
    assert [document["title"] for document in report["documents"]] == [
        "First",
        "z",
    ]


def test_inspect_can_include_normalized_content(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    source = tmp_path / "notes.txt"
    source.write_bytes(b"First\r\nSecond")

    exit_code = main(("inspect", str(source), "--include-content"))
    captured = capsys.readouterr()
    report = json.loads(captured.out)

    assert exit_code == 0
    assert report["documents"][0]["content"] == "First\nSecond\n"


def test_inspect_returns_actionable_error_for_unsupported_source(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    source = tmp_path / "archive.bin"
    source.write_bytes(b"binary")

    exit_code = main(("inspect", str(source)))
    captured = capsys.readouterr()

    assert exit_code == 2
    assert captured.out == ""
    assert "no loader supports source" in captured.err
    assert "archive.bin" in captured.err


def test_inspect_enforces_command_line_size_override(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    source = tmp_path / "notes.txt"
    source.write_text("too large", encoding="utf-8")

    exit_code = main(
        ("inspect", str(source), "--max-source-bytes", "4")
    )
    captured = capsys.readouterr()

    assert exit_code == 2
    assert "exceeding the 4-byte limit" in captured.err
