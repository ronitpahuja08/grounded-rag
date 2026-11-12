"""Tests for source hashes and timestamps."""

import os
from pathlib import Path

from grounded_rag import hash_bytes, hash_text, source_provenance


def test_source_provenance_separates_file_and_normalized_content_hashes(
    tmp_path: Path,
) -> None:
    source = tmp_path / "notes.txt"
    source.write_bytes(b"Evidence\r\n")

    metadata = source_provenance(source, "Evidence\n")

    assert metadata["source_hash"] == hash_bytes(b"Evidence\r\n")
    assert metadata["content_hash"] == hash_text("Evidence\n")
    assert metadata["source_hash"] != metadata["content_hash"]


def test_source_provenance_emits_utc_timestamp_with_microseconds(
    tmp_path: Path,
) -> None:
    source = tmp_path / "notes.txt"
    source.write_text("Evidence", encoding="utf-8")
    modified_ns = 1_700_000_000_987_654_000
    os.utime(source, ns=(modified_ns, modified_ns))

    metadata = source_provenance(source, "Evidence\n")

    assert metadata["source_modified_at"] == (
        "2023-11-14T22:13:20.987654+00:00"
    )
