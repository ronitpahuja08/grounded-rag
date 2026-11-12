"""Source provenance metadata shared by document loaders."""

from datetime import UTC, datetime
from pathlib import Path

from pydantic import JsonValue

from grounded_rag.hashing import hash_bytes, hash_text

_NANOSECONDS_PER_SECOND = 1_000_000_000
_NANOSECONDS_PER_MICROSECOND = 1_000


def source_provenance(source: Path, content: str) -> dict[str, JsonValue]:
    """Return deterministic hashes and the source modification timestamp."""
    stat = source.stat()
    seconds, nanoseconds = divmod(stat.st_mtime_ns, _NANOSECONDS_PER_SECOND)
    modified_at = datetime.fromtimestamp(seconds, tz=UTC).replace(
        microsecond=nanoseconds // _NANOSECONDS_PER_MICROSECOND
    )
    return {
        "content_hash": hash_text(content),
        "source_hash": hash_bytes(source.read_bytes()),
        "source_modified_at": modified_at.isoformat(),
    }
