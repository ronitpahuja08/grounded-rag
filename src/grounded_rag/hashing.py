"""Deterministic SHA-256 helpers for source and chunk identity."""

import hashlib
import unicodedata

_LENGTH_PREFIX_BYTES = 8


def normalize_text(text: str) -> str:
    """Normalize Unicode and line endings without changing other whitespace."""
    normalized_lines = text.replace("\r\n", "\n").replace("\r", "\n")
    return unicodedata.normalize("NFC", normalized_lines)


def hash_bytes(content: bytes) -> str:
    """Return a lowercase SHA-256 digest for exact bytes."""
    return hashlib.sha256(content).hexdigest()


def hash_text(content: str) -> str:
    """Hash normalized UTF-8 text consistently across platforms."""
    return hash_bytes(normalize_text(content).encode("utf-8"))


def hash_parts(*parts: str | bytes) -> str:
    """Hash ordered values using length prefixes to preserve boundaries."""
    if not parts:
        msg = "at least one hash part is required"
        raise ValueError(msg)

    digest = hashlib.sha256()
    for part in parts:
        payload = (
            normalize_text(part).encode("utf-8") if isinstance(part, str) else part
        )
        digest.update(len(payload).to_bytes(_LENGTH_PREFIX_BYTES, byteorder="big"))
        digest.update(payload)
    return digest.hexdigest()
