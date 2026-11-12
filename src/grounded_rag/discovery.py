"""Deterministic source discovery for directory ingestion."""

from fnmatch import fnmatchcase
from pathlib import Path

from grounded_rag.loaders import LoaderRegistry, UnsupportedSourceError
from grounded_rag.models import SourceDocument

DEFAULT_IGNORED_PATHS = (".git", ".venv", "__pycache__", "node_modules")


def discover_sources(
    root: str | Path,
    registry: LoaderRegistry,
    *,
    ignored_paths: tuple[str, ...] = DEFAULT_IGNORED_PATHS,
) -> tuple[Path, ...]:
    """Return supported files beneath ``root`` in stable relative-path order."""
    directory = Path(root).resolve()
    if not directory.is_dir():
        msg = f"source directory does not exist: {directory}"
        raise NotADirectoryError(msg)

    discovered: list[Path] = []
    for candidate in directory.rglob("*"):
        if not candidate.is_file():
            continue
        relative = candidate.relative_to(directory)
        if _is_ignored(relative, ignored_paths):
            continue
        try:
            registry.resolve(candidate)
        except UnsupportedSourceError:
            continue
        discovered.append(candidate.resolve())

    return tuple(
        sorted(
            discovered,
            key=lambda path: (
                path.relative_to(directory).as_posix().casefold(),
                path.relative_to(directory).as_posix(),
            ),
        )
    )


def load_directory(
    root: str | Path,
    registry: LoaderRegistry,
    *,
    ignored_paths: tuple[str, ...] = DEFAULT_IGNORED_PATHS,
) -> tuple[SourceDocument, ...]:
    """Load every discovered source while retaining deterministic order."""
    return tuple(
        document
        for source in discover_sources(
            root,
            registry,
            ignored_paths=ignored_paths,
        )
        for document in registry.load(source)
    )


def _is_ignored(relative: Path, patterns: tuple[str, ...]) -> bool:
    """Match path globs and directory names consistently across platforms."""
    relative_path = relative.as_posix().casefold()
    parts = tuple(part.casefold() for part in relative.parts)
    for raw_pattern in patterns:
        pattern = raw_pattern.strip().replace("\\", "/").casefold()
        if not pattern:
            continue
        if fnmatchcase(relative_path, pattern):
            return True
        if "/" not in pattern and any(
            fnmatchcase(part, pattern) for part in parts
        ):
            return True
    return False
