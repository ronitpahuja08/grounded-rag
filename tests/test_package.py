"""Package-level checks that protect the installable project baseline."""

from importlib.metadata import version
from importlib.resources import files

import grounded_rag


def test_public_version_matches_installed_metadata() -> None:
    """The public version must stay aligned with package metadata."""
    assert grounded_rag.__version__ == version("grounded-rag")


def test_typing_marker_is_included_in_distribution() -> None:
    """Consumers should receive the PEP 561 marker with the package."""
    typing_marker = files("grounded_rag").joinpath("py.typed")

    assert typing_marker.is_file()
