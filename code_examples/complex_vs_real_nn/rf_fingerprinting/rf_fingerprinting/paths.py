"""Path utilities for the RF fingerprinting module."""

from __future__ import annotations

from pathlib import Path


def project_root() -> Path:
    """Return the project root for this module.

    Returns:
        Absolute path to ``.../rf_fingerprinting``.
    """
    return Path(__file__).resolve().parents[1]


def ensure_dir(path: str | Path) -> Path:
    """Create a directory if needed and return it.

    Args:
        path: Directory path.

    Returns:
        Resolved directory path.
    """
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p
