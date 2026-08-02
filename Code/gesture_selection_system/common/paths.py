"""Path helpers shared by training and the runtime pipeline."""

from __future__ import annotations

from pathlib import Path


def resolve_under(root: Path, value: str | Path) -> Path:
    """Resolve a configured path against a folder root when it is relative."""
    path = Path(value)
    if path.is_absolute():
        return path
    return (root / path).resolve()
