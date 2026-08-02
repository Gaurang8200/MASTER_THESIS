"""Inference device selection shared by training and the runtime pipeline."""

from __future__ import annotations


def resolve_device(requested: str) -> str:
    """Pick a device, falling back to cpu when nothing else is usable."""
    if requested != "auto":
        return requested
    try:
        import torch
    except ImportError:
        return "cpu"
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return "mps"
    return "cpu"
