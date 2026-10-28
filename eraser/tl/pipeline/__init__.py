"""Locate and remove burned-in overlays from photographs.

    from pipeline import clean
    clean("listing.jpg", "cleaned.jpg")

Stages, and why each exists, are documented in the module that implements it:
`localize` (whether and where), `masking` (which pixels), `erase` (what replaces them).
"""
from .run import clean, load_all      # noqa: F401

__all__ = ["clean", "load_all"]
