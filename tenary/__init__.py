"""Reference tools for the T10B1 packed binary matrix format."""

__version__ = "0.1.0"

from .format import PackedMatrix, pack, read, unpack

__all__ = ["PackedMatrix", "pack", "read", "unpack"]

