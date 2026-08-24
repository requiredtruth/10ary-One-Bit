"""Correctness-first CPU reference kernels; no hidden native dependencies."""
from __future__ import annotations
import numpy as np
from .format import GROUP_SIZE, PackedMatrix


def matvec(packed: PackedMatrix, vector: np.ndarray) -> np.ndarray:
    """Multiply directly from masks/scales without materializing the matrix."""
    x = np.asarray(vector, dtype=np.float32)
    if x.shape != (packed.cols,):
        raise ValueError(f"vector shape must be ({packed.cols},)")
    if not np.isfinite(x).all():
        raise ValueError("vector contains NaN or infinity")
    out = np.zeros(packed.rows, dtype=np.float32)
    index = 0
    for row in range(packed.rows):
        total = np.float32(0)
        for start in range(0, packed.cols, GROUP_SIZE):
            width = min(GROUP_SIZE, packed.cols - start)
            mask, scale = int(packed.masks[index]), np.float32(packed.scales[index])
            subtotal = np.float32(0)
            for bit in range(width):
                subtotal += x[start + bit] if mask & (1 << bit) else -x[start + bit]
            total += subtotal * scale
            index += 1
        out[row] = total
    return out

