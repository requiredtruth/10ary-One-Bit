"""Deterministic N:M masks for FFN experiments aligned to flat blocks."""
from __future__ import annotations
import numpy as np


def nm_mask(weights: np.ndarray, n: int = 2, m: int = 4) -> np.ndarray:
    w = np.asarray(weights)
    if w.ndim != 2: raise ValueError("weights must be rank 2")
    if not 0 < n <= m or w.shape[1] % m: raise ValueError("require 0 < n <= m and columns divisible by m")
    blocks = np.abs(w).reshape(w.shape[0], -1, m)
    keep = np.argpartition(blocks, m - n, axis=-1)[..., m - n :]
    mask = np.zeros_like(blocks, dtype=bool)
    np.put_along_axis(mask, keep, True, axis=-1)
    return mask.reshape(w.shape)


def apply_nm(weights: np.ndarray, n: int = 2, m: int = 4) -> tuple[np.ndarray, np.ndarray]:
    mask = nm_mask(weights, n, m)
    return np.where(mask, weights, 0), mask

