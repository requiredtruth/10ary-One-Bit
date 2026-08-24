"""Reproducible reference-kernel measurement with correctness checks."""
from __future__ import annotations
import platform
import statistics
import time
import numpy as np
from .format import PackedMatrix, unpack
from .runtime import matvec


def benchmark(packed: PackedMatrix, repeats: int = 25, seed: int = 7) -> dict:
    if repeats < 5: raise ValueError("repeats must be at least 5")
    rng = np.random.default_rng(seed)
    x = rng.standard_normal(packed.cols, dtype=np.float32)
    expected = unpack(packed) @ x
    actual = matvec(packed, x)
    if not np.allclose(actual, expected, rtol=2e-4, atol=2e-4):
        raise RuntimeError("reference matvec failed correctness check")
    timings = []
    for _ in range(repeats):
        start = time.perf_counter_ns(); matvec(packed, x); timings.append((time.perf_counter_ns() - start) / 1e6)
    ordered = sorted(timings)
    p95 = ordered[min(len(ordered) - 1, int(.95 * len(ordered)))]
    return {"rows": packed.rows, "cols": packed.cols, "repeats": repeats, "median_ms": statistics.median(timings), "p95_ms": p95, "stored_bits_per_weight": packed.stored_bits_per_weight, "python": platform.python_version(), "machine": platform.machine(), "kernel": "scalar-reference"}

