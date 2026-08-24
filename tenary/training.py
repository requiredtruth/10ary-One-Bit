"""Framework-neutral quantization scheduling and surrogate-gradient helpers."""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class QuantizationTelemetry:
    hardness: float
    flip_rate: float
    relative_error: float
    positive_fraction: float
    gradient_scale_mean: float


def cosine_hardness(step: int, total_steps: int, warmup_fraction: float = .05) -> float:
    if total_steps <= 0 or not 0 <= step <= total_steps:
        raise ValueError("step must be between zero and total_steps")
    warmup = int(total_steps * warmup_fraction)
    if step <= warmup: return 0.0
    progress = (step - warmup) / max(total_steps - warmup, 1)
    return float(.5 - .5 * np.cos(np.pi * progress))


def scheduled_binary(latent: np.ndarray, hardness: float) -> np.ndarray:
    """Blend latent weights toward per-row scaled binary values."""
    if not 0 <= hardness <= 1: raise ValueError("hardness must be in [0, 1]")
    w = np.asarray(latent, dtype=np.float32)
    scale = np.maximum(np.mean(np.abs(w), axis=-1, keepdims=True), np.finfo(np.float32).tiny)
    quantized = np.where(w >= 0, scale, -scale)
    return (1 - hardness) * w + hardness * quantized


def error_aware_surrogate(gradient: np.ndarray, latent: np.ndarray, quantized: np.ndarray, damping: float = .5) -> np.ndarray:
    """Dampen STE updates in proportion to normalized local quantization error."""
    g, w, q = map(lambda value: np.asarray(value, dtype=np.float32), (gradient, latent, quantized))
    if g.shape != w.shape or q.shape != w.shape: raise ValueError("gradient, latent, and quantized shapes must match")
    relative = np.abs(w - q) / np.maximum(np.abs(w), 1e-6)
    return g / (1 + damping * relative)


def telemetry(latent: np.ndarray, previous_signs: np.ndarray | None, gradient: np.ndarray, hardness: float) -> tuple[QuantizationTelemetry, np.ndarray]:
    w = np.asarray(latent, dtype=np.float32)
    signs = w >= 0
    q = scheduled_binary(w, 1.0)
    flip = 0.0 if previous_signs is None else float(np.mean(signs != previous_signs))
    relative = float(np.linalg.norm(w - q) / max(float(np.linalg.norm(w)), 1e-12))
    surrogate = error_aware_surrogate(gradient, w, q)
    return QuantizationTelemetry(hardness, flip, relative, float(np.mean(signs)), float(np.mean(np.abs(surrogate)))), signs

