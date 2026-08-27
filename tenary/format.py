"""Versioned, checksummed T10B1 serialization.

T10B1 stores ten binary signs per logical group. The sign payload is one bit
per weight; the v1 reference file uses a 16-bit aligned mask and one float16
scale per group so its honest physical average is 3.2 bits/weight before the
fixed header and row padding.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import struct
import zlib
import numpy as np

MAGIC = b"T10B"
MAJOR, MINOR = 1, 0
GROUP_SIZE = 10
HEADER = struct.Struct("<4sBBBBIIQII")
FLAG_SCALED = 1
MASK_VALUE_BITS = (1 << GROUP_SIZE) - 1
UINT32_MAX = (1 << 32) - 1
UINT64_MAX = (1 << 64) - 1
FLOAT16_MAX = float(np.finfo(np.float16).max)


@dataclass(frozen=True)
class PackedMatrix:
    rows: int
    cols: int
    scales: np.ndarray
    masks: np.ndarray
    major: int = MAJOR
    minor: int = MINOR

    @property
    def groups_per_row(self) -> int:
        return (self.cols + GROUP_SIZE - 1) // GROUP_SIZE

    @property
    def group_count(self) -> int:
        return self.rows * self.groups_per_row

    @property
    def payload_bits_per_weight(self) -> float:
        return 1.0

    @property
    def stored_bits_per_weight(self) -> float:
        return 32.0 * self.group_count / (self.rows * self.cols)


def _validate_packed(packed: PackedMatrix) -> None:
    """Reject objects that cannot have one canonical T10B1 v1 encoding."""
    if not isinstance(packed.rows, (int, np.integer)) or not isinstance(packed.cols, (int, np.integer)):
        raise ValueError("matrix dimensions must be integers")
    if not 0 < int(packed.rows) <= UINT32_MAX or not 0 < int(packed.cols) <= UINT32_MAX:
        raise ValueError("matrix dimensions must be non-zero uint32 values")
    if packed.major != MAJOR or packed.minor != MINOR:
        raise ValueError(f"unsupported T10B1 version: {packed.major}.{packed.minor}")

    expected = int(packed.rows) * ((int(packed.cols) + GROUP_SIZE - 1) // GROUP_SIZE)
    if expected > UINT64_MAX:
        raise ValueError("group count exceeds the T10B1 uint64 field")
    scales = np.asarray(packed.scales)
    masks = np.asarray(packed.masks)
    if scales.ndim != 1 or masks.ndim != 1 or len(scales) != expected or len(masks) != expected:
        raise ValueError("scale/mask count does not match matrix dimensions")
    if not np.issubdtype(scales.dtype, np.number) or np.issubdtype(scales.dtype, np.complexfloating):
        raise ValueError("scales must be real numbers")
    if not np.isfinite(scales).all() or np.any(scales <= 0) or np.any(scales > FLOAT16_MAX):
        raise ValueError("scales must be finite positive float16 values")
    if not np.issubdtype(masks.dtype, np.integer):
        raise ValueError("masks must be integers")
    if np.any(masks < 0) or np.any(masks.astype(np.uint64) & ~np.uint64(MASK_VALUE_BITS)):
        raise ValueError("mask uses reserved bits outside the ten-weight group")

    final_width = int(packed.cols) % GROUP_SIZE
    if final_width:
        groups_per_row = (int(packed.cols) + GROUP_SIZE - 1) // GROUP_SIZE
        padding_mask = MASK_VALUE_BITS ^ ((1 << final_width) - 1)
        final_masks = masks[groups_per_row - 1 :: groups_per_row].astype(np.uint64)
        if np.any(final_masks & np.uint64(padding_mask)):
            raise ValueError("final group has non-zero row-padding bits")


def pack(weights: np.ndarray) -> PackedMatrix:
    """Quantize a finite rank-2 array to scaled binary groups."""
    matrix = np.asarray(weights, dtype=np.float32)
    if matrix.ndim != 2 or 0 in matrix.shape:
        raise ValueError("weights must be a non-empty rank-2 array")
    if not np.isfinite(matrix).all():
        raise ValueError("weights contain NaN or infinity")
    rows, cols = matrix.shape
    if rows > UINT32_MAX or cols > UINT32_MAX:
        raise ValueError("matrix dimensions exceed the T10B1 uint32 fields")
    groups = (cols + GROUP_SIZE - 1) // GROUP_SIZE
    scales = np.empty(rows * groups, dtype="<f2")
    masks = np.zeros(rows * groups, dtype="<u2")
    index = 0
    for row in matrix:
        for start in range(0, cols, GROUP_SIZE):
            block = row[start : start + GROUP_SIZE]
            scale = float(np.mean(np.abs(block), dtype=np.float64))
            if scale > FLOAT16_MAX:
                raise ValueError("group scale exceeds the finite float16 range")
            scales[index] = max(scale, np.finfo(np.float16).tiny)
            mask = 0
            for bit, value in enumerate(block):
                if value >= 0: mask |= 1 << bit
            masks[index] = mask
            index += 1
    return PackedMatrix(rows, cols, scales, masks)


def unpack(packed: PackedMatrix) -> np.ndarray:
    """Decode to float32 without reading padded signs outside matrix bounds."""
    out = np.empty((packed.rows, packed.cols), dtype=np.float32)
    index = 0
    for row in range(packed.rows):
        for start in range(0, packed.cols, GROUP_SIZE):
            width = min(GROUP_SIZE, packed.cols - start)
            mask, scale = int(packed.masks[index]), float(packed.scales[index])
            for bit in range(width):
                out[row, start + bit] = scale if mask & (1 << bit) else -scale
            index += 1
    return out


def to_bytes(packed: PackedMatrix) -> bytes:
    _validate_packed(packed)
    payload = np.asarray(packed.scales).astype("<f2", copy=False).tobytes() + np.asarray(packed.masks).astype("<u2", copy=False).tobytes()
    payload_crc = zlib.crc32(payload)
    prefix = HEADER.pack(MAGIC, packed.major, packed.minor, 1, FLAG_SCALED, packed.rows, packed.cols, packed.group_count, payload_crc, 0)
    header_crc = zlib.crc32(prefix[:-4])
    return HEADER.pack(MAGIC, packed.major, packed.minor, 1, FLAG_SCALED, packed.rows, packed.cols, packed.group_count, payload_crc, header_crc) + payload


def from_bytes(data: bytes) -> PackedMatrix:
    if len(data) < HEADER.size:
        raise ValueError("file is shorter than the 32-byte T10B1 header")
    magic, major, minor, endian, flags, rows, cols, count, payload_crc, header_crc = HEADER.unpack_from(data)
    if magic != MAGIC: raise ValueError("bad T10B1 magic; expected T10B")
    if major != MAJOR: raise ValueError(f"unsupported T10B1 major version: {major}")
    if endian != 1: raise ValueError("unsupported endianness marker")
    if flags != FLAG_SCALED: raise ValueError(f"unsupported T10B1 flags: {flags}")
    if not rows or not cols: raise ValueError("matrix dimensions must be non-zero")
    expected = rows * ((cols + GROUP_SIZE - 1) // GROUP_SIZE)
    if count != expected: raise ValueError("group count does not match matrix dimensions")
    if zlib.crc32(data[: HEADER.size - 4]) != header_crc: raise ValueError("T10B1 header checksum mismatch")
    payload = data[HEADER.size :]
    if len(payload) != count * 4: raise ValueError("T10B1 payload length mismatch")
    if zlib.crc32(payload) != payload_crc: raise ValueError("T10B1 payload checksum mismatch")
    split = count * 2
    scales = np.frombuffer(payload[:split], dtype="<f2").copy()
    masks = np.frombuffer(payload[split:], dtype="<u2").copy()
    packed = PackedMatrix(rows, cols, scales, masks, major, minor)
    _validate_packed(packed)
    return packed


def write(path: str | Path, packed: PackedMatrix) -> None:
    destination = Path(path)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_bytes(to_bytes(packed))
    temporary.replace(destination)


def read(path: str | Path) -> PackedMatrix:
    return from_bytes(Path(path).read_bytes())
