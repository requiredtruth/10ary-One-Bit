# T10B1 format specification — version 1.0

T10B1 is the repository's first **public reference profile**. It does not claim binary compatibility with any earlier experiment or private implementation.

## Meaning of the name

- **T10**: weights are grouped ten at a time within each matrix row.
- **B1**: each quantized weight has a one-bit sign payload with logical alphabet `{-scale, +scale}`.
- This is not a ten-symbol alphabet. A ten-symbol alphabet needs at least four stored code bits and must use a different profile identifier.

Calling the entire file “one bit per weight” would be misleading. T10B1 v1 stores each ten-bit logical sign group in an aligned `uint16` plus one `float16` scale. It therefore uses 32 physical group bits, or 3.2 bits per weight for complete groups, before the fixed header. Row-end padding can increase the average. Tools must report both figures.

## Quantizer

For each row-local group `G` of at most ten finite FP32 master weights:

```text
scale = max(mean(abs(G)), smallest_positive_float16_normal)
code(w) = 1 when w >= 0, otherwise 0
decode(code) = +scale when code is 1, otherwise -scale
```

Zero therefore maps to `+scale`; zero has no native code. Final partial groups are padded only in storage and padding must not enter scale calculation or matrix operations.

## Byte layout

All integers, masks, and scales are little-endian. The 32-byte header is:

| Offset | Type | Meaning |
|---:|---|---|
| 0 | `char[4]` | magic `T10B` |
| 4 | `uint8` | major version `1` |
| 5 | `uint8` | minor version `0` |
| 6 | `uint8` | endianness marker `1` for little |
| 7 | `uint8` | flags; v1 requires `1` (scaled) |
| 8 | `uint32` | matrix rows |
| 12 | `uint32` | matrix columns |
| 16 | `uint64` | group count |
| 24 | `uint32` | CRC32 of complete payload |
| 28 | `uint32` | CRC32 of header bytes 0–27 |

Payload order:

1. `group_count` IEEE-754 binary16 scales in row-major group order.
2. `group_count` `uint16` masks in the same order.

Within a mask, bit 0 represents the first weight and bit 9 the tenth. Bits 10–15 must be zero. Unused bits in a row's final partial group must also be zero, giving each logical matrix one canonical encoding. Groups never cross row boundaries. `group_count = rows × ceil(columns / 10)`.

## Arithmetic and training

- Reference decode and accumulation use FP32.
- Norm parameters, embeddings, activations, optimizer state, attention layout, tokenization, and LM-head storage are deliberately outside this matrix-profile specification.
- Training should keep latent/master weights in at least FP16 and may use scheduled hardening plus an explicitly documented surrogate gradient.
- Files are inference artifacts. They do not contain optimizer state.

## Validation requirements

Readers must reject bad magic, unsupported versions or flags, impossible dimensions/group counts, incorrect lengths, non-finite or non-positive scales, non-zero reserved/padding bits, and failed header or payload checksums. Writers must reject a group whose scale is not representable as finite binary16 instead of silently producing infinity. Load files only from sources you trust; checksums detect corruption, not malicious intent.
