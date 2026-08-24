# 10ary-One-Bit

A CPU-first reference format and measurement toolkit for ten-weight, scaled one-bit matrix groups. Version 0.1 makes the storage claims inspectable: it packs matrices, validates checksummed artifacts, multiplies directly from packed signs and scales, reports honest physical bits per weight, and supplies framework-neutral training/sparsity experiment primitives.

This is a foundation for low-bit language-model research—not a pretrained model, optimized production kernel, or claim of language-model quality.

## Verify in one command

```console
$ sh doit.sh
...
Ran 7 tests
OK
T10B1 self-test: PASS
{
  "format": "T10B1 v1.0",
  "shape": [32, 40],
  "group_size": 10,
  "logical_payload_bits_per_weight": 1.0,
  "stored_bits_per_weight_including_scales_and_padding": 3.2
}
```

Only NumPy is required. No weights or datasets are downloaded.

## Useful commands

```bash
python3 -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt

python3 -m tenary pack weights.npy weights.t10b
python3 -m tenary inspect weights.t10b
python3 -m tenary unpack weights.t10b decoded.npy
python3 -m tenary benchmark weights.t10b --repeats 25
```

The benchmark verifies packed matvec correctness before timing and reports median/p95 latency, Python/machine identity, format density, and the exact kernel name. It does not invent tokens/second from a matrix microbenchmark.

## What is different

[BitNet](https://github.com/microsoft/BitNet) is an optimized inference framework for established ternary/low-bit model families. [llama.cpp](https://github.com/ggml-org/llama.cpp) is a broad production inference runtime with many model and quantization formats. This repository is narrower: a small auditable laboratory for one explicitly specified ten-weight binary packing profile, its physical accounting, correctness oracle, and training experiments that leave attention design outside the format.

## Included research primitives

- cosine quantization-hardness scheduling
- per-row scaled binary forward quantization
- quantization-error-aware surrogate-gradient damping
- code occupancy, flip-rate, error, and gradient telemetry
- deterministic N:M magnitude sparsity masks
- checksummed, versioned, little-endian T10B1 artifacts
- packed scalar CPU matvec without materializing weights

The full format, equations, alignment, padding, precision, serialization, and rejection rules are in [FORMAT.md](FORMAT.md).

## Honest limitations

- The current kernel is a correctness-first Python scalar reference, not SIMD.
- T10B1 is a new public reference profile and does not claim compatibility with undocumented earlier experiments.
- The one-bit claim applies to each logical sign code. Float16 scales, aligned mask containers, headers, and padding raise physical storage above one bit per weight.
- Version 0.1 does not implement attention, tokenization, model training, text generation, GGUF conversion, native kernels, energy measurement, or a GUI.
- Any previously observed training/decode figures are project telemetry, not benchmarks reproduced by this repository.

## Support

Public donation addresses and the confirmed-transaction request process are in [SUPPORT.md](SUPPORT.md). Confirm the asset and network before sending.

## License

MIT. See [LICENSE](LICENSE).

