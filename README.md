# 10ary-One-Bit

A CPU-first PySide6 laboratory for the T10B1 ten-weight scaled one-bit matrix format. It packs and validates checksummed artifacts, runs packed matvec correctness checks, measures physical storage honestly, and exposes the full workflow through a live desktop control panel.

## Run the GUI

```sh
chmod +x install.sh run.sh cli.sh
./run.sh
```

`run.sh` is the normal entry point. It detects a missing or incomplete `.venv`, runs `install.sh` automatically, then opens the PySide6 control panel. The GUI shows live status, indeterminate/complete progress, output, errors, and actions for:

- a complete generated synthetic demo
- packing a selected NumPy matrix
- inspecting a selected T10B1 artifact
- benchmarking a selected artifact

The default **Run Complete Demo** action generates a deterministic 32×40 float32 matrix in a temporary directory, creates its `.t10b` artifact, unpacks it, verifies packed matvec against NumPy, benchmarks it, reports the measured result, and removes temporary files. It never expects `weights.t10b` or another external file to exist.

For headless verification:

```sh
./run.sh --demo --repeats 5
```

## Command line

CLI-only behavior is preserved through `cli.sh`:

```sh
./cli.sh self-test
./cli.sh pack weights.npy weights.t10b
./cli.sh inspect weights.t10b
./cli.sh unpack weights.t10b decoded.npy
./cli.sh benchmark weights.t10b --repeats 25
```

Explicit installation or repair:

```sh
./install.sh
./install.sh doctor
./install.sh repair
```

## Format and research primitives

- cosine quantization-hardness scheduling
- per-row scaled binary forward quantization
- error-aware surrogate-gradient damping
- code occupancy, flip-rate, error, and gradient telemetry
- deterministic N:M magnitude sparsity masks
- checksummed, versioned, canonical little-endian T10B1 artifacts with strict scale and padding validation
- packed scalar CPU matvec without materializing weights

See [FORMAT.md](FORMAT.md) for equations, alignment, padding, precision, serialization, and rejection rules.

## Honest limitations

- The current kernel is a correctness-first Python scalar reference, not SIMD.
- The GUI demonstrates matrix-format operations; this repository is not a pretrained text model.
- Float16 scales, aligned containers, headers, and padding raise physical storage above one logical sign bit per weight.
- No weights or datasets are downloaded or committed.
- Any earlier training/decode figures are project telemetry, not reproduced benchmarks.

## Support

See [SUPPORT.md](SUPPORT.md). Confirm the asset, address, and network before sending.

## License

MIT. See [LICENSE](LICENSE).
