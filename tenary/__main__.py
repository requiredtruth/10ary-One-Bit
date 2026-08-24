"""Command line for inspecting and verifying T10B1 artifacts."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import tempfile
import numpy as np
from .benchmark import benchmark
from .format import pack, read, unpack, write
from .runtime import matvec


def details(packed) -> dict:
    return {"format": f"T10B1 v{packed.major}.{packed.minor}", "shape": [packed.rows, packed.cols], "group_size": 10, "group_count": packed.group_count, "logical_payload_bits_per_weight": 1.0, "stored_bits_per_weight_including_scales_and_padding": packed.stored_bits_per_weight, "accumulator": "float32", "endianness": "little"}


def main() -> int:
    parser = argparse.ArgumentParser(description="T10B1 packed binary matrix reference tools")
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("pack"); p.add_argument("input"); p.add_argument("output")
    p = sub.add_parser("unpack"); p.add_argument("input"); p.add_argument("output")
    p = sub.add_parser("inspect"); p.add_argument("input")
    p = sub.add_parser("benchmark"); p.add_argument("input"); p.add_argument("--repeats", type=int, default=25); p.add_argument("--seed", type=int, default=7)
    sub.add_parser("self-test")
    args = parser.parse_args()
    if args.command == "pack": write(args.output, pack(np.load(args.input, allow_pickle=False))); print(json.dumps(details(read(args.output)), indent=2))
    elif args.command == "unpack": np.save(args.output, unpack(read(args.input)), allow_pickle=False); print(args.output)
    elif args.command == "inspect": print(json.dumps(details(read(args.input)), indent=2))
    elif args.command == "benchmark": print(json.dumps(benchmark(read(args.input), args.repeats, args.seed), indent=2, sort_keys=True))
    else:
        rng = np.random.default_rng(7); weights = rng.standard_normal((32, 40), dtype=np.float32)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "self-test.t10b"; packed = pack(weights); write(path, packed); loaded = read(path)
            vector = rng.standard_normal(40, dtype=np.float32)
            assert np.allclose(matvec(loaded, vector), unpack(loaded) @ vector, rtol=2e-4, atol=2e-4)
            print("T10B1 self-test: PASS")
            print(json.dumps(details(loaded), indent=2))
    return 0


if __name__ == "__main__": raise SystemExit(main())

