import tempfile
import unittest
from pathlib import Path
import numpy as np

from tenary.benchmark import benchmark
from tenary.format import from_bytes, pack, read, to_bytes, unpack, write
from tenary.runtime import matvec
from tenary.sparsity import apply_nm
from tenary.training import cosine_hardness, error_aware_surrogate, scheduled_binary, telemetry
from tenary.gui import run_demo


class FormatTests(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(7)
        self.weights = self.rng.standard_normal((3, 17), dtype=np.float32)

    def test_roundtrip_and_physical_accounting(self):
        packed = pack(self.weights); decoded = unpack(from_bytes(to_bytes(packed)))
        self.assertEqual(decoded.shape, self.weights.shape)
        self.assertEqual(packed.group_count, 6)
        self.assertAlmostEqual(packed.stored_bits_per_weight, 192 / 51)

    def test_reference_matvec_matches_decoded_numpy(self):
        packed = pack(self.weights); x = self.rng.standard_normal(17, dtype=np.float32)
        np.testing.assert_allclose(matvec(packed, x), unpack(packed) @ x, rtol=2e-4, atol=2e-4)

    def test_corruption_is_rejected(self):
        damaged = bytearray(to_bytes(pack(self.weights))); damaged[-1] ^= 1
        with self.assertRaisesRegex(ValueError, "payload checksum mismatch"): from_bytes(bytes(damaged))

    def test_atomic_file_roundtrip(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "matrix.t10b"; write(path, pack(self.weights))
            self.assertEqual(read(path).rows, 3)

    def test_quantization_schedule_and_surrogate(self):
        self.assertEqual(cosine_hardness(0, 100), 0)
        self.assertAlmostEqual(cosine_hardness(100, 100), 1)
        hard = scheduled_binary(self.weights, 1)
        grad = np.ones_like(self.weights)
        corrected = error_aware_surrogate(grad, self.weights, hard)
        self.assertTrue((corrected <= grad).all())
        report, signs = telemetry(self.weights, None, grad, 1)
        self.assertEqual(signs.shape, self.weights.shape)
        self.assertGreaterEqual(report.relative_error, 0)

    def test_two_of_four_sparsity(self):
        weights = self.rng.standard_normal((4, 12), dtype=np.float32)
        sparse, mask = apply_nm(weights)
        self.assertTrue((mask.reshape(4, 3, 4).sum(axis=-1) == 2).all())
        self.assertTrue((sparse[~mask] == 0).all())

    def test_benchmark_checks_and_reports(self):
        result = benchmark(pack(self.weights), repeats=5)
        self.assertEqual(result["kernel"], "scalar-reference")
        self.assertGreater(result["median_ms"], 0)

    def test_gui_demo_needs_no_preexisting_artifact(self):
        result = run_demo(repeats=5)
        self.assertEqual(result["status"], "PASS")
        self.assertTrue(result["artifact_created"])
        self.assertTrue(result["matvec_matches_oracle"])


if __name__ == "__main__": unittest.main()

