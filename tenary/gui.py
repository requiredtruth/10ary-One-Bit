"""PySide6 control panel and self-contained T10B1 demo."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import tempfile
import traceback
import numpy as np
from .benchmark import benchmark
from .format import pack, read, unpack, write
from .runtime import matvec


def run_demo(repeats: int = 25) -> dict:
    """Create synthetic weights and prove the complete packed workflow."""
    rng = np.random.default_rng(7)
    weights = rng.standard_normal((32, 40), dtype=np.float32)
    vector = rng.standard_normal(40, dtype=np.float32)
    with tempfile.TemporaryDirectory(prefix="t10b1-demo-") as tmp:
        root = Path(tmp)
        source = root / "synthetic-weights.npy"
        artifact = root / "synthetic-weights.t10b"
        decoded = root / "decoded.npy"
        np.save(source, weights, allow_pickle=False)
        write(artifact, pack(np.load(source, allow_pickle=False)))
        loaded = read(artifact)
        np.save(decoded, unpack(loaded), allow_pickle=False)
        packed_result = matvec(loaded, vector)
        oracle = np.load(decoded, allow_pickle=False) @ vector
        np.testing.assert_allclose(packed_result, oracle, rtol=2e-4, atol=2e-4)
        report = benchmark(loaded, repeats=repeats, seed=7)
        return {
            "status": "PASS",
            "input": "generated synthetic 32x40 float32 matrix",
            "artifact_created": True,
            "shape": [loaded.rows, loaded.cols],
            "stored_bits_per_weight": loaded.stored_bits_per_weight,
            "matvec_matches_oracle": True,
            "benchmark": report,
            "note": "Temporary demo files were removed after verification.",
        }


def launch() -> int:
    from PySide6.QtCore import QThread, Signal
    from PySide6.QtWidgets import (
        QApplication, QFileDialog, QHBoxLayout, QLabel, QMainWindow,
        QMessageBox, QPlainTextEdit, QProgressBar, QPushButton, QVBoxLayout, QWidget,
    )

    class Worker(QThread):
        completed = Signal(object)
        failed = Signal(str)
        def __init__(self, action):
            super().__init__(); self.action = action
        def run(self):
            try: self.completed.emit(self.action())
            except Exception: self.failed.emit(traceback.format_exc())

    class Window(QMainWindow):
        def __init__(self):
            super().__init__()
            self.setWindowTitle("10ary-One-Bit Control Panel")
            self.resize(920, 640)
            self.worker = None
            root = QWidget(); layout = QVBoxLayout(root)
            layout.addWidget(QLabel("T10B1 packed-format laboratory — all demos use synthetic local data."))
            buttons = QHBoxLayout()
            for label, slot in [
                ("Run Complete Demo", self.demo),
                ("Pack .npy", self.pack_file),
                ("Inspect .t10b", self.inspect_file),
                ("Benchmark .t10b", self.benchmark_file),
            ]:
                button = QPushButton(label); button.clicked.connect(slot); buttons.addWidget(button)
            layout.addLayout(buttons)
            self.progress = QProgressBar(); self.progress.setRange(0, 100); layout.addWidget(self.progress)
            self.status = QLabel("Ready"); layout.addWidget(self.status)
            self.output = QPlainTextEdit(); self.output.setReadOnly(True); layout.addWidget(self.output)
            self.setCentralWidget(root)

        def start(self, label, action):
            if self.worker and self.worker.isRunning(): return
            self.status.setText(label); self.progress.setRange(0, 0); self.output.clear()
            self.worker = Worker(action)
            self.worker.completed.connect(self.done); self.worker.failed.connect(self.fail); self.worker.start()

        def done(self, value):
            self.progress.setRange(0, 100); self.progress.setValue(100); self.status.setText("Complete")
            self.output.setPlainText(json.dumps(value, indent=2, sort_keys=True) if not isinstance(value, str) else value)

        def fail(self, error):
            self.progress.setRange(0, 100); self.progress.setValue(0); self.status.setText("Failed")
            self.output.setPlainText(error); QMessageBox.critical(self, "Operation failed", error.splitlines()[-1])

        def demo(self): self.start("Generating and validating synthetic demo…", run_demo)

        def pack_file(self):
            source, _ = QFileDialog.getOpenFileName(self, "Select NumPy weights", "", "NumPy (*.npy)")
            if not source: return
            target, _ = QFileDialog.getSaveFileName(self, "Save T10B1 artifact", str(Path(source).with_suffix(".t10b")), "T10B1 (*.t10b)")
            if not target: return
            def action():
                write(target, pack(np.load(source, allow_pickle=False)))
                loaded = read(target)
                return {"status": "PASS", "artifact": target, "shape": [loaded.rows, loaded.cols]}
            self.start("Packing weights…", action)

        def inspect_file(self):
            path, _ = QFileDialog.getOpenFileName(self, "Select T10B1 artifact", "", "T10B1 (*.t10b)")
            if path: self.start("Inspecting artifact…", lambda: {"path": path, "rows": read(path).rows, "cols": read(path).cols, "stored_bits_per_weight": read(path).stored_bits_per_weight})

        def benchmark_file(self):
            path, _ = QFileDialog.getOpenFileName(self, "Select T10B1 artifact", "", "T10B1 (*.t10b)")
            if path: self.start("Benchmarking packed matvec…", lambda: benchmark(read(path), repeats=25, seed=7))

    app = QApplication([])
    window = Window(); window.show()
    return app.exec()


def main() -> int:
    parser = argparse.ArgumentParser(description="10ary-One-Bit PySide6 control panel")
    parser.add_argument("--demo", action="store_true", help="run the synthetic demo without opening a window")
    parser.add_argument("--repeats", type=int, default=25)
    args = parser.parse_args()
    if args.demo:
        print(json.dumps(run_demo(args.repeats), indent=2, sort_keys=True)); return 0
    return launch()


if __name__ == "__main__":
    raise SystemExit(main())
