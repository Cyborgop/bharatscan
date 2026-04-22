"""
Benchmark FP32 vs FP16 vs INT8 — size, latency, accuracy.

Produces a markdown table for the paper / supervisor report.

Usage:
    python quantization/benchmark.py --tflite_dir runs/corner_detector/tflite
"""
from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

import numpy as np
import tensorflow as tf


def bench(tflite_path: str, runs: int = 50) -> float:
    interp = tf.lite.Interpreter(model_path=tflite_path)
    interp.allocate_tensors()
    inp = interp.get_input_details()[0]
    dummy = np.random.rand(*inp["shape"]).astype(np.float32)
    if inp["dtype"] == np.uint8:
        dummy = (dummy * 255).astype(np.uint8)
    for _ in range(10):
        interp.set_tensor(inp["index"], dummy)
        interp.invoke()
    times = []
    for _ in range(runs):
        t0 = time.perf_counter()
        interp.set_tensor(inp["index"], dummy)
        interp.invoke()
        times.append((time.perf_counter() - t0) * 1000)
    return float(np.median(times))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--tflite_dir", required=True)
    args = p.parse_args()

    rows = []
    for name in ["model_fp32.tflite", "model_fp16.tflite", "model_int8.tflite"]:
        path = Path(args.tflite_dir) / name
        if not path.exists():
            continue
        sz_kb = os.path.getsize(path) / 1024
        lat_ms = bench(str(path))
        rows.append((name, sz_kb, lat_ms))

    print("\n| Model | Size (KB) | Latency (ms, desktop CPU) |")
    print("|---|---:|---:|")
    for name, sz, lat in rows:
        print(f"| {name} | {sz:.1f} | {lat:.1f} |")
    print("\nNote: desktop CPU latency ≠ mobile NNAPI latency. Use")
    print("      scripts/measure_device_latency.sh for real device numbers.")


if __name__ == "__main__":
    main()
