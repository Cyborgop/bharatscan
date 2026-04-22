"""
Shared ONNX → TFLite pipeline (FP32 / FP16 / INT8).

Port of MCUDetector's quantization flow. Takes any ONNX file + a directory
of calibration images and produces three TFLite files.

Usage:
    python quantization/onnx_to_tflite.py \
        --onnx       runs/corner_detector/model_fp32.onnx \
        --calib_dir  datasets/calib/corner_detector/ \
        --out_dir    runs/corner_detector/tflite \
        --img_size   256
"""
from __future__ import annotations

import argparse
import os
import subprocess
from pathlib import Path

import numpy as np
import tensorflow as tf


def build_saved_model(onnx_path: str, out_dir: str) -> str:
    """Convert ONNX → TF SavedModel via onnx2tf CLI."""
    subprocess.run(["onnx2tf", "-i", onnx_path, "-o", out_dir], check=True)
    for root, _, files in os.walk(out_dir):
        if "saved_model.pb" in files:
            return root
    raise RuntimeError(f"saved_model.pb not found inside {out_dir}")


def representative_dataset_fn(calib_dir: str, img_size: int):
    """Yields calibration tensors for INT8 PTQ."""
    import cv2

    imgs = sorted(Path(calib_dir).glob("*.jpg")) + sorted(Path(calib_dir).glob("*.png"))
    assert len(imgs) >= 50, f"Need >=50 calib images, got {len(imgs)}"

    def gen():
        for p in imgs[:200]:
            im = cv2.imread(str(p))
            im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
            im = cv2.resize(im, (img_size, img_size)).astype(np.float32) / 255.0
            mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
            std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
            im = (im - mean) / std
            yield [im[None, ...]]

    return gen


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--onnx", required=True)
    p.add_argument("--calib_dir", required=True)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--img_size", type=int, default=256)
    args = p.parse_args()

    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    saved_model = build_saved_model(args.onnx, f"{args.out_dir}/tf_savedmodel")

    # FP32
    conv = tf.lite.TFLiteConverter.from_saved_model(saved_model)
    Path(f"{args.out_dir}/model_fp32.tflite").write_bytes(conv.convert())

    # FP16
    conv = tf.lite.TFLiteConverter.from_saved_model(saved_model)
    conv.optimizations = [tf.lite.Optimize.DEFAULT]
    conv.target_spec.supported_types = [tf.float16]
    Path(f"{args.out_dir}/model_fp16.tflite").write_bytes(conv.convert())

    # INT8 full-integer
    conv = tf.lite.TFLiteConverter.from_saved_model(saved_model)
    conv.optimizations = [tf.lite.Optimize.DEFAULT]
    conv.representative_dataset = representative_dataset_fn(args.calib_dir, args.img_size)
    conv.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    conv.inference_input_type = tf.uint8
    conv.inference_output_type = tf.float32
    Path(f"{args.out_dir}/model_int8.tflite").write_bytes(conv.convert())

    for f in ["model_fp32.tflite", "model_fp16.tflite", "model_int8.tflite"]:
        sz = os.path.getsize(f"{args.out_dir}/{f}") / 1024
        print(f"  {f}: {sz:.1f} KB")


if __name__ == "__main__":
    main()
