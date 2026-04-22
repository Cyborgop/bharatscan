"""
Export corner detector → ONNX, ready for onnx2tf → TFLite.

Usage:
    python models/corner_detector/export_onnx.py \
        --ckpt runs/corner_detector/best.pt \
        --out  runs/corner_detector/model_fp32.onnx
"""
from __future__ import annotations

import argparse

import torch
import onnx
from onnxsim import simplify

from model import CornerDetector


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--img_size", type=int, default=256)
    args = p.parse_args()

    model = CornerDetector(pretrained=False).eval()
    model.load_state_dict(torch.load(args.ckpt, map_location="cpu"))

    dummy = torch.randn(1, 3, args.img_size, args.img_size)
    torch.onnx.export(
        model,
        dummy,
        args.out,
        input_names=["input"],
        output_names=["heatmap"],
        opset_version=13,
        do_constant_folding=True,
    )

    simplified, ok = simplify(onnx.load(args.out))
    if ok:
        onnx.save(simplified, args.out)
        print(f"[saved] simplified ONNX → {args.out}")
    else:
        print(f"[saved] ONNX (simplify failed) → {args.out}")


if __name__ == "__main__":
    main()
