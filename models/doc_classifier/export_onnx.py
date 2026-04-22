"""Export doc classifier to ONNX."""
from __future__ import annotations

import argparse

import onnx
import torch
from onnxsim import simplify

from model import DocClassifier


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--out", required=True)
    args = p.parse_args()

    model = DocClassifier(pretrained=False).eval()
    model.load_state_dict(torch.load(args.ckpt, map_location="cpu"))

    dummy = torch.randn(1, 3, 224, 224)
    torch.onnx.export(
        model, dummy, args.out,
        input_names=["input"], output_names=["logits"],
        opset_version=13, do_constant_folding=True,
    )
    simplified, ok = simplify(onnx.load(args.out))
    if ok:
        onnx.save(simplified, args.out)
    print(f"[saved] {args.out}")


if __name__ == "__main__":
    main()
