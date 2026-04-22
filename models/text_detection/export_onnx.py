"""Export DBNet → ONNX, ready for onnx2tf → TFLite."""
from __future__ import annotations

import argparse


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--img_size", type=int, default=640)
    args = p.parse_args()

    # TODO(week 2): same pattern as corner_detector/export_onnx.py
    raise NotImplementedError


if __name__ == "__main__":
    main()
