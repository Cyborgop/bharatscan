"""Export text recognizer → ONNX."""
from __future__ import annotations

import argparse


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--img_h", type=int, default=48)
    p.add_argument("--img_w", type=int, default=320)
    args = p.parse_args()
    raise NotImplementedError("Implement in week 3.")


if __name__ == "__main__":
    main()
