"""
Build a calibration set (100–200 images) per module for INT8 PTQ.

Rules:
    - Use training set images only (never val/test)
    - Mix doc types: aadhaar, pan, cheque, invoice, marksheet, generic
    - Mix conditions: indoor, outdoor, low-light, tilted

Usage:
    python quantization/calibration/build_calib_set.py \
        --module corner_detector \
        --source datasets/collected/train \
        --n 150
"""
from __future__ import annotations

import argparse
import random
import shutil
from pathlib import Path


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--module", required=True,
                   choices=["corner_detector", "doc_classifier",
                            "text_detection", "text_recognition"])
    p.add_argument("--source", required=True)
    p.add_argument("--n", type=int, default=150)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    out = Path(f"datasets/calib/{args.module}")
    out.mkdir(parents=True, exist_ok=True)

    imgs = list(Path(args.source).rglob("*.jpg")) + list(Path(args.source).rglob("*.png"))
    random.seed(args.seed)
    picked = random.sample(imgs, min(args.n, len(imgs)))

    for i, src in enumerate(picked):
        shutil.copy(src, out / f"calib_{i:04d}{src.suffix}")
    print(f"[{args.module}] {len(picked)} calibration images → {out}")


if __name__ == "__main__":
    main()
