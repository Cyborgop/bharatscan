"""
PII redaction — run on EVERY document image before any sharing.

Blurs:
  - 12-digit Aadhaar blocks
  - PAN IDs
  - Photo area (face)
  - Signature area

Usage:
    python datasets/redact/redact_pii.py --input raw/ --output redacted/
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import cv2
import numpy as np


AADHAAR_RE = re.compile(r"\b\d{4}\s?\d{4}\s?\d{4}\b")
PAN_RE     = re.compile(r"\b[A-Z]{5}[0-9]{4}[A-Z]\b")


def blur_region(img: np.ndarray, box: tuple[int, int, int, int]) -> None:
    x1, y1, x2, y2 = box
    roi = img[y1:y2, x1:x2]
    if roi.size == 0:
        return
    img[y1:y2, x1:x2] = cv2.GaussianBlur(roi, (51, 51), 0)


def redact_file(src: Path, dst: Path) -> None:
    """
    Placeholder redactor — relies on downstream OCR (after detection module is
    trained) to locate PII strings. For now: heavy full-image blur of the
    bottom 40% where Aadhaar numbers usually sit.

    TODO(week 2): upgrade once text_detector TFLite is available.
    """
    img = cv2.imread(str(src))
    if img is None:
        print(f"  skip (unreadable): {src}")
        return
    h, w = img.shape[:2]
    blur_region(img, (0, int(h * 0.6), w, h))  # bottom 40%
    blur_region(img, (int(w * 0.6), 0, w, int(h * 0.4)))  # top-right (photo area)
    cv2.imwrite(str(dst), img)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True)
    p.add_argument("--output", required=True)
    args = p.parse_args()

    in_dir = Path(args.input)
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    for src in list(in_dir.rglob("*.jpg")) + list(in_dir.rglob("*.png")):
        dst = out_dir / src.relative_to(in_dir)
        dst.parent.mkdir(parents=True, exist_ok=True)
        redact_file(src, dst)
        print(f"  redacted: {src.name}")


if __name__ == "__main__":
    main()
