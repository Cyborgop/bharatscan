"""
Generate synthetic text lines for OCR recognition training.

Uses trdg (text-recognition-data-generator) with Noto Sans fonts per script.
Produces 200k lines per script by default.

Usage:
    python datasets/synth/generate_indic_lines.py \
        --script devanagari --count 200000 --out datasets/synth/out/devanagari
"""
from __future__ import annotations

import argparse
import subprocess
from pathlib import Path


# Noto Sans font paths (download from https://fonts.google.com/noto)
FONTS = {
    "devanagari": "fonts/NotoSansDevanagari-Regular.ttf",
    "bengali":    "fonts/NotoSansBengali-Regular.ttf",
    "tamil":      "fonts/NotoSansTamil-Regular.ttf",
    "telugu":     "fonts/NotoSansTelugu-Regular.ttf",
    "kannada":    "fonts/NotoSansKannada-Regular.ttf",
    "malayalam":  "fonts/NotoSansMalayalam-Regular.ttf",
    "gurmukhi":   "fonts/NotoSansGurmukhi-Regular.ttf",
    "odia":       "fonts/NotoSansOriya-Regular.ttf",
    "gujarati":   "fonts/NotoSansGujarati-Regular.ttf",
    "arabic":     "fonts/NotoNaskhArabic-Regular.ttf",
    "latin":      "fonts/NotoSans-Regular.ttf",
}

# Corpus source: Wikipedia dumps per language, deduplicated + cleaned
CORPUS = {s: f"corpora/{s}_wiki.txt" for s in FONTS}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--script", required=True, choices=list(FONTS))
    p.add_argument("--count", type=int, default=200_000)
    p.add_argument("--out", required=True)
    args = p.parse_args()

    Path(args.out).mkdir(parents=True, exist_ok=True)

    # trdg CLI — see https://github.com/Belval/TextRecognitionDataGenerator
    cmd = [
        "trdg",
        "-c", str(args.count),
        "-l", _trdg_lang(args.script),
        "-w", "1",          # one word line length variations via -rs below
        "-rs",              # random string mode
        "-rk",              # random skew
        "-k", "5",          # skew ±5°
        "-bl", "1", "-rbl", # random blur
        "-b", "3",          # random background
        "-na", "2",         # name images by content
        "-ft", FONTS[args.script],
        "--output_dir", args.out,
    ]
    print("Running:", " ".join(cmd))
    subprocess.run(cmd, check=True)


def _trdg_lang(script: str) -> str:
    # trdg language codes
    return {
        "devanagari": "hi", "bengali": "bn", "tamil": "ta", "telugu": "te",
        "kannada": "kn", "malayalam": "ml", "gurmukhi": "pa", "odia": "or",
        "gujarati": "gu", "arabic": "ar", "latin": "en",
    }[script]


if __name__ == "__main__":
    main()
