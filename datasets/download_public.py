"""
Download public datasets needed for training.

Currently supports:
  - MIDV-500 / MIDV-2020   (document corner detection)
  - ICDAR-MLT 2019          (multilingual scene text, Indic subset)
  - SynthText               (pretraining for DBNet)

Usage:
    python datasets/download_public.py --which midv2020 --out datasets/raw/midv2020
"""
from __future__ import annotations

import argparse
import subprocess
from pathlib import Path


SOURCES = {
    "midv500":  "https://github.com/fcakyon/midv500  # see repo for download script",
    "midv2020": "ftp://smartengines.com/midv-2020/",
    "icdar_mlt": "https://rrc.cvc.uab.es/?ch=15  # registration required",
    "synthtext": "https://www.robots.ox.ac.uk/~vgg/data/scenetext/",
    "iiit_ilst": "http://cvit.iiit.ac.in/research/projects/cvit-projects/iiit-indic-ocr",
}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--which", required=True, choices=list(SOURCES))
    p.add_argument("--out", required=True)
    args = p.parse_args()

    print(f"\n[{args.which}] source:\n  {SOURCES[args.which]}\n")
    print("Auto-download is not implemented for datasets with registration / ToS.")
    print("Follow the link above, download manually, and extract to:")
    print(f"  {args.out}")
    print("\nThen run the corresponding manifest builder under datasets/manifests/.")

    Path(args.out).mkdir(parents=True, exist_ok=True)


if __name__ == "__main__":
    main()
