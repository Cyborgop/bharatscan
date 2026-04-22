"""
End-to-end pipeline evaluation on held-out docs.

Runs: corner → warp → classify → detect → recognize → extract
Reports: corner IoU, classifier accuracy, text F1, CER, field extraction F1.
"""
from __future__ import annotations

import argparse


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--test_manifest", required=True)
    p.add_argument("--models_dir", default="app/assets/models")
    args = p.parse_args()

    # TODO(week 4): load all TFLites via tflite-runtime, run on host,
    # compute full metrics table for the paper.
    raise NotImplementedError("Implement in week 4.")


if __name__ == "__main__":
    main()
