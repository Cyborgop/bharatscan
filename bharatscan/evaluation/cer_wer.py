"""
Character Error Rate (CER) and Word Error Rate (WER) — OCR accuracy metrics.

CER = edit_distance(pred, gt) / len(gt)
WER = word-level version of the above

Usage:
    python evaluation/cer_wer.py --predictions preds.tsv --ground_truth gt.tsv
    # each TSV has: image_path\ttext

Aggregate metric is averaged over lines, weighted by ground-truth length.
"""
from __future__ import annotations

import argparse
import csv

import editdistance


def load_tsv(path: str) -> dict[str, str]:
    out: dict[str, str] = {}
    with open(path, encoding="utf-8") as f:
        for row in csv.reader(f, delimiter="\t"):
            if len(row) >= 2:
                out[row[0]] = row[1]
    return out


def cer(pred: str, gt: str) -> tuple[int, int]:
    return editdistance.eval(pred, gt), max(len(gt), 1)


def wer(pred: str, gt: str) -> tuple[int, int]:
    return editdistance.eval(pred.split(), gt.split()), max(len(gt.split()), 1)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--predictions", required=True)
    p.add_argument("--ground_truth", required=True)
    args = p.parse_args()

    preds = load_tsv(args.predictions)
    gts = load_tsv(args.ground_truth)

    total_c_ed = total_c_len = 0
    total_w_ed = total_w_len = 0
    missing = 0

    for key, gt in gts.items():
        if key not in preds:
            missing += 1
            total_c_len += len(gt)
            total_c_ed += len(gt)  # count as full error
            total_w_len += len(gt.split())
            total_w_ed += len(gt.split())
            continue
        c_ed, c_len = cer(preds[key], gt)
        w_ed, w_len = wer(preds[key], gt)
        total_c_ed += c_ed;  total_c_len += c_len
        total_w_ed += w_ed;  total_w_len += w_len

    print(f"Lines evaluated: {len(gts)} (missing preds: {missing})")
    print(f"CER: {total_c_ed / total_c_len * 100:.2f}%")
    print(f"WER: {total_w_ed / total_w_len * 100:.2f}%")


if __name__ == "__main__":
    main()
