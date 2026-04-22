"""
Text recognition fine-tuning — PP-OCRv5 Devanagari mobile_rec.

Strategy:
    1. Start from PaddlePaddle/PP-OCRv5_mobile_rec (devanagari pretrained)
    2. Fine-tune on:
       - 200k synthetic Devanagari lines (trdg + Noto Sans Devanagari)
       - 5k real lines cropped from our collected Aadhaar/PAN/marksheet captures
    3. Target CER < 8% on held-out Indic test set
    4. Export: Paddle → ONNX → TFLite INT8 (fallback FP16 if CER degrades > 3%)

CRITICAL RISK: PP-OCRv5 INT8 accuracy on Indic scripts is undocumented.
Validate FP32 vs INT8 CER in week 3. Have FP16 fallback ready (+4 MB APK cost).
"""
from __future__ import annotations

import argparse


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--script", default="devanagari",
                   choices=["devanagari", "tamil", "telugu", "bengali",
                            "gurmukhi", "arabic", "kannada", "malayalam",
                            "odia", "gujarati", "latin"])
    p.add_argument("--base_ckpt", default="PaddlePaddle/PP-OCRv5_mobile_rec")
    p.add_argument("--train_manifest", required=True)
    p.add_argument("--val_manifest", required=True)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--out", default="runs/text_rec")
    args = p.parse_args()

    # TODO(week 3):
    #   1. Load PaddleOCR rec model + dict (ppocrv5_devanagari_dict.txt)
    #   2. Build PyTorch-side trainer OR use PaddleOCR training loop directly
    #   3. CTC loss, AdamW, cosine LR
    #   4. Track CER on val every epoch
    raise NotImplementedError(
        "Recognition fine-tune stub. Implement in week 3."
    )


if __name__ == "__main__":
    main()
