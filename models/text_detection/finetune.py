"""
Text detection fine-tuning — DBNet-MobileNetV3-large.

Strategy:
    1. Start from PP-OCRv4_mobile_det Paddle checkpoint
    2. Convert Paddle → PyTorch via paddle2torch (or re-implement in torch + load weights)
    3. Fine-tune on ICDAR-MLT Indic split + our synthetic Indic lines
    4. Export to ONNX → TFLite INT8 (or FP16 if accuracy drops)

NOTE: This is the most complex module. Prefer reusing PaddleOCR's training loop
and only migrating to our codebase at ONNX export time. See:
    https://github.com/PaddlePaddle/PaddleOCR/blob/main/configs/det/ch_PP-OCRv4/ch_PP-OCRv4_det_cml.yml
"""
from __future__ import annotations

import argparse


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="models/text_detection/config.yml")
    args = p.parse_args()

    # TODO(week 2):
    #   1. Load PP-OCRv4_mobile_det weights (Paddle format)
    #   2. Build PyTorch DBNet model
    #   3. Implement DB loss (binary + threshold + approx binary maps)
    #      — formulas in paper 1911.08947v2, Eq. 2 (differentiable binarization)
    #   4. Train on ICDAR-MLT Indic + synth data
    #   5. Validate F-measure @ IoU 0.5 on held-out set
    raise NotImplementedError(
        "Fine-tuning stub. Implement in week 2 using paper 1911.08947v2."
    )


if __name__ == "__main__":
    main()
