"""
Dataset for corner detector.

Expects a CSV manifest with columns:
    image_path, tl_x, tl_y, tr_x, tr_y, br_x, br_y, bl_x, bl_y

Coordinates are in absolute pixels (we normalize to the resized image).
"""
from __future__ import annotations

import csv
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


class CornerDataset(Dataset):
    def __init__(
        self,
        manifest_path: str,
        img_size: int = 256,
        heatmap_size: int = 64,
        sigma: float = 2.0,
        augment: bool = True,
    ) -> None:
        self.img_size = img_size
        self.heatmap_size = heatmap_size
        self.sigma = sigma
        self.augment = augment
        with open(manifest_path) as f:
            self.rows = list(csv.DictReader(f))

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        row = self.rows[idx]
        img = cv2.imread(row["image_path"])
        assert img is not None, f"missing image: {row['image_path']}"
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w = img.shape[:2]

        corners = np.array(
            [
                [float(row["tl_x"]), float(row["tl_y"])],
                [float(row["tr_x"]), float(row["tr_y"])],
                [float(row["br_x"]), float(row["br_y"])],
                [float(row["bl_x"]), float(row["bl_y"])],
            ],
            dtype=np.float32,
        )

        img = cv2.resize(img, (self.img_size, self.img_size))
        corners[:, 0] *= self.heatmap_size / w
        corners[:, 1] *= self.heatmap_size / h

        # TODO(week 1): albumentations for augment=True (perspective, blur, color jitter)

        img_t = torch.from_numpy(img).float().permute(2, 0, 1) / 255.0
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        img_t = (img_t - mean) / std

        heatmaps = np.stack(
            [_gaussian_heatmap(self.heatmap_size, c, self.sigma) for c in corners]
        )
        return img_t, torch.from_numpy(heatmaps).float()


def _gaussian_heatmap(size: int, center: np.ndarray, sigma: float) -> np.ndarray:
    y, x = np.mgrid[0:size, 0:size]
    cx, cy = float(center[0]), float(center[1])
    return np.exp(-((x - cx) ** 2 + (y - cy) ** 2) / (2 * sigma**2)).astype(np.float32)
