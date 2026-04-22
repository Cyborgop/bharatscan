"""
Training loop for corner detector.

Usage:
    python models/corner_detector/train.py \
        --train_csv datasets/manifests/corner_train.csv \
        --val_csv   datasets/manifests/corner_val.csv \
        --epochs 50 --batch_size 32

Expected runtime on RTX 2080 Ti: ~6–8 hours for 50 epochs on 10k images.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from dataset import CornerDataset
from model import CornerDetector


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--train_csv", required=True)
    p.add_argument("--val_csv", required=True)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--out", default="runs/corner_detector")
    args = p.parse_args()

    Path(args.out).mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    train_ds = CornerDataset(args.train_csv, augment=True)
    val_ds = CornerDataset(args.val_csv, augment=False)
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True
    )
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, num_workers=2)

    model = CornerDetector(pretrained=True).to(device)
    optim = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=args.epochs)
    loss_fn = nn.MSELoss()

    best_val = float("inf")
    for epoch in range(args.epochs):
        model.train()
        train_loss = 0.0
        for img, heat in train_loader:
            img, heat = img.to(device), heat.to(device)
            pred = model(img)
            loss = loss_fn(pred, heat)
            optim.zero_grad()
            loss.backward()
            optim.step()
            train_loss += loss.item() * img.size(0)
        train_loss /= len(train_ds)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for img, heat in val_loader:
                img, heat = img.to(device), heat.to(device)
                pred = model(img)
                val_loss += loss_fn(pred, heat).item() * img.size(0)
        val_loss /= len(val_ds)
        sched.step()

        print(f"[{epoch+1:3d}/{args.epochs}] train={train_loss:.5f} val={val_loss:.5f}")

        if val_loss < best_val:
            best_val = val_loss
            torch.save(model.state_dict(), Path(args.out) / "best.pt")
            print(f"  → saved best.pt (val={best_val:.5f})")


if __name__ == "__main__":
    main()
