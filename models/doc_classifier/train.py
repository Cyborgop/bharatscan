"""
Doc classifier training — ImageFolder style.

Layout:
    datasets/collected/doc_classifier/
        train/
            aadhaar/*.jpg
            pan/*.jpg
            cheque/*.jpg
            gst_invoice/*.jpg
            marksheet/*.jpg
            generic/*.jpg
        val/
            ...

Expected runtime on RTX 2080 Ti: ~2 hours for 30 epochs on 6k images.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder

from model import DOC_CLASSES, DocClassifier


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", required=True,
                   help="Contains train/ and val/ with ImageFolder layout")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--out", default="runs/doc_classifier")
    args = p.parse_args()

    Path(args.out).mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    train_tf = transforms.Compose([
        transforms.Resize(256),
        transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),
        transforms.ColorJitter(0.2, 0.2, 0.2),
        transforms.RandomRotation(5),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    val_tf = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    train_ds = ImageFolder(f"{args.data_dir}/train", transform=train_tf)
    val_ds = ImageFolder(f"{args.data_dir}/val", transform=val_tf)
    assert train_ds.classes == DOC_CLASSES, \
        f"Class order mismatch. Got {train_ds.classes}, expected {DOC_CLASSES}"

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, num_workers=2)

    model = DocClassifier(pretrained=True).to(device)
    optim = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=args.epochs)
    loss_fn = nn.CrossEntropyLoss(label_smoothing=0.1)

    best_acc = 0.0
    for epoch in range(args.epochs):
        model.train()
        for img, y in train_loader:
            img, y = img.to(device), y.to(device)
            loss = loss_fn(model(img), y)
            optim.zero_grad(); loss.backward(); optim.step()

        model.eval()
        correct = total = 0
        with torch.no_grad():
            for img, y in val_loader:
                img, y = img.to(device), y.to(device)
                pred = model(img).argmax(1)
                correct += (pred == y).sum().item()
                total += y.size(0)
        acc = correct / total
        sched.step()
        print(f"[{epoch+1:3d}/{args.epochs}] val_acc={acc:.4f}")

        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), Path(args.out) / "best.pt")
            print(f"  → saved best.pt (acc={best_acc:.4f})")


if __name__ == "__main__":
    main()
