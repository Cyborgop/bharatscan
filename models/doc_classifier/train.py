"""
Doc classifier training — ImageFolder style.

Layout:
    datasets/doc_classifier/
        train/
            aadhaar/*.jpg
            pan/*.jpg
        val/
            aadhaar/*.jpg
            pan/*.jpg
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
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--patience", type=int, default=5)
    p.add_argument("--out", default="runs/doc_classifier")
    args = p.parse_args()

    Path(args.out).mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("=" * 60)
    print(f"Using Device : {device}")
    if torch.cuda.is_available():
        print(f"GPU          : {torch.cuda.get_device_name(0)}")
        print(f"CUDA Version : {torch.version.cuda}")
        print(f"GPU Count    : {torch.cuda.device_count()}")
    else:
        print("Running on CPU")
    print("=" * 60)

    train_tf = transforms.Compose([
        transforms.Resize(256),
        transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),
        transforms.ColorJitter(0.2, 0.2, 0.2),
        transforms.RandomRotation(5),
        transforms.ToTensor(),
        transforms.Normalize(
            [0.485, 0.456, 0.406],
            [0.229, 0.224, 0.225]
        ),
    ])

    val_tf = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            [0.485, 0.456, 0.406],
            [0.229, 0.224, 0.225]
        ),
    ])

    train_ds = ImageFolder(f"{args.data_dir}/train", transform=train_tf)
    val_ds = ImageFolder(f"{args.data_dir}/val", transform=val_tf)

    assert train_ds.classes == DOC_CLASSES, \
        f"Class order mismatch. Got {train_ds.classes}, expected {DOC_CLASSES}"

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )

    model = DocClassifier(pretrained=True).to(device)

    if torch.cuda.is_available():
        print("\nModel loaded on GPU")
        print(f"GPU Memory Allocated : {torch.cuda.memory_allocated()/1024**2:.2f} MB")
        print(f"GPU Memory Reserved  : {torch.cuda.memory_reserved()/1024**2:.2f} MB\n")

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=1e-4,
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.epochs,
    )

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    best_acc = 0.0
    patience_counter = 0

    for epoch in range(args.epochs):

        # ---------------- TRAIN ----------------

        model.train()

        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for img, y in train_loader:

            img = img.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            optimizer.zero_grad()

            logits = model(img)

            loss = criterion(logits, y)

            loss.backward()

            optimizer.step()

            train_loss += loss.item() * img.size(0)

            pred = logits.argmax(1)

            train_correct += (pred == y).sum().item()
            train_total += y.size(0)

        train_loss /= len(train_ds)
        train_acc = train_correct / train_total

        # ---------------- VALIDATION ----------------

        model.eval()

        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():

            for img, y in val_loader:

                img = img.to(device, non_blocking=True)
                y = y.to(device, non_blocking=True)

                logits = model(img)

                loss = criterion(logits, y)

                val_loss += loss.item() * img.size(0)

                pred = logits.argmax(1)

                val_correct += (pred == y).sum().item()
                val_total += y.size(0)

        val_loss /= len(val_ds)
        val_acc = val_correct / val_total

        scheduler.step()

        print(
            f"[{epoch+1:02d}/{args.epochs}] "
            f"train_loss={train_loss:.4f} "
            f"train_acc={train_acc:.4f} "
            f"val_loss={val_loss:.4f} "
            f"val_acc={val_acc:.4f}"
        )

        # ---------------- SAVE BEST ----------------

        if val_acc > best_acc:

            best_acc = val_acc
            patience_counter = 0

            torch.save(
                model.state_dict(),
                Path(args.out) / "best.pt",
            )

            print(f"--> Saved best.pt (val_acc={best_acc:.4f})")

        else:
            patience_counter += 1

        # ---------------- EARLY STOPPING ----------------

        if patience_counter >= args.patience:
            print("\nEarly stopping triggered.")
            break

    print("\n==============================")
    print(f"Training Finished")
    print(f"Best Validation Accuracy : {best_acc:.4f}")
    print("==============================")


if __name__ == "__main__":
    torch.multiprocessing.freeze_support()
    main()