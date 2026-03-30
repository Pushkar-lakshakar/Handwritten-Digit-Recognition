"""
train.py — Train a CNN on MNIST and save the model weights.

Usage:
    python train.py [--epochs N] [--batch-size N] [--lr FLOAT] [--no-cuda]

Example:
    python train.py --epochs 10
"""

import argparse
import os
import time

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets

from model_cnn import DigitCNN
from utils import (
    get_train_transform,
    get_eval_transform,
    compute_accuracy,
    ensure_model_dir,
    MODEL_FILE,
)


# ──────────────────────────────────────────────
# Argument Parser
# ──────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="Train CNN for Handwritten Digit Recognition (MNIST)"
    )
    parser.add_argument("--epochs",     type=int,   default=10,    help="Number of training epochs (default: 10)")
    parser.add_argument("--batch-size", type=int,   default=64,    help="Mini-batch size (default: 64)")
    parser.add_argument("--lr",         type=float, default=0.001, help="Learning rate (default: 0.001)")
    parser.add_argument("--no-cuda",    action="store_true",        help="Disable GPU even if available")
    return parser.parse_args()


# ──────────────────────────────────────────────
# Data Loaders
# ──────────────────────────────────────────────

def get_loaders(batch_size: int):
    """Downloads MNIST (if needed) and returns train/test DataLoaders."""
    os.makedirs("data", exist_ok=True)

    print("📥  Loading MNIST dataset (auto-downloads on first run)...")

    train_dataset = datasets.MNIST(
        root="data",
        train=True,
        download=True,
        transform=get_train_transform(),
    )
    test_dataset = datasets.MNIST(
        root="data",
        train=False,
        download=True,
        transform=get_eval_transform(),
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=0
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=256, shuffle=False, num_workers=0
    )

    print(f"    Training samples : {len(train_dataset):,}")
    print(f"    Test samples     : {len(test_dataset):,}")
    return train_loader, test_loader


# ──────────────────────────────────────────────
# Training Loop
# ──────────────────────────────────────────────

def train_one_epoch(model, loader, criterion, optimizer, device, epoch, total_epochs):
    model.train()
    total_loss = 0.0
    total_acc  = 0.0
    batches    = len(loader)

    for batch_idx, (images, labels) in enumerate(loader):
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss    = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_acc  += compute_accuracy(outputs.detach(), labels)

        # Progress bar (every 100 batches)
        if (batch_idx + 1) % 100 == 0 or (batch_idx + 1) == batches:
            pct = 100.0 * (batch_idx + 1) / batches
            bar = ("█" * int(pct / 5)).ljust(20)
            print(
                f"\r  Epoch [{epoch}/{total_epochs}]  [{bar}] "
                f"{pct:5.1f}%  loss: {total_loss/(batch_idx+1):.4f}",
                end="",
                flush=True,
            )

    print()  # newline after progress bar
    return total_loss / batches, total_acc / batches


# ──────────────────────────────────────────────
# Evaluation Loop
# ──────────────────────────────────────────────

def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    total_acc  = 0.0
    batches    = len(loader)

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            total_loss += criterion(outputs, labels).item()
            total_acc  += compute_accuracy(outputs, labels)

    return total_loss / batches, total_acc / batches


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────

def main():
    args = parse_args()

    # ── Device selection ─────────────────────
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device   = torch.device("cuda" if use_cuda else "cpu")
    print(f"\n🖥️  Device : {device}")
    if use_cuda:
        print(f"    GPU    : {torch.cuda.get_device_name(0)}")

    # ── Data ──────────────────────────────────
    train_loader, test_loader = get_loaders(args.batch_size)

    # ── Model ─────────────────────────────────
    model     = DigitCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

    print(f"\n🏗️  Model   : DigitCNN")
    print(f"    Params  : {sum(p.numel() for p in model.parameters()):,}")
    print(f"    Epochs  : {args.epochs}")
    print(f"    Batch   : {args.batch_size}")
    print(f"    LR      : {args.lr}\n")

    best_acc = 0.0
    start    = time.time()

    # ── Training ──────────────────────────────
    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device, epoch, args.epochs
        )
        val_loss, val_acc = evaluate(model, test_loader, criterion, device)
        scheduler.step()

        marker = " ← best" if val_acc > best_acc else ""
        if val_acc > best_acc:
            best_acc = val_acc

        print(
            f"  train  loss: {train_loss:.4f}  acc: {train_acc:.2f}%  |  "
            f"val  loss: {val_loss:.4f}  acc: {val_acc:.2f}%{marker}"
        )

    elapsed = time.time() - start
    print(f"\n⏱️  Training complete in {elapsed/60:.1f} min")
    print(f"✅  Best validation accuracy : {best_acc:.2f}%")

    # ── Save model ────────────────────────────
    ensure_model_dir()
    torch.save(model.state_dict(), MODEL_FILE)
    print(f"💾  Model saved → {MODEL_FILE}\n")


if __name__ == "__main__":
    main()
