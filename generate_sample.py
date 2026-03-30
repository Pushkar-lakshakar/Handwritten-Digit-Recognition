"""
generate_sample.py — Export sample digit images from the MNIST test set.

This script creates 10 sample images (one per digit class) in the
sample_images/ directory so evaluators can immediately test predict.py
without providing their own images.

Usage:
    python generate_sample.py
"""

import os
import torch
from torchvision import datasets
from PIL import Image

from utils import get_eval_transform


def main():
    output_dir = "sample_images"
    os.makedirs(output_dir, exist_ok=True)

    print("📥  Loading MNIST test set (auto-downloads if needed)...")
    test_dataset = datasets.MNIST(
        root="data",
        train=False,
        download=True,
        transform=None,     # raw PIL images
    )

    # Collect one example per digit (0–9)
    saved   = {}
    saved_count = 0

    for img, label in test_dataset:
        if label not in saved:
            saved[label] = img
            saved_count += 1
        if saved_count == 10:
            break

    # Save each image as PNG
    for digit in sorted(saved.keys()):
        img = saved[digit]
        filename = os.path.join(output_dir, f"digit_{digit}.png")
        img.save(filename)
        print(f"  ✅  Saved {filename}  (label: {digit})")

    print(f"\n🎉  {saved_count} sample images saved to '{output_dir}/'")
    print("     Run: python predict.py --image sample_images/digit_7.png")


if __name__ == "__main__":
    main()
