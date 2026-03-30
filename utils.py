"""
utils.py — Preprocessing and helper utilities for Handwritten Digit Recognition.

Provides:
  - Image transformation pipelines for training and inference
  - Custom image loader for external (non-MNIST) images
  - Accuracy computation helper
"""

import os
from PIL import Image
import torch
from torchvision import transforms


# ──────────────────────────────────────────────
# Transformation Pipelines
# ──────────────────────────────────────────────

def get_train_transform():
    """
    Returns the torchvision transform pipeline used during training.
    Applies slight augmentation to improve generalisation.
    """
    return transforms.Compose([
        transforms.RandomRotation(degrees=10),          # mild rotation augmentation
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),  # slight shift
        transforms.ToTensor(),                           # [0,255] → [0.0,1.0]
        transforms.Normalize((0.1307,), (0.3081,))      # MNIST mean & std
    ])


def get_eval_transform():
    """
    Returns the transform pipeline used for evaluation and inference.
    No augmentation — only normalisation.
    """
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])


def get_predict_transform():
    """
    Returns the transform pipeline used when predicting from an external image file.
    Converts to grayscale, resizes to 28×28, then normalises.
    """
    return transforms.Compose([
        transforms.Grayscale(num_output_channels=1),    # ensure single channel
        transforms.Resize((28, 28)),                    # MNIST input size
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])


# ──────────────────────────────────────────────
# Image Loader
# ──────────────────────────────────────────────

def load_image(image_path: str) -> torch.Tensor:
    """
    Loads an image from disk and prepares it for model inference.

    Args:
        image_path (str): Path to the input image (PNG, JPG, etc.)

    Returns:
        torch.Tensor: Shape (1, 1, 28, 28) — batch of one image.

    Raises:
        FileNotFoundError: If the image file does not exist.
        ValueError: If the image cannot be opened.
    """
    if not os.path.isfile(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")

    try:
        img = Image.open(image_path).convert("L")   # force grayscale
    except Exception as e:
        raise ValueError(f"Could not open image '{image_path}': {e}")

    transform = get_predict_transform()
    tensor = transform(img)          # shape: (1, 28, 28)
    tensor = tensor.unsqueeze(0)     # add batch dim → (1, 1, 28, 28)
    return tensor


# ──────────────────────────────────────────────
# Metric Helper
# ──────────────────────────────────────────────

def compute_accuracy(outputs: torch.Tensor, labels: torch.Tensor) -> float:
    """
    Computes top-1 classification accuracy for a batch.

    Args:
        outputs (torch.Tensor): Raw logits from the model, shape (N, num_classes).
        labels  (torch.Tensor): Ground truth class indices, shape (N,).

    Returns:
        float: Accuracy as a percentage (0.0 – 100.0).
    """
    _, predicted = torch.max(outputs, dim=1)
    correct = (predicted == labels).sum().item()
    return 100.0 * correct / labels.size(0)


# ──────────────────────────────────────────────
# Model path helper
# ──────────────────────────────────────────────

MODEL_DIR  = "model"
MODEL_FILE = os.path.join(MODEL_DIR, "cnn_model.pth")


def ensure_model_dir():
    """Creates the model/ directory if it does not already exist."""
    os.makedirs(MODEL_DIR, exist_ok=True)
