"""
predict.py — Run digit inference on an image using the trained CNN model.

Usage:
    python predict.py --image sample_images/digit_7.png
    python predict.py --image path/to/your/image.png [--model model/cnn_model.pth]

Expected output:
    Predicted Digit : 7
    Confidence      : 0.9834
"""

import argparse
import sys

import torch
import torch.nn.functional as F

from model_cnn import DigitCNN
from utils import load_image, MODEL_FILE


# ──────────────────────────────────────────────
# Argument Parser
# ──────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="Predict a handwritten digit from an image file."
    )
    parser.add_argument(
        "--image",
        type=str,
        required=True,
        help="Path to the input image (PNG, JPG, BMP, etc.)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=MODEL_FILE,
        help=f"Path to saved model weights (default: {MODEL_FILE})",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=3,
        help="Show top-K predictions (default: 3)",
    )
    parser.add_argument(
        "--no-cuda",
        action="store_true",
        help="Disable GPU even if available",
    )
    return parser.parse_args()


# ──────────────────────────────────────────────
# Inference
# ──────────────────────────────────────────────

def predict(model, image_tensor, device, top_k=3):
    """
    Runs a forward pass and returns the predicted digit and confidence scores.

    Args:
        model        : Trained DigitCNN instance.
        image_tensor : Preprocessed image tensor of shape (1, 1, 28, 28).
        device       : torch.device to run inference on.
        top_k        : Number of top predictions to return.

    Returns:
        predicted_digit (int)  : Top-1 predicted class (0–9).
        confidence (float)     : Softmax probability of the top-1 prediction.
        top_k_results (list)   : [(digit, prob), ...] for the top-K predictions.
    """
    model.eval()
    image_tensor = image_tensor.to(device)

    with torch.no_grad():
        logits = model(image_tensor)                  # shape: (1, 10)
        probs  = F.softmax(logits, dim=1).squeeze()   # shape: (10,)

    top_probs, top_indices = torch.topk(probs, k=top_k)

    predicted_digit = top_indices[0].item()
    confidence      = top_probs[0].item()
    top_k_results   = [
        (top_indices[i].item(), top_probs[i].item()) for i in range(top_k)
    ]

    return predicted_digit, confidence, top_k_results


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────

def main():
    args = parse_args()

    # ── Device selection ─────────────────────
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device   = torch.device("cuda" if use_cuda else "cpu")

    # ── Load model ────────────────────────────
    import os
    if not os.path.isfile(args.model):
        print(
            f"[ERROR] Model file not found: '{args.model}'\n"
            f"        Please run 'python train.py' first to train and save the model.",
            file=sys.stderr,
        )
        sys.exit(1)

    model = DigitCNN().to(device)
    model.load_state_dict(torch.load(args.model, map_location=device))
    model.eval()

    # ── Load and preprocess image ─────────────
    try:
        image_tensor = load_image(args.image)
    except (FileNotFoundError, ValueError) as e:
        print(f"[ERROR] {e}", file=sys.stderr)
        sys.exit(1)

    # ── Run inference ─────────────────────────
    predicted_digit, confidence, top_k_results = predict(
        model, image_tensor, device, top_k=min(args.top_k, 10)
    )

    # ── Display results ───────────────────────
    print()
    print("═" * 38)
    print("  🔢  Handwritten Digit Recognition")
    print("═" * 38)
    print(f"  Image           : {args.image}")
    print(f"  Predicted Digit : {predicted_digit}")
    print(f"  Confidence      : {confidence:.4f}  ({confidence*100:.2f}%)")
    print()
    print(f"  Top-{len(top_k_results)} predictions:")
    for rank, (digit, prob) in enumerate(top_k_results, start=1):
        bar = "█" * int(prob * 20)
        print(f"    {rank}. Digit {digit}  {prob:.4f}  {bar}")
    print("═" * 38)
    print()


if __name__ == "__main__":
    main()
