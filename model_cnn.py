"""
model_cnn.py — CNN architecture for Handwritten Digit Recognition.

Architecture (DigitCNN):
    ┌─────────────────────────────────────────────────────────────────┐
    │  Input        : (1, 28, 28)  — grayscale 28×28 pixels          │
    │  Conv Block 1 : Conv2d(1→32, 3×3) → BatchNorm → ReLU → MaxPool │
    │  Conv Block 2 : Conv2d(32→64, 3×3)→ BatchNorm → ReLU → MaxPool │
    │  Dropout      : p=0.25                                           │
    │  Flatten      : 64 × 7 × 7 = 3,136                             │
    │  FC1          : 3136 → 256  → ReLU → Dropout(0.5)              │
    │  FC2          : 256  → 128  → ReLU                              │
    │  Output       : 128  → 10   (logits, one per digit class)       │
    └─────────────────────────────────────────────────────────────────┘

Typical Accuracy: ~99.2% on MNIST test set after 10 epochs.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DigitCNN(nn.Module):
    """
    A compact Convolutional Neural Network for MNIST digit classification.

    The network uses two convolutional blocks followed by fully connected
    layers. Batch normalisation stabilises training and dropout regularises
    the model to prevent over-fitting.
    """

    def __init__(self, num_classes: int = 10, dropout_conv: float = 0.25, dropout_fc: float = 0.5):
        """
        Args:
            num_classes  (int)   : Number of output classes (10 for digits 0–9).
            dropout_conv (float) : Dropout rate after convolutional blocks.
            dropout_fc   (float) : Dropout rate in fully connected layers.
        """
        super(DigitCNN, self).__init__()

        # ── Convolutional Block 1 ─────────────────────────────────────
        # Input : (N, 1, 28, 28)
        # Output: (N, 32, 14, 14)  after MaxPool
        self.conv_block1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),   # 28×28 → 14×14
        )

        # ── Convolutional Block 2 ─────────────────────────────────────
        # Input : (N, 32, 14, 14)
        # Output: (N, 64, 7, 7)   after MaxPool
        self.conv_block2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),   # 14×14 → 7×7
        )

        # Dropout after convolutional feature extraction
        self.dropout_conv = nn.Dropout2d(p=dropout_conv)

        # ── Fully Connected Layers ────────────────────────────────────
        # 64 channels × 7 × 7 spatial = 3,136 features
        self.fc_block = nn.Sequential(
            nn.Linear(64 * 7 * 7, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_fc),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
        )

        # ── Output Layer ──────────────────────────────────────────────
        self.classifier = nn.Linear(128, num_classes)

        # ── Weight Initialisation ─────────────────────────────────────
        self._init_weights()

    def _init_weights(self):
        """Applies He (Kaiming) initialisation to Conv and Linear layers."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x (torch.Tensor): Input tensor of shape (N, 1, 28, 28).

        Returns:
            torch.Tensor: Logits of shape (N, 10).
        """
        x = self.conv_block1(x)          # → (N, 32, 14, 14)
        x = self.conv_block2(x)          # → (N, 64, 7, 7)
        x = self.dropout_conv(x)

        x = x.view(x.size(0), -1)        # flatten → (N, 3136)

        x = self.fc_block(x)             # → (N, 128)
        logits = self.classifier(x)      # → (N, 10)

        return logits


# ──────────────────────────────────────────────
# Quick sanity check
# ──────────────────────────────────────────────

if __name__ == "__main__":
    model = DigitCNN()
    dummy = torch.randn(4, 1, 28, 28)    # batch of 4 fake images
    output = model(dummy)
    print(f"Model output shape : {output.shape}")   # expected: torch.Size([4, 10])
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters   : {total_params:,}")
    print("✅  Model forward pass OK")
