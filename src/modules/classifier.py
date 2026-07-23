"""Final classification head (paper Sec IV-e).

Why: map the fused multimodal representation h to 2-class logits.

    y_hat = Dense(h) in R^2

Tensors:
    in:  h [B, fused_dim]
    out: logits [B, num_classes]

Hidden layout (user decision): fused_dim -> 128 -> num_classes, ReLU + dropout
0.1. The 128-unit hidden matches the paper's PROCESS-2 BERT baseline.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class Classifier(nn.Module):
    def __init__(self, input_dim: int, hidden: int = 128, num_classes: int = 2,
                 dropout: float = 0.1) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(input_dim, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, num_classes),
        )

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        # h: [B, input_dim] -> logits: [B, num_classes]
        return self.net(h)
