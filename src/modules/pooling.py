"""Frame-level pooling strategies (paper Sec IV-b + Ablation "Pooling").

Why: HuBERT emits a variable-length sequence of frame embeddings per 10s chunk;
we need one fixed vector per chunk. The paper uses Attentive Statistics Pooling
(ASP) to emphasize cognitively-informative frames; the ablation also compares
mean and max pooling.

Tensors (all poolings):
    in:  frames [N, L, D], frame_mask [N, L] (1 = valid, 0 = padded)
    out: pooled [N, out_dim]

ASP equations (paper Sec IV-b, after Okabe et al. 2018 [31]):
    e_t     = v^T f(W h_t + b) + k          # scalar attention score per frame
    alpha_t = softmax_t(e_t)
    mu~     = sum_t alpha_t h_t              # weighted mean
    sigma~  = sqrt( sum_t alpha_t (h_t . h_t) - mu~ . mu~ )   # weighted std
    f_a     = [mu~, sigma~]                  # concatenation -> out_dim = 2D
"""

from __future__ import annotations

import torch
import torch.nn as nn


def _masked_fill_scores(scores: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Set scores at padded positions to -inf so softmax ignores them.

    scores: [N, L, 1]; mask: [N, L] (1 valid / 0 pad).
    """
    mask = mask.unsqueeze(-1).bool()  # [N, L, 1]
    return scores.masked_fill(~mask, float("-inf"))


class AttentiveStatisticsPooling(nn.Module):
    def __init__(self, input_dim: int, bottleneck: int = 128, eps: float = 1e-6) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.eps = eps
        # W h_t + b (project to bottleneck), then f = tanh, then v^T(.) + k.
        self.W = nn.Linear(input_dim, bottleneck)
        self.activation = nn.Tanh()
        self.v = nn.Linear(bottleneck, 1)  # includes bias term k

    @property
    def output_dim(self) -> int:
        return 2 * self.input_dim  # [mu~, sigma~]

    def forward(self, frames: torch.Tensor, frame_mask: torch.Tensor) -> torch.Tensor:
        # frames: [N, L, D]; frame_mask: [N, L]
        scores = self.v(self.activation(self.W(frames)))  # [N, L, 1]
        scores = _masked_fill_scores(scores, frame_mask)  # [N, L, 1]
        alpha = torch.softmax(scores, dim=1)  # [N, L, 1]

        mean = torch.sum(alpha * frames, dim=1)  # [N, D]
        mean_sq = torch.sum(alpha * frames * frames, dim=1)  # [N, D]
        var = torch.clamp(mean_sq - mean * mean, min=self.eps)  # [N, D]
        std = torch.sqrt(var)  # [N, D]
        return torch.cat([mean, std], dim=1)  # [N, 2D]


class MeanPooling(nn.Module):
    """Masked mean over frames (ablation baseline). out_dim = D."""

    def __init__(self, input_dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.eps = eps

    @property
    def output_dim(self) -> int:
        return self.input_dim

    def forward(self, frames: torch.Tensor, frame_mask: torch.Tensor) -> torch.Tensor:
        mask = frame_mask.unsqueeze(-1).to(frames.dtype)  # [N, L, 1]
        summed = torch.sum(frames * mask, dim=1)  # [N, D]
        counts = torch.clamp(mask.sum(dim=1), min=self.eps)  # [N, 1]
        return summed / counts  # [N, D]


class MaxPooling(nn.Module):
    """Masked max over frames (ablation baseline). out_dim = D."""

    def __init__(self, input_dim: int) -> None:
        super().__init__()
        self.input_dim = input_dim

    @property
    def output_dim(self) -> int:
        return self.input_dim

    def forward(self, frames: torch.Tensor, frame_mask: torch.Tensor) -> torch.Tensor:
        mask = frame_mask.unsqueeze(-1).bool()  # [N, L, 1]
        neg_inf = torch.finfo(frames.dtype).min
        masked = frames.masked_fill(~mask, neg_inf)  # [N, L, D]
        return masked.max(dim=1).values  # [N, D]


def build_pooling(name: str, input_dim: int, asp_bottleneck: int = 128) -> nn.Module:
    name = name.lower()
    if name == "asp":
        return AttentiveStatisticsPooling(input_dim, bottleneck=asp_bottleneck)
    if name == "mean":
        return MeanPooling(input_dim)
    if name == "max":
        return MaxPooling(input_dim)
    raise ValueError(f"Unknown pooling '{name}'. Expected asp|mean|max.")
