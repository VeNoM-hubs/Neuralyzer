"""Multimodal fusion modules (paper Sec IV-d + Ablation "Fusion").

Why: combine speech embedding f_a and text embedding f_t into one vector for
classification. Proposed method is AT-Fusion; the ablation also compares
Concatenation, GMU, MUTAN, MFB, MFH, and BLOCK.

Every fusion module:
    in:  f_a [B, Da], f_t [B, Dt]
    out: h   [B, output_dim]

Proposed AT-Fusion (paper Sec IV-d):
    f_cat = Concat(f_a, f_t) in R^{D x 2}
    alpha = softmax( w^T tanh(W f_cat) ) in R^{1x2}
    h     = f_cat alpha^T in R^{D x 1}       # modality-weighted sum
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn


class ATFusion(nn.Module):
    """Attention-based Audio-Text Fusion (the paper's proposed fusion)."""

    def __init__(self, dim: int, attn_hidden: int = 128) -> None:
        super().__init__()
        self.dim = dim
        self.W = nn.Linear(dim, attn_hidden)
        self.w = nn.Linear(attn_hidden, 1, bias=False)

    @property
    def output_dim(self) -> int:
        return self.dim

    def forward(self, f_a: torch.Tensor, f_t: torch.Tensor) -> torch.Tensor:
        modalities = torch.stack([f_a, f_t], dim=1)  # [B, 2, D]
        scores = self.w(torch.tanh(self.W(modalities)))  # [B, 2, 1]
        alpha = torch.softmax(scores, dim=1)  # [B, 2, 1]
        return torch.sum(alpha * modalities, dim=1)  # [B, D]


class ConcatFusion(nn.Module):
    """Plain concatenation baseline. output_dim = Da + Dt."""

    def __init__(self, dim_a: int, dim_t: int) -> None:
        super().__init__()
        self._out = dim_a + dim_t

    @property
    def output_dim(self) -> int:
        return self._out

    def forward(self, f_a: torch.Tensor, f_t: torch.Tensor) -> torch.Tensor:
        return torch.cat([f_a, f_t], dim=1)  # [B, Da + Dt]


class GMUFusion(nn.Module):
    """Gated Multimodal Unit (Arevalo et al. [37]), equations per paper.

        h_x = tanh(W_x x + b_x); h_y = tanh(W_y y + b_y)
        z   = sigmoid(W_z [x; y] + b_z)
        h   = z * h_x + (1 - z) * h_y

    Paper uses hidden size 128. output_dim = hidden.
    """

    def __init__(self, dim_a: int, dim_t: int, hidden: int = 128) -> None:
        super().__init__()
        self.hidden = hidden
        self.Wx = nn.Linear(dim_a, hidden)
        self.Wy = nn.Linear(dim_t, hidden)
        self.Wz = nn.Linear(dim_a + dim_t, hidden)

    @property
    def output_dim(self) -> int:
        return self.hidden

    def forward(self, f_a: torch.Tensor, f_t: torch.Tensor) -> torch.Tensor:
        hx = torch.tanh(self.Wx(f_a))  # [B, H]
        hy = torch.tanh(self.Wy(f_t))  # [B, H]
        z = torch.sigmoid(self.Wz(torch.cat([f_a, f_t], dim=1)))  # [B, H]
        return z * hx + (1.0 - z) * hy  # [B, H]


_BILINEAR = {"mutan", "mfb", "mfh", "block"}


def build_fusion(
    name: str,
    dim_a: int,
    dim_t: int,
    attn_hidden: int = 128,
    output_dim: Optional[int] = None,
    bilinear: Optional[dict] = None,
) -> nn.Module:
    """Construct a fusion module by name."""
    name = name.lower()
    if name == "at_fusion":
        if dim_a != dim_t:
            raise ValueError(
                f"AT-Fusion requires equal modality dims, got {dim_a} vs {dim_t}."
            )
        return ATFusion(dim_a, attn_hidden=attn_hidden)
    if name == "concat":
        return ConcatFusion(dim_a, dim_t)
    if name == "gmu":
        return GMUFusion(dim_a, dim_t)

    if name in _BILINEAR:
        from .bilinear_fusion import BLOCK, MFB, MFH, MUTAN

        out = output_dim if output_dim is not None else dim_a
        p = bilinear or {}
        dropout = p.get("dropout", 0.1)
        if name == "mfb":
            return MFB(dim_a, dim_t, out, mm_dim=p.get("mfb_mm_dim", 1200),
                       factor=p.get("factor", 2), dropout=dropout)
        if name == "mfh":
            return MFH(dim_a, dim_t, out, mm_dim=p.get("mfb_mm_dim", 1200),
                       factor=p.get("factor", 2), dropout=dropout)
        if name == "mutan":
            return MUTAN(dim_a, dim_t, out, mm_dim=p.get("mm_dim", 1600),
                         rank=p.get("rank", 15), dropout=dropout)
        if name == "block":
            return BLOCK(dim_a, dim_t, out, mm_dim=p.get("mm_dim", 1600),
                         chunks=p.get("chunks", 20), rank=p.get("rank", 15),
                         dropout=dropout)

    raise ValueError(
        f"Unknown fusion '{name}'. Expected at_fusion|concat|gmu|mutan|mfb|mfh|block."
    )
