"""Bilinear multimodal fusion baselines: MUTAN, MFB, MFH, BLOCK.

Ablation baselines (paper Sec VII-D). The paper does NOT specify their
factorization hyperparameters, so defaults here are the reference values from
the original papers and block.bootstrap.pytorch (Cadene et al.) -- an explicit,
user-approved assumption:

    MFB  [39]  : mm_dim=1200, factor(k)=2
    MFH  [39]  : mm_dim=1200, factor(k)=2 (two stacked MFB orders)
    MUTAN[38]  : mm_dim=1600, rank=15
    BLOCK[40]  : mm_dim=1600, chunks=20, rank=15

Interface:
    in:  f_a [B, Da], f_t [B, Dt]
    out: h   [B, output_dim]
"""

from __future__ import annotations

from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F


def _signed_sqrt_l2(z: torch.Tensor) -> torch.Tensor:
    """Signed square-root power normalization followed by L2 normalization."""
    z = torch.sqrt(F.relu(z)) - torch.sqrt(F.relu(-z))
    return F.normalize(z, p=2, dim=1)


def _get_sizes_list(dim: int, chunks: int) -> List[int]:
    """Split ``dim`` into ``chunks`` near-equal positive sizes (block.bootstrap)."""
    split_size = (dim + chunks - 1) // chunks  # ceil
    sizes_list = [split_size] * chunks
    diff = sum(sizes_list) - dim
    i = 0
    while diff > 0:
        sizes_list[i % chunks] -= 1
        diff -= 1
        i += 1
    return [s for s in sizes_list if s > 0]


class MFB(nn.Module):
    """Multimodal Factorized Bilinear pooling [39]."""

    def __init__(self, dim_a, dim_t, output_dim, mm_dim=1200, factor=2,
                 dropout=0.1, normalize=False) -> None:
        super().__init__()
        self.mm_dim = mm_dim
        self.factor = factor
        self.normalize = normalize
        self._out = output_dim
        self.linear0 = nn.Linear(dim_a, mm_dim * factor)
        self.linear1 = nn.Linear(dim_t, mm_dim * factor)
        self.dropout = nn.Dropout(dropout)
        self.linear_out = nn.Linear(mm_dim, output_dim)

    @property
    def output_dim(self) -> int:
        return self._out

    def forward(self, f_a: torch.Tensor, f_t: torch.Tensor) -> torch.Tensor:
        x0 = F.relu(self.linear0(f_a))  # [B, mm_dim*factor]
        x1 = F.relu(self.linear1(f_t))  # [B, mm_dim*factor]
        z = self.dropout(x0 * x1)
        z = z.view(z.size(0), self.mm_dim, self.factor).sum(dim=2)  # [B, mm_dim]
        if self.normalize:
            z = _signed_sqrt_l2(z)
        return self.linear_out(z)  # [B, output_dim]


class MFH(nn.Module):
    """Multimodal Factorized High-order pooling [39] (two stacked MFB orders)."""

    def __init__(self, dim_a, dim_t, output_dim, mm_dim=1200, factor=2,
                 dropout=0.1, normalize=True) -> None:
        super().__init__()
        self.mm_dim = mm_dim
        self.factor = factor
        self.normalize = normalize
        self._out = output_dim
        self.linear0_0 = nn.Linear(dim_a, mm_dim * factor)
        self.linear1_0 = nn.Linear(dim_t, mm_dim * factor)
        self.linear0_1 = nn.Linear(dim_a, mm_dim * factor)
        self.linear1_1 = nn.Linear(dim_t, mm_dim * factor)
        self.dropout = nn.Dropout(dropout)
        self.linear_out = nn.Linear(mm_dim * 2, output_dim)

    @property
    def output_dim(self) -> int:
        return self._out

    def _order(self, x0: torch.Tensor, x1: torch.Tensor):
        z_skip = x0 * x1  # [B, mm_dim*factor]
        z = self.dropout(z_skip)
        z = z.view(z.size(0), self.mm_dim, self.factor).sum(dim=2)  # [B, mm_dim]
        if self.normalize:
            z = _signed_sqrt_l2(z)
        return z, z_skip

    def forward(self, f_a: torch.Tensor, f_t: torch.Tensor) -> torch.Tensor:
        x0 = F.relu(self.linear0_0(f_a))
        x1 = F.relu(self.linear1_0(f_t))
        z0, z0_skip = self._order(x0, x1)
        x0 = F.relu(self.linear0_1(f_a))
        x1 = F.relu(self.linear1_1(f_t))
        z1, _ = self._order(x0 * z0_skip, x1)
        z = torch.cat([z0, z1], dim=1)  # [B, mm_dim*2]
        return self.linear_out(z)  # [B, output_dim]


class MUTAN(nn.Module):
    """Multimodal Tucker fusion [38]."""

    def __init__(self, dim_a, dim_t, output_dim, mm_dim=1600, rank=15,
                 dropout=0.1, normalize=False) -> None:
        super().__init__()
        self.mm_dim = mm_dim
        self.rank = rank
        self.normalize = normalize
        self._out = output_dim
        self.linear0 = nn.Linear(dim_a, mm_dim)
        self.linear1 = nn.Linear(dim_t, mm_dim)
        self.merge0 = nn.Linear(mm_dim, mm_dim * rank)
        self.merge1 = nn.Linear(mm_dim, mm_dim * rank)
        self.dropout = nn.Dropout(dropout)
        self.linear_out = nn.Linear(mm_dim, output_dim)

    @property
    def output_dim(self) -> int:
        return self._out

    def forward(self, f_a: torch.Tensor, f_t: torch.Tensor) -> torch.Tensor:
        x0 = F.relu(self.linear0(f_a))  # [B, mm_dim]
        x1 = F.relu(self.linear1(f_t))  # [B, mm_dim]
        m = self.dropout(self.merge0(x0) * self.merge1(x1))  # [B, mm_dim*rank]
        m = m.view(m.size(0), self.rank, self.mm_dim).sum(dim=1)  # [B, mm_dim]
        if self.normalize:
            m = _signed_sqrt_l2(m)
        return self.linear_out(m)  # [B, output_dim]


class BLOCK(nn.Module):
    """BLOCK: bilinear superdiagonal (block-term) fusion [40]."""

    def __init__(self, dim_a, dim_t, output_dim, mm_dim=1600, chunks=20, rank=15,
                 dropout=0.1) -> None:
        super().__init__()
        self.rank = rank
        self._out = output_dim
        self.linear0 = nn.Linear(dim_a, mm_dim)
        self.linear1 = nn.Linear(dim_t, mm_dim)
        self.sizes = _get_sizes_list(mm_dim, chunks)
        self.merge0 = nn.ModuleList([nn.Linear(s, s * rank) for s in self.sizes])
        self.merge1 = nn.ModuleList([nn.Linear(s, s * rank) for s in self.sizes])
        self.dropout = nn.Dropout(dropout)
        self.linear_out = nn.Linear(sum(self.sizes), output_dim)

    @property
    def output_dim(self) -> int:
        return self._out

    def forward(self, f_a: torch.Tensor, f_t: torch.Tensor) -> torch.Tensor:
        x0 = self.dropout(self.linear0(f_a))  # [B, mm_dim]
        x1 = self.dropout(self.linear1(f_t))  # [B, mm_dim]
        x0_chunks = torch.split(x0, self.sizes, dim=1)
        x1_chunks = torch.split(x1, self.sizes, dim=1)
        zs = []
        for i, size in enumerate(self.sizes):
            m = self.merge0[i](x0_chunks[i]) * self.merge1[i](x1_chunks[i])  # [B, size*rank]
            m = m.view(m.size(0), self.rank, size).sum(dim=1)  # [B, size]
            zs.append(_signed_sqrt_l2(m))
        z = torch.cat(zs, dim=1)  # [B, sum(sizes)] == [B, mm_dim]
        return self.linear_out(z)  # [B, output_dim]
