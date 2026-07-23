"""MINE statistics network (paper Sec IV-c; Belghazi et al. 2018 [32]).

Why: maximize mutual information (MI) between acoustic embedding f_a and textual
embedding f_t so the modalities preserve correlated, dementia-relevant info. MI
is estimated with a small network T_theta and the Donsker-Varadhan bound.

This module is ONLY T_theta. The DV bound / loss is in src/losses/mine_loss.py.

Tensors:
    in:  x [B, Da], z [B, Dz]
    out: T [B, 1]   (scalar score per (x, z) pair)

Architecture (user decision; paper only says "simple linear layers + ReLU"):
input = concat(x, z); two hidden Linear+ReLU layers of size ``hidden``; scalar out.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class MINE(nn.Module):
    def __init__(self, dim_x: int, dim_z: int, hidden: int = 512) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim_x + dim_z, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, x: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        # x: [B, Da], z: [B, Dz] -> T: [B, 1]
        return self.net(torch.cat([x, z], dim=1))
