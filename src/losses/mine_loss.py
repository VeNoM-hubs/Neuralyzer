"""MINE mutual-information loss + combined objective (paper Sec IV-c, IV-e).

Donsker-Varadhan lower bound of the mutual information:

    I(X, Z) >= E_p(x,z)[T_theta] - log E_p(x)p(z)[ e^{T_theta} ]

Minibatch estimation:
  * joint pairs      (f_a[i], f_t[i])          ~ p(x, z)
  * marginal pairs   (f_a[i], f_t[perm(i)])    ~ p(x) p(z)   (shuffle f_t)
    I_hat = mean(T_joint) - logmeanexp(T_marginal)

Multimodal MI objective: L_mi = -I_hat(f_a, f_t).
Final objective (paper Sec IV-e): L = L_cls + lambda * L_mi.
"""

from __future__ import annotations

import math
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def mine_lower_bound(mine: nn.Module, f_a: torch.Tensor, f_t: torch.Tensor) -> torch.Tensor:
    """Return the DV lower-bound estimate I_hat(f_a, f_t) for the batch."""
    batch_size = f_a.shape[0]
    perm = torch.randperm(batch_size, device=f_t.device)
    f_t_shuffled = f_t[perm]

    t_joint = mine(f_a, f_t)  # [B, 1]
    t_marginal = mine(f_a, f_t_shuffled)  # [B, 1]

    log_mean_exp = torch.logsumexp(t_marginal, dim=0) - math.log(batch_size)
    mi_estimate = t_joint.mean() - log_mean_exp
    return mi_estimate.squeeze()


def mine_loss(mine: nn.Module, f_a: torch.Tensor, f_t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return (L_mi, I_hat) where L_mi = -I_hat (to be minimized)."""
    mi_estimate = mine_lower_bound(mine, f_a, f_t)
    return -mi_estimate, mi_estimate


def combined_loss(logits, labels, mine, f_a, f_t, mine_lambda: float = 0.25):
    """Compute L = CE + lambda * L_mi. Returns (total, ce, mi_estimate)."""
    ce = F.cross_entropy(logits, labels)
    l_mi, mi_estimate = mine_loss(mine, f_a, f_t)
    total = ce + mine_lambda * l_mi
    return total, ce, mi_estimate
