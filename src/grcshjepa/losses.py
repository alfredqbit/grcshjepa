from __future__ import annotations

from typing import Dict, Tuple

import torch
import torch.nn.functional as F


def normalize_latent(z: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    return z / (z.norm(dim=-1, keepdim=True) + eps)


def normalized_prediction_loss(z_hat: torch.Tensor, z_tgt: torch.Tensor) -> torch.Tensor:
    """Cosine-style squared error on normalized latent states."""
    z_hat_n = normalize_latent(z_hat)
    z_tgt_n = normalize_latent(z_tgt.detach())
    return F.mse_loss(z_hat_n, z_tgt_n)


def effective_rank_from_eigs(eigs: torch.Tensor, eps: float = 1e-12) -> float:
    vals = torch.as_tensor(eigs, dtype=torch.float64).clamp_min(0)
    total = vals.sum()
    if float(total) <= eps:
        return 0.0
    p = vals / total
    entropy = -(p * (p + eps).log()).sum()
    return float(torch.exp(entropy))


def anti_collapse_loss(y: torch.Tensor, eps: float = 1e-6) -> Tuple[torch.Tensor, Dict[str, float]]:
    """First- and second-moment anti-collapse loss.

    The loss penalizes nonzero batch mean and covariance away from identity. A constant embedding
    has covariance near zero and therefore incurs a dimension-scale penalty.
    """
    if y.ndim != 2:
        raise ValueError(f"Expected y with shape [batch, dim], got {tuple(y.shape)}")
    batch_size, dim = y.shape
    mean = y.mean(dim=0)
    centered = y - mean
    if batch_size <= 1:
        cov = torch.zeros(dim, dim, device=y.device, dtype=y.dtype)
    else:
        cov = centered.T @ centered / (batch_size - 1)
    eye = torch.eye(dim, device=y.device, dtype=y.dtype)
    loss = mean.pow(2).sum() + (cov - eye).pow(2).sum()
    with torch.no_grad():
        eig = torch.linalg.eigvalsh(cov).clamp_min(0)
        stats = {
            "ac_mean_norm": float(mean.norm().detach().cpu()),
            "ac_cov_trace": float(torch.trace(cov).detach().cpu()),
            "ac_cov_min_eig": float(eig.min().detach().cpu()) if eig.numel() else 0.0,
            "ac_cov_max_eig": float(eig.max().detach().cpu()) if eig.numel() else 0.0,
            "ac_effective_rank": effective_rank_from_eigs(eig.detach().cpu()),
        }
    return loss, stats
