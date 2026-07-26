from __future__ import annotations

import numpy as np
import torch

from grcshjepa.losses import effective_rank_from_eigs


def covariance_diagnostics(y: torch.Tensor) -> dict[str, float]:
    if y.ndim != 2:
        raise ValueError("Expected [batch, dim] tensor.")
    centered = y - y.mean(dim=0)
    cov = centered.T @ centered / max(1, y.shape[0] - 1)
    eig = torch.linalg.eigvalsh(cov).clamp_min(0)
    return {
        "cov_trace": float(torch.trace(cov).detach().cpu()),
        "cov_min_eig": float(eig.min().detach().cpu()) if eig.numel() else 0.0,
        "cov_max_eig": float(eig.max().detach().cpu()) if eig.numel() else 0.0,
        "effective_rank": effective_rank_from_eigs(eig.detach().cpu()),
    }


def latent_aliasing_rate(z: torch.Tensor, labels: torch.Tensor, distance_quantile: float = 0.05) -> float:
    """Crude harmful-alias diagnostic: near latent pairs with different labels."""
    if z.shape[0] < 3:
        return 0.0
    z = torch.nn.functional.normalize(z.detach(), dim=-1)
    d = torch.cdist(z, z)
    mask = ~torch.eye(z.shape[0], dtype=torch.bool, device=z.device)
    vals = d[mask]
    threshold = torch.quantile(vals, distance_quantile)
    near = (d <= threshold) & mask
    diff = labels.view(-1, 1) != labels.view(1, -1)
    denom = near.sum().item()
    if denom == 0:
        return 0.0
    return float((near & diff).sum().item() / denom)


def timescale_autocorrelation_time(values: np.ndarray, max_lag: int = 20) -> float:
    x = np.asarray(values, dtype=float)
    if x.size < 3 or np.std(x) == 0:
        return 0.0
    x = x - x.mean()
    denom = np.dot(x, x)
    ac = []
    for lag in range(1, min(max_lag, x.size - 1) + 1):
        ac.append(float(np.dot(x[:-lag], x[lag:]) / denom))
    positive = [v for v in ac if v > 0]
    return float(1.0 + 2.0 * np.sum(positive))


def risk_coverage_curve(confidence: np.ndarray, correct: np.ndarray, steps: int = 10) -> list[dict[str, float]]:
    confidence = np.asarray(confidence)
    correct = np.asarray(correct).astype(float)
    out = []
    for q in np.linspace(0.0, 1.0, steps):
        thresh = np.quantile(confidence, q)
        keep = confidence >= thresh
        out.append(
            {
                "coverage": float(keep.mean()),
                "risk": float(1.0 - correct[keep].mean()) if keep.any() else float("nan"),
                "threshold": float(thresh),
            }
        )
    return out
