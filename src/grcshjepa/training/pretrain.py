from __future__ import annotations

from typing import Iterable

import torch
from torch.utils.data import DataLoader

from grcshjepa.losses import anti_collapse_loss, normalized_prediction_loss


def _move_batch(batch: dict, device: torch.device) -> dict:
    return {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}


def train_jepa_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    *,
    lambda_ac: float = 0.03,
    ema_tau: float = 0.99,
) -> dict[str, float]:
    model.train()
    total = 0.0
    pred_total = 0.0
    ac_total = 0.0
    n = 0
    last_stats: dict[str, float] = {}
    for batch in loader:
        batch = _move_batch(batch, device)
        out = model(batch["x"], batch["x_future"], batch["horizon"])
        pred_loss = normalized_prediction_loss(out["z_hat"], out["z_tgt"])
        ac_loss, ac_stats = anti_collapse_loss(out["y_proj"])
        loss = pred_loss + lambda_ac * ac_loss
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        model.update_target(ema_tau)
        bs = batch["x"].shape[0]
        total += float(loss.item()) * bs
        pred_total += float(pred_loss.item()) * bs
        ac_total += float(ac_loss.item()) * bs
        n += bs
        last_stats = ac_stats
    return {
        "loss": total / max(1, n),
        "pred_loss": pred_total / max(1, n),
        "anti_collapse": ac_total / max(1, n),
        **last_stats,
    }


@torch.no_grad()
def evaluate_jepa(model: torch.nn.Module, loader: DataLoader, device: torch.device) -> dict[str, float]:
    model.eval()
    pred_total = 0.0
    n = 0
    zs = []
    labels = []
    for batch in loader:
        batch = _move_batch(batch, device)
        out = model(batch["x"], batch["x_future"], batch["horizon"])
        pred_loss = normalized_prediction_loss(out["z_hat"], out["z_tgt"])
        bs = batch["x"].shape[0]
        pred_total += float(pred_loss.item()) * bs
        n += bs
        zs.append(out["z"].detach().cpu())
        if "action" in batch:
            labels.append(batch["action"].detach().cpu())
    z_all = torch.cat(zs, dim=0) if zs else torch.empty(0)
    result = {"pred_loss": pred_total / max(1, n)}
    if z_all.numel() > 0:
        from grcshjepa.diagnostics import covariance_diagnostics

        result.update(covariance_diagnostics(z_all))
    if labels and z_all.shape[0] > 2:
        from grcshjepa.diagnostics import latent_aliasing_rate

        result["aliasing_rate"] = latent_aliasing_rate(z_all, torch.cat(labels, dim=0))
    return result
