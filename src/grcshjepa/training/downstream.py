from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset


def make_labeled_subset(dataset, fraction: float, seed: int) -> Subset:
    rng = np.random.default_rng(seed)
    n = len(dataset)
    k = max(4, int(round(n * fraction)))
    idx = np.sort(rng.choice(n, size=min(k, n), replace=False))
    return Subset(dataset, idx.tolist())


@torch.no_grad()
def encode_dataset(encoder: torch.nn.Module, loader: DataLoader, device: torch.device, target_key: str):
    encoder.eval()
    zs, ys = [], []
    for batch in loader:
        x = batch["x"].to(device)
        zs.append(encoder(x).detach().cpu())
        ys.append(batch[target_key].detach().cpu())
    return torch.cat(zs, dim=0), torch.cat(ys, dim=0)


def train_classification_head(head, z_train, y_train, *, epochs: int, lr: float) -> dict[str, float]:
    optimizer = torch.optim.AdamW(head.parameters(), lr=lr)
    for _ in range(epochs):
        logits = head(z_train)
        loss = F.cross_entropy(logits, y_train)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    with torch.no_grad():
        logits = head(z_train)
        acc = (logits.argmax(dim=-1) == y_train).float().mean().item()
    return {"train_loss": float(loss.item()), "train_acc": float(acc)}


def evaluate_classification_head(head, z_test, y_test) -> dict[str, float]:
    with torch.no_grad():
        logits = head(z_test)
        acc = (logits.argmax(dim=-1) == y_test).float().mean().item()
        loss = F.cross_entropy(logits, y_test).item()
    return {"test_loss": float(loss), "test_acc": float(acc)}


def train_regression_head(head, z_train, y_train, *, epochs: int, lr: float) -> dict[str, float]:
    optimizer = torch.optim.AdamW(head.parameters(), lr=lr)
    for _ in range(epochs):
        pred = head(z_train)
        loss = F.mse_loss(pred, y_train)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    return {"train_mse": float(loss.item())}


def evaluate_regression_head(head, z_test, y_test) -> dict[str, float]:
    with torch.no_grad():
        pred = head(z_test)
        mse = F.mse_loss(pred, y_test).item()
        mae = torch.mean(torch.abs(pred - y_test)).item()
        exactish = (torch.mean(torch.abs(pred - y_test), dim=-1) < 0.05).float().mean().item()
    return {"test_mse": float(mse), "test_mae": float(mae), "exactish_rate": float(exactish)}
