from __future__ import annotations

import copy
from typing import Dict

import torch
import torch.nn as nn

from grcshjepa.models.predictors import HorizonEmbedding


class HJEPA(nn.Module):
    """H-JEPA wrapper with online encoder, EMA target encoder, predictor, and projector."""

    def __init__(
        self,
        encoder: nn.Module,
        latent_dim: int,
        projection_dim: int,
        max_horizon: int,
        predictor: nn.Module,
        horizon_dim: int = 16,
    ):
        super().__init__()
        self.encoder = encoder
        self.target_encoder = copy.deepcopy(encoder)
        for p in self.target_encoder.parameters():
            p.requires_grad = False
        self.horizon = HorizonEmbedding(max_horizon=max_horizon, dim=horizon_dim)
        self.predictor = predictor
        self.projector = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.GELU(),
            nn.Linear(64, projection_dim),
        )

    @torch.no_grad()
    def update_target(self, tau: float = 0.99) -> None:
        if not 0.0 <= tau <= 1.0:
            raise ValueError(f"EMA tau must be in [0,1], got {tau}")
        for target, online in zip(self.target_encoder.parameters(), self.encoder.parameters()):
            target.data.mul_(tau).add_(online.data, alpha=1.0 - tau)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    @torch.no_grad()
    def encode_target(self, x_future: torch.Tensor) -> torch.Tensor:
        # stop-gradient target embedding
        return self.target_encoder(x_future).detach()

    def forward(self, x: torch.Tensor, x_future: torch.Tensor, horizon: torch.Tensor) -> Dict[str, torch.Tensor]:
        z = self.encoder(x)
        h = self.horizon(horizon)
        z_hat = self.predictor(z, h)
        z_tgt = self.encode_target(x_future)
        y_proj = self.projector(z)
        return {"z": z, "z_hat": z_hat, "z_tgt": z_tgt, "y_proj": y_proj}
