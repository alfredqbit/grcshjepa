from __future__ import annotations

import torch
import torch.nn as nn


class MazeActionHead(nn.Module):
    def __init__(self, latent_dim: int = 32, n_actions: int = 4):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(latent_dim, 48), nn.GELU(), nn.Linear(48, n_actions))

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)


class SortingHead(nn.Module):
    def __init__(self, latent_dim: int = 32, length: int = 8):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(latent_dim, 64), nn.GELU(), nn.Linear(64, length))

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)


class ValueHead(nn.Module):
    def __init__(self, latent_dim: int = 32):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(latent_dim, 48), nn.GELU(), nn.Linear(48, 1))

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z).squeeze(-1)


class VerifierPredicateHead(nn.Module):
    def __init__(self, latent_dim: int = 32, n_predicates: int = 4):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(latent_dim, 48), nn.GELU(), nn.Linear(48, n_predicates))

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)
