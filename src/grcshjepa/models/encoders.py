from __future__ import annotations

import torch
import torch.nn as nn


class MazeEncoder(nn.Module):
    """Small CNN encoder for smoke-mode maze observations."""

    def __init__(self, latent_dim: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(4, 16, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(16, 24, kernel_size=3, padding=1),
            nn.GELU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(24, latent_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class SortingEncoder(nn.Module):
    """Small MLP encoder for sequence states."""

    def __init__(self, length: int, latent_dim: int = 32):
        super().__init__()
        self.length = int(length)
        self.net = nn.Sequential(
            nn.Linear(length, 64),
            nn.GELU(),
            nn.Linear(64, latent_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
