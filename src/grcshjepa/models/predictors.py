from __future__ import annotations

import torch
import torch.nn as nn


class HorizonEmbedding(nn.Module):
    def __init__(self, max_horizon: int, dim: int):
        super().__init__()
        self.max_horizon = int(max_horizon)
        self.emb = nn.Embedding(self.max_horizon + 1, dim)

    def forward(self, horizon: torch.Tensor) -> torch.Tensor:
        horizon = horizon.clamp(0, self.max_horizon)
        return self.emb(horizon)


class MLPPredictor(nn.Module):
    """Simple latent predictor baseline."""

    def __init__(self, latent_dim: int = 32, horizon_dim: int = 16):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim + horizon_dim, 96),
            nn.GELU(),
            nn.Linear(96, latent_dim),
        )

    def forward(self, z: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([z, h], dim=-1))


class RelaxedContinuousSpikingPredictor(nn.Module):
    """Tiny relaxed continuous-spiking predictor used as an executable stand-in.

    This is not a full neuromorphic simulator. It implements the proposal's training-time
    relaxation pattern: smooth surrogate spikes, finite filter state, Euler-style internal updates,
    and a latent projection after internal computation.
    """

    def __init__(
        self,
        latent_dim: int = 32,
        horizon_dim: int = 16,
        n_neurons: int = 48,
        n_filters: int = 2,
        internal_steps: int = 4,
        dt: float = 0.08,
        noise_std: float = 0.01,
        surrogate_eps: float = 0.7,
    ):
        super().__init__()
        self.latent_dim = int(latent_dim)
        self.horizon_dim = int(horizon_dim)
        self.n_neurons = int(n_neurons)
        self.n_filters = int(n_filters)
        self.internal_steps = int(internal_steps)
        self.dt = float(dt)
        self.noise_std = float(noise_std)
        self.surrogate_eps = float(surrogate_eps)

        self.inp = nn.Linear(latent_dim + horizon_dim, n_neurons)
        self.recurrent = nn.Linear(n_neurons, n_neurons, bias=False)
        self.filter_mix = nn.Linear(n_neurons * n_filters, n_neurons)
        self.out = nn.Linear(n_neurons * (1 + n_filters), latent_dim)
        self.threshold = nn.Parameter(torch.zeros(n_neurons))
        self.tau_m_log = nn.Parameter(torch.zeros(n_neurons))
        self.tau_f_log = nn.Parameter(torch.zeros(n_filters))

    def forward(self, z: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        batch = z.shape[0]
        context = self.inp(torch.cat([z, h], dim=-1))
        u = torch.tanh(context)
        filters = torch.zeros(batch, self.n_neurons, self.n_filters, device=z.device, dtype=z.dtype)

        tau_m = torch.nn.functional.softplus(self.tau_m_log) + 0.2
        tau_f = torch.nn.functional.softplus(self.tau_f_log) + 0.2

        for _ in range(self.internal_steps):
            spikes = torch.sigmoid((u - self.threshold) / self.surrogate_eps)
            rec = self.recurrent(spikes)
            filt_drive = self.filter_mix(filters.reshape(batch, -1))
            drift = -u / tau_m + rec + filt_drive + context
            if self.training and self.noise_std > 0:
                noise = self.noise_std * (self.dt**0.5) * torch.randn_like(u)
            else:
                noise = 0.0
            u = u + self.dt * drift + noise
            filters = filters + self.dt * (-filters / tau_f.view(1, 1, -1) + spikes.unsqueeze(-1))

        state = torch.cat([u, filters.reshape(batch, -1)], dim=-1)
        return self.out(state)
