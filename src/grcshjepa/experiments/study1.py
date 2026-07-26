from __future__ import annotations

from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader

from grcshjepa.config import ExperimentConfig
from grcshjepa.data.datasets import MazeTrajectoryDataset, SortingTrajectoryDataset
from grcshjepa.models.encoders import MazeEncoder, SortingEncoder
from grcshjepa.models.hjepa import HJEPA
from grcshjepa.models.predictors import MLPPredictor, RelaxedContinuousSpikingPredictor
from grcshjepa.training.pretrain import train_jepa_epoch, evaluate_jepa
from grcshjepa.utils import resolve_device, set_seed, write_manifest, environment_report


def _make_hjepa(cfg: ExperimentConfig, encoder, predictor_type: str):
    max_horizon = max(cfg.horizon_set)
    horizon_dim = 16
    if predictor_type == "spiking":
        predictor = RelaxedContinuousSpikingPredictor(
            latent_dim=cfg.latent_dim,
            horizon_dim=horizon_dim,
            n_neurons=cfg.n_neurons,
            n_filters=cfg.n_filters,
            internal_steps=cfg.internal_steps,
            dt=cfg.dt,
            noise_std=cfg.noise_std,
            surrogate_eps=cfg.surrogate_eps,
        )
    else:
        predictor = MLPPredictor(latent_dim=cfg.latent_dim, horizon_dim=horizon_dim)
    return HJEPA(
        encoder=encoder,
        latent_dim=cfg.latent_dim,
        projection_dim=cfg.projection_dim,
        max_horizon=max_horizon,
        predictor=predictor,
        horizon_dim=horizon_dim,
    )


def run_study1_smoke(cfg: ExperimentConfig) -> pd.DataFrame:
    """Run Study 1 smoke: H-JEPA predictive pretraining on maze and sorting pairs."""
    set_seed(cfg.seed)
    device = resolve_device(cfg.device)
    outdir = Path(cfg.output_dir) / "study1_smoke"
    outdir.mkdir(parents=True, exist_ok=True)

    rows = []
    for task in ["maze", "sorting"]:
        if task == "maze":
            train_ds = MazeTrajectoryDataset(cfg.train_samples, cfg.maze_size_train, cfg.maze_obstacle_prob, cfg.seed, cfg.horizon_set)
            val_ds = MazeTrajectoryDataset(cfg.val_samples, cfg.maze_size_train, cfg.maze_obstacle_prob, cfg.seed + 101, cfg.horizon_set)
            encoder = MazeEncoder(cfg.latent_dim)
        else:
            train_ds = SortingTrajectoryDataset(cfg.train_samples, cfg.sorting_length_train, cfg.seed + 2, cfg.horizon_set)
            val_ds = SortingTrajectoryDataset(cfg.val_samples, cfg.sorting_length_train, cfg.seed + 202, cfg.horizon_set)
            encoder = SortingEncoder(cfg.sorting_length_train, cfg.latent_dim)

        for predictor_type in ["mlp", "spiking"]:
            model = _make_hjepa(cfg, encoder if predictor_type == "mlp" else type(encoder)(*([cfg.latent_dim] if task == "maze" else [cfg.sorting_length_train, cfg.latent_dim])), predictor_type)
            model.to(device)
            opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr)
            train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True)
            val_loader = DataLoader(val_ds, batch_size=cfg.batch_size)
            metrics = {}
            for epoch in range(cfg.epochs):
                train_metrics = train_jepa_epoch(model, train_loader, opt, device, lambda_ac=cfg.lambda_ac, ema_tau=cfg.ema_tau)
                val_metrics = evaluate_jepa(model, val_loader, device)
                metrics = {**train_metrics, **{f"val_{k}": v for k, v in val_metrics.items()}}
            rows.append({"study": "study1", "task": task, "predictor": predictor_type, "seed": cfg.seed, **metrics})

    df = pd.DataFrame(rows)
    df.to_csv(outdir / f"study1_seed{cfg.seed}.csv", index=False)
    write_manifest(outdir / f"manifest_seed{cfg.seed}.json", {"study": "study1", "seed": cfg.seed, "rows": len(df), "environment": environment_report()})
    return df
