from __future__ import annotations

from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader

from grcshjepa.config import ExperimentConfig
from grcshjepa.data.datasets import MazeTrajectoryDataset, SortingTrajectoryDataset
from grcshjepa.experiments.study1 import _make_hjepa
from grcshjepa.models.encoders import MazeEncoder, SortingEncoder
from grcshjepa.models.heads import MazeActionHead, SortingHead
from grcshjepa.training.downstream import (
    make_labeled_subset,
    encode_dataset,
    train_classification_head,
    evaluate_classification_head,
    train_regression_head,
    evaluate_regression_head,
)
from grcshjepa.training.pretrain import train_jepa_epoch
from grcshjepa.utils import resolve_device, set_seed, write_manifest, environment_report


def _pretrain_for_downstream(cfg: ExperimentConfig, task: str, device: torch.device):
    if task == "maze":
        train_ds = MazeTrajectoryDataset(cfg.train_samples, cfg.maze_size_train, cfg.maze_obstacle_prob, cfg.seed, cfg.horizon_set)
        encoder = MazeEncoder(cfg.latent_dim)
    else:
        train_ds = SortingTrajectoryDataset(cfg.train_samples, cfg.sorting_length_train, cfg.seed + 2, cfg.horizon_set)
        encoder = SortingEncoder(cfg.sorting_length_train, cfg.latent_dim)
    model = _make_hjepa(cfg, encoder, "spiking").to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr)
    loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True)
    for _ in range(cfg.epochs):
        train_jepa_epoch(model, loader, opt, device, lambda_ac=cfg.lambda_ac, ema_tau=cfg.ema_tau)
    return model


def run_study2_smoke(cfg: ExperimentConfig) -> pd.DataFrame:
    """Run Study 2 smoke: frozen encoder plus small downstream heads."""
    set_seed(cfg.seed)
    device = resolve_device(cfg.device)
    outdir = Path(cfg.output_dir) / "study2_smoke"
    outdir.mkdir(parents=True, exist_ok=True)
    rows = []

    # Maze action head
    maze_model = _pretrain_for_downstream(cfg, "maze", device)
    train_maze = MazeTrajectoryDataset(cfg.train_samples, cfg.maze_size_train, cfg.maze_obstacle_prob, cfg.seed + 10, cfg.horizon_set)
    test_maze = MazeTrajectoryDataset(cfg.test_samples, cfg.maze_size_train, cfg.maze_obstacle_prob, cfg.seed + 11, cfg.horizon_set)
    labeled_maze = make_labeled_subset(train_maze, cfg.downstream_label_fraction, cfg.seed)
    z_train, y_train = encode_dataset(maze_model.encoder, DataLoader(labeled_maze, batch_size=cfg.batch_size), device, "action")
    z_test, y_test = encode_dataset(maze_model.encoder, DataLoader(test_maze, batch_size=cfg.batch_size), device, "action")
    head = MazeActionHead(cfg.latent_dim)
    train_m = train_classification_head(head, z_train, y_train, epochs=cfg.head_epochs, lr=cfg.head_lr)
    test_m = evaluate_classification_head(head, z_test, y_test)
    rows.append({"study": "study2", "task": "maze_action", "seed": cfg.seed, **train_m, **test_m})

    # Sorting regression head
    sort_model = _pretrain_for_downstream(cfg, "sorting", device)
    train_sort = SortingTrajectoryDataset(cfg.train_samples, cfg.sorting_length_train, cfg.seed + 20, cfg.horizon_set)
    test_sort = SortingTrajectoryDataset(cfg.test_samples, cfg.sorting_length_train, cfg.seed + 21, cfg.horizon_set)
    labeled_sort = make_labeled_subset(train_sort, cfg.downstream_label_fraction, cfg.seed)
    z_train, y_train = encode_dataset(sort_model.encoder, DataLoader(labeled_sort, batch_size=cfg.batch_size), device, "y_sorted")
    z_test, y_test = encode_dataset(sort_model.encoder, DataLoader(test_sort, batch_size=cfg.batch_size), device, "y_sorted")
    head = SortingHead(cfg.latent_dim, cfg.sorting_length_train)
    train_m = train_regression_head(head, z_train, y_train, epochs=cfg.head_epochs, lr=cfg.head_lr)
    test_m = evaluate_regression_head(head, z_test, y_test)
    rows.append({"study": "study2", "task": "sorting_head", "seed": cfg.seed, **train_m, **test_m})

    df = pd.DataFrame(rows)
    df.to_csv(outdir / f"study2_seed{cfg.seed}.csv", index=False)
    write_manifest(outdir / f"manifest_seed{cfg.seed}.json", {"study": "study2", "seed": cfg.seed, "rows": len(df), "environment": environment_report()})
    return df
