
from __future__ import annotations

import argparse
import json
import time
import traceback
from pathlib import Path
from typing import Callable, Iterable

import pandas as pd
import torch
import yaml
from torch.utils.data import DataLoader

from grcshjepa.config import ExperimentConfig
from grcshjepa.data.datasets import MazeTrajectoryDataset, SortingTrajectoryDataset
from grcshjepa.models.encoders import MazeEncoder, SortingEncoder
from grcshjepa.models.hjepa import HJEPA
from grcshjepa.models.predictors import MLPPredictor, RelaxedContinuousSpikingPredictor
from grcshjepa.models.heads import MazeActionHead, SortingHead
from grcshjepa.routing.graph import make_toy_routing_graph
from grcshjepa.routing.surface import routing_surface
from grcshjepa.routing.damage import apply_damage
from grcshjepa.training.pretrain import train_jepa_epoch, evaluate_jepa
from grcshjepa.training.downstream import (
    make_labeled_subset,
    encode_dataset,
    train_classification_head,
    evaluate_classification_head,
    train_regression_head,
    evaluate_regression_head,
)
from grcshjepa.utils import (
    archive_directory,
    config_hash,
    environment_report,
    resolve_device,
    save_json,
    set_seed,
    write_manifest,
)


def load_yaml(path: str | Path) -> dict:
    path = Path(path)
    data = yaml.safe_load(path.read_text()) if path.exists() else {}
    return {} if data is None else dict(data)


def _fresh_encoder(task: str, cfg: ExperimentConfig):
    if task == "maze":
        return MazeEncoder(cfg.latent_dim)
    if task == "sorting":
        return SortingEncoder(cfg.sorting_length_train, cfg.latent_dim)
    raise ValueError(f"Unknown task: {task}")


def _make_hjepa(cfg: ExperimentConfig, task: str, predictor_type: str) -> HJEPA:
    max_horizon = max(cfg.horizon_set)
    horizon_dim = 16
    encoder = _fresh_encoder(task, cfg)
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
    elif predictor_type == "mlp":
        predictor = MLPPredictor(latent_dim=cfg.latent_dim, horizon_dim=horizon_dim)
    else:
        raise ValueError(f"Unknown predictor_type: {predictor_type}")
    return HJEPA(
        encoder=encoder,
        latent_dim=cfg.latent_dim,
        projection_dim=cfg.projection_dim,
        max_horizon=max_horizon,
        predictor=predictor,
        horizon_dim=horizon_dim,
    )


def _dataset(task: str, cfg: ExperimentConfig, split: str, seed_offset: int):
    seed = cfg.seed + seed_offset
    if task == "maze":
        n = {"train": cfg.train_samples, "val": cfg.val_samples, "test": cfg.test_samples}[split]
        return MazeTrajectoryDataset(n, cfg.maze_size_train, cfg.maze_obstacle_prob, seed, cfg.horizon_set)
    if task == "sorting":
        n = {"train": cfg.train_samples, "val": cfg.val_samples, "test": cfg.test_samples}[split]
        return SortingTrajectoryDataset(n, cfg.sorting_length_train, seed, cfg.horizon_set)
    raise ValueError(f"Unknown task: {task}")


def _manifest_payload(study: str, cfg: ExperimentConfig, rows: int, status: str, started: float, failure: str | None = None) -> dict:
    return {
        "phase": "phase1_pilot",
        "study": study,
        "seed": cfg.seed,
        "status": status,
        "rows": rows,
        "config_hash": config_hash(cfg),
        "config": cfg.to_dict(),
        "runtime_seconds": round(time.time() - started, 3),
        "failure": failure,
        "environment": environment_report(),
    }


def run_study1_pilot_seed(cfg: ExperimentConfig) -> pd.DataFrame:
    """Pilot Study 1: predictive pretraining diagnostics across task and predictor variants."""
    started = time.time()
    set_seed(cfg.seed)
    device = resolve_device(cfg.device)
    outdir = Path(cfg.output_dir) / "study1" / f"seed_{cfg.seed:03d}"
    outdir.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []

    for task in ["maze", "sorting"]:
        train_ds = _dataset(task, cfg, "train", 11 if task == "maze" else 21)
        val_ds = _dataset(task, cfg, "val", 111 if task == "maze" else 211)
        train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=cfg.batch_size)

        for predictor_type in ["mlp", "spiking"]:
            model = _make_hjepa(cfg, task, predictor_type).to(device)
            opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr)
            metrics: dict[str, float] = {}
            for epoch in range(cfg.epochs):
                train_metrics = train_jepa_epoch(
                    model, train_loader, opt, device,
                    lambda_ac=cfg.lambda_ac,
                    ema_tau=cfg.ema_tau,
                )
                val_metrics = evaluate_jepa(model, val_loader, device)
                metrics = {
                    **{f"train_{k}": v for k, v in train_metrics.items()},
                    **{f"val_{k}": v for k, v in val_metrics.items()},
                }
            rows.append({
                "phase": "phase1_pilot",
                "study": "study1",
                "task": task,
                "predictor": predictor_type,
                "seed": cfg.seed,
                "status": "complete",
                "epochs": cfg.epochs,
                "train_samples": cfg.train_samples,
                "val_samples": cfg.val_samples,
                **metrics,
            })

    df = pd.DataFrame(rows)
    df.to_csv(outdir / f"study1_seed{cfg.seed:03d}.csv", index=False)
    write_manifest(outdir / f"manifest_seed{cfg.seed:03d}.json", _manifest_payload("study1", cfg, len(df), "complete", started))
    return df


def _pretrain_for_downstream(cfg: ExperimentConfig, task: str, device: torch.device) -> HJEPA:
    train_ds = _dataset(task, cfg, "train", 31 if task == "maze" else 41)
    loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True)
    model = _make_hjepa(cfg, task, "spiking").to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr)
    for _ in range(cfg.epochs):
        train_jepa_epoch(model, loader, opt, device, lambda_ac=cfg.lambda_ac, ema_tau=cfg.ema_tau)
    return model


def run_study2_pilot_seed(cfg: ExperimentConfig) -> pd.DataFrame:
    """Pilot Study 2: frozen-backbone downstream heads under limited labels."""
    started = time.time()
    set_seed(cfg.seed)
    device = resolve_device(cfg.device)
    outdir = Path(cfg.output_dir) / "study2" / f"seed_{cfg.seed:03d}"
    outdir.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []

    # Maze action head.
    maze_model = _pretrain_for_downstream(cfg, "maze", device)
    train_maze = _dataset("maze", cfg, "train", 51)
    test_maze = _dataset("maze", cfg, "test", 52)
    labeled_maze = make_labeled_subset(train_maze, cfg.downstream_label_fraction, cfg.seed)
    z_train, y_train = encode_dataset(maze_model.encoder, DataLoader(labeled_maze, batch_size=cfg.batch_size), device, "action")
    z_test, y_test = encode_dataset(maze_model.encoder, DataLoader(test_maze, batch_size=cfg.batch_size), device, "action")
    head = MazeActionHead(cfg.latent_dim)
    train_m = train_classification_head(head, z_train, y_train, epochs=cfg.head_epochs, lr=cfg.head_lr)
    test_m = evaluate_classification_head(head, z_test, y_test)
    rows.append({
        "phase": "phase1_pilot",
        "study": "study2",
        "task": "maze_action",
        "head_type": "classification",
        "seed": cfg.seed,
        "status": "complete",
        "label_fraction": cfg.downstream_label_fraction,
        "head_epochs": cfg.head_epochs,
        **train_m,
        **test_m,
    })

    # Sorting regression head.
    sort_model = _pretrain_for_downstream(cfg, "sorting", device)
    train_sort = _dataset("sorting", cfg, "train", 61)
    test_sort = _dataset("sorting", cfg, "test", 62)
    labeled_sort = make_labeled_subset(train_sort, cfg.downstream_label_fraction, cfg.seed)
    z_train, y_train = encode_dataset(sort_model.encoder, DataLoader(labeled_sort, batch_size=cfg.batch_size), device, "y_sorted")
    z_test, y_test = encode_dataset(sort_model.encoder, DataLoader(test_sort, batch_size=cfg.batch_size), device, "y_sorted")
    head = SortingHead(cfg.latent_dim, cfg.sorting_length_train)
    train_m = train_regression_head(head, z_train, y_train, epochs=cfg.head_epochs, lr=cfg.head_lr)
    test_m = evaluate_regression_head(head, z_test, y_test)
    rows.append({
        "phase": "phase1_pilot",
        "study": "study2",
        "task": "sorting_head",
        "head_type": "regression",
        "seed": cfg.seed,
        "status": "complete",
        "label_fraction": cfg.downstream_label_fraction,
        "head_epochs": cfg.head_epochs,
        **train_m,
        **test_m,
    })

    df = pd.DataFrame(rows)
    df.to_csv(outdir / f"study2_seed{cfg.seed:03d}.csv", index=False)
    write_manifest(outdir / f"manifest_seed{cfg.seed:03d}.json", _manifest_payload("study2", cfg, len(df), "complete", started))
    return df


def run_study3_pilot_seed(cfg: ExperimentConfig) -> pd.DataFrame:
    """Pilot Study 3: routing-surface metrics and cable-damage interventions."""
    started = time.time()
    set_seed(cfg.seed)
    outdir = Path(cfg.output_dir) / "study3" / f"seed_{cfg.seed:03d}"
    outdir.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []

    variants = ["sparsity", "euclidean_length", "tube_only", "full_surface"]
    damage_types = ["uniform", "spatial", "load_targeted"]
    for variant in variants:
        graph = make_toy_routing_graph(cfg.routing_nodes, cfg.routing_segments, cfg.seed, variant)
        base = routing_surface(graph)
        rows.append({
            "phase": "phase1_pilot",
            "study": "study3",
            "variant": variant,
            "damage_type": "none",
            "damage_level": 0.0,
            "seed": cfg.seed,
            "status": "complete",
            **base,
        })
        base_norm = float(base.get("normalized_surface", float("nan")))
        base_traffic = float(base.get("delivered_traffic", float("nan")))
        for dtype in damage_types:
            for level in cfg.damage_levels:
                damaged = apply_damage(graph, dtype, level, seed=cfg.seed + int(level * 1000) + len(dtype))
                metrics = routing_surface(damaged)
                metrics["surface_degradation"] = float(metrics["normalized_surface"] - base_norm)
                metrics["traffic_degradation"] = float(base_traffic - metrics["delivered_traffic"])
                rows.append({
                    "phase": "phase1_pilot",
                    "study": "study3",
                    "variant": variant,
                    "damage_type": dtype,
                    "damage_level": float(level),
                    "seed": cfg.seed,
                    "status": "complete",
                    **metrics,
                })

    df = pd.DataFrame(rows)
    df.to_csv(outdir / f"study3_seed{cfg.seed:03d}.csv", index=False)
    write_manifest(outdir / f"manifest_seed{cfg.seed:03d}.json", _manifest_payload("study3", cfg, len(df), "complete", started))
    return df


def _failure_frame(study: str, seed: int, error: BaseException) -> pd.DataFrame:
    return pd.DataFrame([{
        "phase": "phase1_pilot",
        "study": study,
        "seed": seed,
        "status": "failure",
        "failure_type": type(error).__name__,
        "failure_message": str(error),
    }])


def _run_one_safely(study: str, cfg: ExperimentConfig, fn: Callable[[ExperimentConfig], pd.DataFrame]) -> pd.DataFrame:
    started = time.time()
    outdir = Path(cfg.output_dir) / study / f"seed_{cfg.seed:03d}"
    outdir.mkdir(parents=True, exist_ok=True)
    try:
        return fn(cfg)
    except BaseException as exc:  # keep pilot loop going and record the failure
        tb = traceback.format_exc()
        (outdir / f"failure_seed{cfg.seed:03d}.txt").write_text(tb)
        write_manifest(outdir / f"manifest_seed{cfg.seed:03d}.json", _manifest_payload(study, cfg, 1, "failure", started, tb))
        df = _failure_frame(study, cfg.seed, exc)
        df.to_csv(outdir / f"{study}_seed{cfg.seed:03d}_failure.csv", index=False)
        return df


def run_phase1_pilot(
    meta_config_path: str | Path,
    output_dir: str | Path | None = None,
    studies_override: Iterable[str] | None = None,
    max_seeds: int | None = None,
    archive: bool = False,
) -> pd.DataFrame:
    meta = load_yaml(meta_config_path)
    root = Path(output_dir or meta.get("output_dir", "runs/phase1_pilot"))
    root.mkdir(parents=True, exist_ok=True)
    studies = list(studies_override or meta.get("studies", ["study1", "study2", "study3"]))
    seeds = [int(s) for s in meta.get("seed_list", [0, 1, 2, 3])]
    if max_seeds is not None:
        seeds = seeds[: int(max_seeds)]

    config_key = {
        "study1": "study1_config",
        "study2": "study2_config",
        "study3": "study3_config",
    }
    runner = {
        "study1": run_study1_pilot_seed,
        "study2": run_study2_pilot_seed,
        "study3": run_study3_pilot_seed,
    }

    frames: list[pd.DataFrame] = []
    for seed in seeds:
        for study in studies:
            if study not in runner:
                raise ValueError(f"Unknown study: {study}")
            cfg_path = meta[config_key[study]]
            cfg = ExperimentConfig.from_yaml(cfg_path).with_updates(seed=seed, output_dir=str(root))
            print(f"[phase1] running {study} seed={seed} config_hash={config_hash(cfg)}")
            df = _run_one_safely(study, cfg, runner[study])
            frames.append(df)

    combined = pd.concat(frames, ignore_index=True, sort=False) if frames else pd.DataFrame()
    combined_path = root / "phase1_pilot_combined_results.csv"
    combined.to_csv(combined_path, index=False)
    write_manifest(root / "phase1_pilot_manifest.json", {
        "phase": "phase1_pilot",
        "status": "complete",
        "meta_config": meta,
        "seed_list": seeds,
        "studies": studies,
        "rows": len(combined),
        "combined_results": str(combined_path),
        "environment": environment_report(),
    })
    if archive:
        archive_name = meta.get("archive_name", "phase1_pilot.tar.gz")
        archive_path = archive_directory(root, root.parent / archive_name)
        print(f"[phase1] archived to {archive_path}")
    return combined


def _parse_studies(value: str | None):
    if value is None or value.strip() == "":
        return None
    return [part.strip() for part in value.split(",") if part.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Phase 1 pilot hardening experiments.")
    parser.add_argument("--meta-config", default="configs/phase1_pilot.yaml")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--studies", default=None, help="Comma-separated subset, e.g. study1,study3")
    parser.add_argument("--max-seeds", type=int, default=None, help="Use only the first N seeds for quick checks")
    parser.add_argument("--archive", action="store_true")
    args = parser.parse_args()
    df = run_phase1_pilot(
        args.meta_config,
        output_dir=args.output_dir,
        studies_override=_parse_studies(args.studies),
        max_seeds=args.max_seeds,
        archive=args.archive,
    )
    print(df.head(40).to_string(index=False))
    print(f"Rows written: {len(df)}")


if __name__ == "__main__":
    main()
