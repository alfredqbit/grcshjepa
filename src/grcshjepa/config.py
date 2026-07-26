from __future__ import annotations

from dataclasses import dataclass, fields, asdict
from pathlib import Path
from typing import Any, Tuple

import yaml


@dataclass(frozen=True)
class ExperimentConfig:
    """Configuration for GR-CS-HJEPA Chapter 4 experiments."""

    project: str = "GR-CS-HJEPA-Chapter4"
    smoke_mode: bool = True
    seed: int = 0
    device: str = "auto"
    output_dir: str = "runs/smoke"

    # Procedural data
    maze_size_train: int = 8
    maze_size_ood: int = 10
    maze_obstacle_prob: float = 0.18
    sorting_length_train: int = 8
    sorting_length_ood: int = 10
    train_samples: int = 96
    val_samples: int = 32
    test_samples: int = 32

    # H-JEPA
    latent_dim: int = 32
    projection_dim: int = 12
    horizon_set: Tuple[int, ...] = (1, 2)
    batch_size: int = 16
    epochs: int = 1
    lr: float = 2e-3
    ema_tau: float = 0.99
    lambda_ac: float = 0.03
    lambda_suff: float = 0.0
    lambda_alias: float = 0.0
    lambda_unc: float = 0.0

    # Relaxed continuous-spiking predictor
    n_neurons: int = 48
    n_filters: int = 2
    internal_steps: int = 4
    dt: float = 0.08
    noise_std: float = 0.01
    surrogate_eps: float = 0.7

    # Downstream heads
    downstream_label_fraction: float = 0.20
    head_epochs: int = 1
    head_lr: float = 2e-3

    # Routing toy model
    routing_nodes: int = 28
    routing_segments: int = 72
    routing_terminals: int = 10
    damage_levels: Tuple[float, ...] = (0.05, 0.10)

    @classmethod
    def from_yaml(cls, path: str | Path) -> "ExperimentConfig":
        path = Path(path)
        data = yaml.safe_load(path.read_text()) if path.exists() else {}
        data = {} if data is None else dict(data)
        valid = {f.name for f in fields(cls)}
        unknown = sorted(set(data) - valid)
        if unknown:
            raise ValueError(f"Unknown config keys in {path}: {unknown}")
        if "horizon_set" in data and isinstance(data["horizon_set"], list):
            data["horizon_set"] = tuple(int(x) for x in data["horizon_set"])
        if "damage_levels" in data and isinstance(data["damage_levels"], list):
            data["damage_levels"] = tuple(float(x) for x in data["damage_levels"])
        return cls(**data)

    def with_updates(self, **kwargs: Any) -> "ExperimentConfig":
        data = asdict(self)
        data.update(kwargs)
        return ExperimentConfig(**data)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
