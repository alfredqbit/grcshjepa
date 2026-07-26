from pathlib import Path

from grcshjepa.config import ExperimentConfig
from grcshjepa.experiments.study1 import run_study1_smoke
from grcshjepa.experiments.study3 import run_study3_smoke


def tiny_cfg(tmp_path: Path) -> ExperimentConfig:
    return ExperimentConfig(
        output_dir=str(tmp_path),
        train_samples=16,
        val_samples=8,
        test_samples=8,
        batch_size=8,
        epochs=1,
        head_epochs=1,
        maze_size_train=6,
        sorting_length_train=6,
        latent_dim=12,
        projection_dim=6,
        n_neurons=16,
        internal_steps=2,
        routing_nodes=12,
        routing_segments=20,
    )


def test_study1_smoke_runs(tmp_path):
    df = run_study1_smoke(tiny_cfg(tmp_path))
    assert not df.empty
    assert "val_pred_loss" in df.columns


def test_study3_smoke_runs(tmp_path):
    df = run_study3_smoke(tiny_cfg(tmp_path))
    assert not df.empty
    assert "normalized_surface" in df.columns
