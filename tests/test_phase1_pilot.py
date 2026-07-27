
from pathlib import Path

from grcshjepa.config import ExperimentConfig
from grcshjepa.pilot.runner import load_yaml


def test_phase1_meta_config_loads():
    meta = load_yaml("configs/phase1_pilot.yaml")
    assert "seed_list" in meta
    assert "study1" in meta["studies"]


def test_pilot_study_configs_are_experiment_configs():
    for path in ["configs/study1_pilot.yaml", "configs/study2_pilot.yaml", "configs/study3_pilot.yaml"]:
        cfg = ExperimentConfig.from_yaml(path)
        assert cfg.train_samples > 0
        assert cfg.epochs >= 1
