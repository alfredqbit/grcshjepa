from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from grcshjepa.config import ExperimentConfig
from grcshjepa.experiments.study1 import run_study1_smoke
from grcshjepa.experiments.study2 import run_study2_smoke
from grcshjepa.experiments.study3 import run_study3_smoke
from grcshjepa.utils import archive_directory


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/smoke.yaml")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--archive", action="store_true")
    args = parser.parse_args()
    cfg = ExperimentConfig.from_yaml(args.config)
    if args.seed is not None:
        cfg = cfg.with_updates(seed=args.seed)
    if args.output_dir is not None:
        cfg = cfg.with_updates(output_dir=args.output_dir)
    Path(cfg.output_dir).mkdir(parents=True, exist_ok=True)
    frames = [run_study1_smoke(cfg), run_study2_smoke(cfg), run_study3_smoke(cfg)]
    summary = pd.concat(frames, ignore_index=True, sort=False)
    summary.to_csv(Path(cfg.output_dir) / "all_smoke_summary.csv", index=False)
    print(summary.head(30).to_string(index=False))
    if args.archive:
        archive = archive_directory(cfg.output_dir, str(Path(cfg.output_dir).with_suffix(".tar.gz")))
        print(f"Archived: {archive}")


if __name__ == "__main__":
    main()
