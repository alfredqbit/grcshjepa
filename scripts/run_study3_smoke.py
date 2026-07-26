from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path

from grcshjepa.config import ExperimentConfig
from grcshjepa.experiments.study3 import run_study3_smoke
from grcshjepa.utils import save_json


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/smoke.yaml")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()
    cfg = ExperimentConfig.from_yaml(args.config)
    if args.seed is not None:
        cfg = cfg.with_updates(seed=args.seed)
    if args.output_dir is not None:
        cfg = cfg.with_updates(output_dir=args.output_dir)
    Path(cfg.output_dir).mkdir(parents=True, exist_ok=True)
    save_json(asdict(cfg), Path(cfg.output_dir) / "resolved_config_study3.json")
    df = run_study3_smoke(cfg)
    print(df.head(20).to_string(index=False))
    print(f"Rows written: {len(df)}")


if __name__ == "__main__":
    main()
