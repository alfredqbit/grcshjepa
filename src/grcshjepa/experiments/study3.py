from __future__ import annotations

from pathlib import Path

import pandas as pd

from grcshjepa.config import ExperimentConfig
from grcshjepa.routing.graph import make_toy_routing_graph
from grcshjepa.routing.surface import routing_surface
from grcshjepa.routing.damage import apply_damage
from grcshjepa.utils import set_seed, write_manifest, environment_report


def run_study3_smoke(cfg: ExperimentConfig) -> pd.DataFrame:
    """Run Study 3 smoke: routing-surface metrics and cable-damage interventions."""
    set_seed(cfg.seed)
    outdir = Path(cfg.output_dir) / "study3_smoke"
    outdir.mkdir(parents=True, exist_ok=True)
    rows = []
    variants = ["sparsity", "euclidean_length", "tube_only", "full_surface"]
    damage_types = ["uniform", "spatial", "load_targeted"]
    for variant in variants:
        graph = make_toy_routing_graph(cfg.routing_nodes, cfg.routing_segments, cfg.seed, variant)
        base = routing_surface(graph)
        rows.append({"study": "study3", "variant": variant, "damage_type": "none", "damage_level": 0.0, **base})
        for dtype in damage_types:
            for level in cfg.damage_levels:
                damaged = apply_damage(graph, dtype, level, seed=cfg.seed + int(level * 1000) + len(dtype))
                metrics = routing_surface(damaged)
                metrics["surface_degradation"] = metrics["normalized_surface"] - base["normalized_surface"]
                metrics["traffic_degradation"] = base["delivered_traffic"] - metrics["delivered_traffic"]
                rows.append({"study": "study3", "variant": variant, "damage_type": dtype, "damage_level": level, **metrics})
    df = pd.DataFrame(rows)
    df.to_csv(outdir / f"study3_seed{cfg.seed}.csv", index=False)
    write_manifest(outdir / f"manifest_seed{cfg.seed}.json", {"study": "study3", "seed": cfg.seed, "rows": len(df), "environment": environment_report()})
    return df
