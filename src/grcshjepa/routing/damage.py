from __future__ import annotations

import numpy as np

from grcshjepa.routing.graph import RoutingGraphToy


def apply_damage(graph: RoutingGraphToy, damage_type: str, level: float, seed: int) -> RoutingGraphToy:
    rng = np.random.default_rng(seed)
    g = graph.copy()
    n = len(g.segments)
    k = max(1, int(round(level * n)))

    if damage_type == "uniform":
        idx = rng.choice(n, size=k, replace=False)
    elif damage_type == "spatial":
        center = rng.random(3)
        mids = (g.positions[g.segments[:, 0]] + g.positions[g.segments[:, 1]]) / 2
        dist = np.linalg.norm(mids - center, axis=1)
        idx = np.argsort(dist)[:k]
    elif damage_type == "load_targeted":
        idx = np.argsort(g.loads * g.gates)[-k:]
    else:
        raise ValueError(f"Unknown damage_type: {damage_type}")

    g.gates[idx] = 0.0
    g.loads[idx] = 0.0
    g.radii[idx] = 0.0
    return g
