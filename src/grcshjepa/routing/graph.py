from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class RoutingGraphToy:
    positions: np.ndarray
    segments: np.ndarray
    gates: np.ndarray
    loads: np.ndarray
    radii: np.ndarray
    variant: str

    def copy(self) -> "RoutingGraphToy":
        return RoutingGraphToy(
            positions=self.positions.copy(),
            segments=self.segments.copy(),
            gates=self.gates.copy(),
            loads=self.loads.copy(),
            radii=self.radii.copy(),
            variant=self.variant,
        )


def make_toy_routing_graph(
    n_nodes: int,
    n_segments: int,
    seed: int,
    variant: str = "full_surface",
) -> RoutingGraphToy:
    rng = np.random.default_rng(seed)
    positions = rng.random((n_nodes, 3))
    segments = set()
    # connect local nearest-ish random pairs
    while len(segments) < n_segments:
        i, j = rng.choice(n_nodes, size=2, replace=False)
        a, b = (int(i), int(j)) if i < j else (int(j), int(i))
        segments.add((a, b))
    seg = np.array(sorted(segments), dtype=np.int64)
    lengths = np.linalg.norm(positions[seg[:, 0]] - positions[seg[:, 1]], axis=1)
    base_loads = rng.gamma(shape=2.0, scale=1.0, size=len(seg))

    if variant == "sparsity":
        gates = rng.beta(2.0, 4.0, size=len(seg))
        loads = base_loads * gates
    elif variant == "euclidean_length":
        gates = np.exp(-2.0 * lengths)
        loads = base_loads * gates
    elif variant == "tube_only":
        gates = rng.beta(3.0, 2.0, size=len(seg))
        loads = base_loads * gates
    else:  # full_surface bundles more load into shared segments
        centrality = 1.0 / (0.2 + np.linalg.norm((positions[seg[:, 0]] + positions[seg[:, 1]]) / 2 - 0.5, axis=1))
        gates = 1.0 / (1.0 + np.exp(-0.8 * (centrality - np.median(centrality))))
        loads = (base_loads + centrality) * gates

    radii = 0.015 + 0.025 * np.sqrt(np.maximum(loads, 0.0))
    return RoutingGraphToy(positions=positions, segments=seg, gates=gates, loads=loads, radii=radii, variant=variant)
