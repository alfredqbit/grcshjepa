from __future__ import annotations

import numpy as np

from grcshjepa.routing.graph import RoutingGraphToy


def gini(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    if x.size == 0 or np.allclose(x, 0):
        return 0.0
    x = np.sort(np.maximum(x, 0))
    n = x.size
    return float((2 * np.arange(1, n + 1) @ x) / (n * x.sum()) - (n + 1) / n)


def routing_lengths(graph: RoutingGraphToy) -> np.ndarray:
    p = graph.positions
    s = graph.segments
    return np.linalg.norm(p[s[:, 0]] - p[s[:, 1]], axis=1)


def gate_entropy(graph: RoutingGraphToy, eps: float = 1e-8) -> float:
    g = np.clip(graph.gates, eps, 1 - eps)
    return float(np.mean(-(g * np.log(g) + (1 - g) * np.log(1 - g))))


def routing_surface(graph: RoutingGraphToy) -> dict[str, float]:
    lengths = routing_lengths(graph)
    active = graph.gates > 0.25
    tube_area = 2.0 * np.pi * graph.radii * lengths * graph.gates
    # crude junction correction: high-load graph nodes add local patch area
    node_load = np.zeros(len(graph.positions))
    for (i, j), load, gate in zip(graph.segments, graph.loads, graph.gates):
        node_load[i] += load * gate
        node_load[j] += load * gate
    junction_area = 0.02 * np.sum(np.sqrt(np.maximum(node_load, 0.0)))
    total_surface = float(np.sum(tube_area) + junction_area)
    delivered = float(np.sum(np.abs(graph.loads) * graph.gates))
    return {
        "variant": graph.variant,
        "segments": float(len(graph.segments)),
        "active_segments": float(np.sum(active)),
        "total_length": float(np.sum(lengths * graph.gates)),
        "tube_area": float(np.sum(tube_area)),
        "junction_area": float(junction_area),
        "surface": total_surface,
        "delivered_traffic": delivered,
        "normalized_surface": float(total_surface / (delivered + 1e-8)),
        "gate_entropy": gate_entropy(graph),
        "load_gini": gini(graph.loads * graph.gates),
    }
