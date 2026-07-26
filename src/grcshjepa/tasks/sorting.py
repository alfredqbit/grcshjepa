from __future__ import annotations

from typing import List, Tuple

import numpy as np


def inversion_count(seq: np.ndarray) -> int:
    seq = np.asarray(seq)
    count = 0
    for i in range(len(seq)):
        for j in range(i + 1, len(seq)):
            count += int(seq[i] > seq[j])
    return count


def bubble_sort_trajectory(seq: np.ndarray) -> List[np.ndarray]:
    """Return states produced by one standard bubble-sort trajectory."""
    arr = np.asarray(seq, dtype=np.float32).copy()
    states = [arr.copy()]
    n = len(arr)
    for _ in range(n):
        changed = False
        for j in range(n - 1):
            if arr[j] > arr[j + 1]:
                arr[j], arr[j + 1] = arr[j + 1], arr[j]
                states.append(arr.copy())
                changed = True
        if not changed:
            break
    return states


def random_sorting_example(length: int, rng: np.random.Generator) -> Tuple[np.ndarray, List[np.ndarray]]:
    seq = rng.permutation(length).astype(np.float32) / max(1, length - 1)
    return seq, bubble_sort_trajectory(seq)


def verify_sorted_output(pred: np.ndarray, target_sorted: np.ndarray, tol: float = 1e-3) -> dict:
    pred = np.asarray(pred)
    target_sorted = np.asarray(target_sorted)
    exact = bool(np.allclose(pred, target_sorted, atol=tol))
    monotone = bool(np.all(np.diff(pred) >= -tol))
    return {"exact": exact, "monotone": monotone, "mae": float(np.mean(np.abs(pred - target_sorted)))}
