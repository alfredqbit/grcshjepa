from __future__ import annotations

from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset

from grcshjepa.tasks.maze import (
    make_random_maze,
    maze_observation,
    next_action_from_path,
    is_rare_critical_maze_state,
)
from grcshjepa.tasks.sorting import random_sorting_example, inversion_count


class MazeTrajectoryDataset(Dataset):
    """Procedural maze context-target pairs for JEPA-style latent prediction."""

    def __init__(self, n_samples: int, size: int, obstacle_prob: float, seed: int, horizon_set=(1, 2)):
        self.n_samples = int(n_samples)
        self.size = int(size)
        self.obstacle_prob = float(obstacle_prob)
        self.horizon_set = tuple(int(h) for h in horizon_set)
        rng = np.random.default_rng(seed)
        self.items: list[dict[str, Any]] = []
        for _ in range(self.n_samples):
            grid, start, goal, path = make_random_maze(size, obstacle_prob, rng)
            idx = int(rng.integers(0, max(1, len(path) - 1)))
            horizon = int(rng.choice(self.horizon_set))
            future_idx = min(idx + horizon, len(path) - 1)
            x = maze_observation(grid, path[idx], goal)
            x_future = maze_observation(grid, path[future_idx], goal)
            action = next_action_from_path(path, idx)
            rare = is_rare_critical_maze_state(grid, path, idx)
            self.items.append(
                {
                    "x": x,
                    "x_future": x_future,
                    "horizon": horizon,
                    "action": action,
                    "rare": rare,
                    "path_len": len(path),
                    "grid": grid,
                    "start": start,
                    "goal": goal,
                    "path": path,
                }
            )

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        item = self.items[idx]
        return {
            "x": torch.tensor(item["x"], dtype=torch.float32),
            "x_future": torch.tensor(item["x_future"], dtype=torch.float32),
            "horizon": torch.tensor(item["horizon"], dtype=torch.long),
            "action": torch.tensor(item["action"], dtype=torch.long),
            "rare": torch.tensor(float(item["rare"]), dtype=torch.float32),
            "difficulty": torch.tensor(float(item["path_len"]), dtype=torch.float32),
        }


class SortingTrajectoryDataset(Dataset):
    """Sorting context-target pairs along a bubble-sort trajectory."""

    def __init__(self, n_samples: int, length: int, seed: int, horizon_set=(1, 2)):
        self.n_samples = int(n_samples)
        self.length = int(length)
        self.horizon_set = tuple(int(h) for h in horizon_set)
        rng = np.random.default_rng(seed)
        self.items: list[dict[str, Any]] = []
        for _ in range(self.n_samples):
            seq, trajectory = random_sorting_example(length, rng)
            idx = int(rng.integers(0, max(1, len(trajectory))))
            idx = min(idx, len(trajectory) - 1)
            horizon = int(rng.choice(self.horizon_set))
            future_idx = min(idx + horizon, len(trajectory) - 1)
            x = trajectory[idx]
            x_future = trajectory[future_idx]
            y_sorted = np.sort(seq)
            self.items.append(
                {
                    "x": x.astype(np.float32),
                    "x_future": x_future.astype(np.float32),
                    "horizon": horizon,
                    "y_sorted": y_sorted.astype(np.float32),
                    "difficulty": inversion_count(x),
                }
            )

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        item = self.items[idx]
        return {
            "x": torch.tensor(item["x"], dtype=torch.float32),
            "x_future": torch.tensor(item["x_future"], dtype=torch.float32),
            "horizon": torch.tensor(item["horizon"], dtype=torch.long),
            "y_sorted": torch.tensor(item["y_sorted"], dtype=torch.float32),
            "difficulty": torch.tensor(float(item["difficulty"]), dtype=torch.float32),
        }
