from __future__ import annotations

from collections import deque
from typing import List, Optional, Tuple

import numpy as np

Cell = Tuple[int, int]
Path = List[Cell]
DIRS: list[Cell] = [(-1, 0), (1, 0), (0, -1), (0, 1)]
DIR_TO_ID = {d: i for i, d in enumerate(DIRS)}


def bfs_shortest_path(grid: np.ndarray, start: Cell, goal: Cell) -> Optional[Path]:
    """Return a shortest path on a binary obstacle grid, or None if unreachable.

    A value of 1 means wall; 0 means free.
    """
    rows, cols = grid.shape
    queue: deque[Cell] = deque([start])
    parent: dict[Cell, Cell | None] = {start: None}

    while queue:
        r, c = queue.popleft()
        if (r, c) == goal:
            path: Path = []
            cur: Cell | None = goal
            while cur is not None:
                path.append(cur)
                cur = parent[cur]
            return list(reversed(path))

        for dr, dc in DIRS:
            nr, nc = r + dr, c + dc
            nxt = (nr, nc)
            if 0 <= nr < rows and 0 <= nc < cols and grid[nr, nc] == 0 and nxt not in parent:
                parent[nxt] = (r, c)
                queue.append(nxt)
    return None


def make_random_maze(
    size: int,
    obstacle_prob: float,
    rng: np.random.Generator,
    max_tries: int = 200,
) -> tuple[np.ndarray, Cell, Cell, Path]:
    """Generate a reachable random maze with a nontrivial shortest path."""
    for _ in range(max_tries):
        grid = (rng.random((size, size)) < obstacle_prob).astype(np.int64)
        start = (int(rng.integers(0, size)), int(rng.integers(0, size)))
        goal = (int(rng.integers(0, size)), int(rng.integers(0, size)))
        if start == goal:
            continue
        grid[start] = 0
        grid[goal] = 0
        path = bfs_shortest_path(grid, start, goal)
        if path is not None and len(path) >= max(4, size // 2):
            return grid, start, goal, path

    grid = np.zeros((size, size), dtype=np.int64)
    start, goal = (0, 0), (size - 1, size - 1)
    path = bfs_shortest_path(grid, start, goal)
    if path is None:
        raise RuntimeError("Empty fallback grid should be reachable.")
    return grid, start, goal, path


def maze_observation(grid: np.ndarray, pos: Cell, goal: Cell) -> np.ndarray:
    """Four-channel observation: walls, position, goal, normalized goal-distance field."""
    rows, cols = grid.shape
    obs = np.zeros((4, rows, cols), dtype=np.float32)
    obs[0] = grid.astype(np.float32)
    obs[1, pos[0], pos[1]] = 1.0
    obs[2, goal[0], goal[1]] = 1.0
    rr, cc = np.meshgrid(np.arange(rows), np.arange(cols), indexing="ij")
    dist = np.abs(rr - goal[0]) + np.abs(cc - goal[1])
    obs[3] = dist.astype(np.float32) / max(1.0, float(dist.max()))
    return obs


def path_mask(size: int, path: Path) -> np.ndarray:
    mask = np.zeros((size, size), dtype=np.float32)
    for r, c in path:
        mask[r, c] = 1.0
    return mask


def next_action_from_path(path: Path, idx: int) -> int:
    if idx >= len(path) - 1:
        return 0
    r, c = path[idx]
    nr, nc = path[idx + 1]
    return DIR_TO_ID[(nr - r, nc - c)]


def verify_maze_path(grid: np.ndarray, start: Cell, goal: Cell, path: Path) -> dict:
    """Verify that a path starts/ends correctly, uses adjacent moves, and avoids walls."""
    if not path:
        return {"valid": False, "reason": "empty_path"}
    if path[0] != start:
        return {"valid": False, "reason": "wrong_start"}
    if path[-1] != goal:
        return {"valid": False, "reason": "wrong_goal"}

    rows, cols = grid.shape
    for idx, (r, c) in enumerate(path):
        if not (0 <= r < rows and 0 <= c < cols):
            return {"valid": False, "reason": f"out_of_bounds_{idx}"}
        if grid[r, c] != 0:
            return {"valid": False, "reason": f"wall_{idx}"}
    for idx, ((r1, c1), (r2, c2)) in enumerate(zip(path, path[1:])):
        if abs(r1 - r2) + abs(c1 - c2) != 1:
            return {"valid": False, "reason": f"non_adjacent_{idx}"}
    return {"valid": True, "reason": "valid", "length": len(path)}


def is_rare_critical_maze_state(grid: np.ndarray, path: Path, idx: int) -> bool:
    """Simple rare-critical flag for chokepoints, turns, and low-branching path states."""
    r, c = path[idx]
    free_neighbors = 0
    for dr, dc in DIRS:
        nr, nc = r + dr, c + dc
        if 0 <= nr < grid.shape[0] and 0 <= nc < grid.shape[1] and grid[nr, nc] == 0:
            free_neighbors += 1
    turn = False
    if 0 < idx < len(path) - 1:
        prev = np.array(path[idx]) - np.array(path[idx - 1])
        nxt = np.array(path[idx + 1]) - np.array(path[idx])
        turn = not np.array_equal(prev, nxt)
    return bool(free_neighbors <= 2 or turn)
