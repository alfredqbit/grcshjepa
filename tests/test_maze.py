import numpy as np

from grcshjepa.tasks.maze import bfs_shortest_path, verify_maze_path


def test_bfs_shortest_path_open_grid():
    grid = np.zeros((4, 4), dtype=int)
    path = bfs_shortest_path(grid, (0, 0), (0, 3))
    assert path[0] == (0, 0)
    assert path[-1] == (0, 3)
    assert len(path) == 4


def test_maze_verifier_accepts_valid_path():
    grid = np.zeros((4, 4), dtype=int)
    path = [(0, 0), (0, 1), (0, 2), (0, 3)]
    assert verify_maze_path(grid, (0, 0), (0, 3), path)["valid"] is True


def test_maze_verifier_rejects_non_adjacent_path():
    grid = np.zeros((4, 4), dtype=int)
    bad_path = [(0, 0), (1, 1), (0, 3)]
    assert verify_maze_path(grid, (0, 0), (0, 3), bad_path)["valid"] is False
