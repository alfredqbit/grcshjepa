import numpy as np

from grcshjepa.tasks.sorting import bubble_sort_trajectory, inversion_count, verify_sorted_output


def test_bubble_sort_trajectory_ends_sorted():
    seq = np.array([3, 1, 2, 0], dtype=float)
    states = bubble_sort_trajectory(seq)
    assert np.allclose(states[-1], np.sort(seq))


def test_inversion_count():
    assert inversion_count(np.array([3, 2, 1])) == 3


def test_verify_sorted_output():
    target = np.array([0, 1, 2])
    assert verify_sorted_output(target.copy(), target)["exact"] is True
