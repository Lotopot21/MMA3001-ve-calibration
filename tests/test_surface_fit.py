"""Tests for scattered data surface fitting."""

import numpy as np
import pytest

from vetuner.surface_fit import cell_sample_counts, nearest_cell_indices, per_cell_average
from vetuner.ve_surface import DEFAULT_MAP_AXIS, DEFAULT_RPM_AXIS


def test_samples_on_breakpoints_map_to_their_own_node():
    rpm_idx, map_idx = nearest_cell_indices(
        np.array([DEFAULT_RPM_AXIS[4]]), np.array([DEFAULT_MAP_AXIS[7]])
    )
    assert rpm_idx[0] == 4
    assert map_idx[0] == 7


def test_samples_outside_axes_clamp_to_edges():
    rpm_idx, map_idx = nearest_cell_indices(
        np.array([100.0, 99_000.0]), np.array([1.0, 500.0])
    )
    assert list(rpm_idx) == [0, len(DEFAULT_RPM_AXIS) - 1]
    assert list(map_idx) == [0, len(DEFAULT_MAP_AXIS) - 1]


def test_counts_sum_to_number_of_samples():
    rng = np.random.default_rng(0)
    rpm = rng.uniform(800.0, 7000.0, 5000)
    map_kpa = rng.uniform(20.0, 100.0, 5000)
    assert cell_sample_counts(rpm, map_kpa).sum() == 5000


def test_average_recovers_constant_exactly():
    rng = np.random.default_rng(1)
    rpm = rng.uniform(800.0, 7000.0, 3000)
    map_kpa = rng.uniform(20.0, 100.0, 3000)
    surface, counts = per_cell_average(np.full(3000, 1.07), rpm, map_kpa)
    assert np.allclose(surface[counts > 0], 1.07)


def test_unvisited_cells_receive_the_default():
    rpm = np.array([DEFAULT_RPM_AXIS[0]])
    map_kpa = np.array([DEFAULT_MAP_AXIS[0]])
    surface, counts = per_cell_average(np.array([2.0]), rpm, map_kpa, default=1.0)
    assert surface[0, 0] == 2.0
    assert np.allclose(surface[counts == 0], 1.0)


def test_averaging_reduces_noise_with_sample_count():
    rng = np.random.default_rng(2)
    n = 40_000
    rpm = np.full(n, DEFAULT_RPM_AXIS[5])
    map_kpa = np.full(n, DEFAULT_MAP_AXIS[5])
    noisy = 1.0 + rng.normal(0.0, 0.1, n)
    surface, counts = per_cell_average(noisy, rpm, map_kpa)
    standard_error = 0.1 / np.sqrt(counts[5, 5])
    assert abs(surface[5, 5] - 1.0) < 4.0 * standard_error


def test_mismatched_value_length_raises():
    with pytest.raises(ValueError):
        per_cell_average(np.ones(5), np.ones(6), np.ones(6))