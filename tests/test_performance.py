"""Tests for timing utilities and cost estimates."""

import numpy as np
import pytest

from vetuner.performance import (
    arithmetic_intensity,
    estimate_flops,
    nearest_cell_indices_fast,
    peak_memory_bytes,
    time_call,
)
from vetuner.surface_fit import SURFACE_METHODS, nearest_cell_indices
from vetuner.ve_surface import DEFAULT_MAP_AXIS, DEFAULT_RPM_AXIS


@pytest.fixture
def samples():
    rng = np.random.default_rng(0)
    return (
        rng.uniform(600.0, 7400.0, 5000),
        rng.uniform(10.0, 110.0, 5000),
    )


def test_fast_assignment_matches_the_reference(samples):
    """The optimised routine must produce identical output, not merely similar."""
    rpm, map_kpa = samples
    expected = nearest_cell_indices(rpm, map_kpa)
    actual = nearest_cell_indices_fast(rpm, map_kpa)
    assert np.array_equal(actual[0], expected[0])
    assert np.array_equal(actual[1], expected[1])


def test_fast_assignment_handles_exact_breakpoints():
    rpm_idx, map_idx = nearest_cell_indices_fast(
        DEFAULT_RPM_AXIS, np.full(len(DEFAULT_RPM_AXIS), DEFAULT_MAP_AXIS[3])
    )
    assert np.array_equal(rpm_idx, np.arange(len(DEFAULT_RPM_AXIS)))
    assert np.all(map_idx == 3)


def test_fast_assignment_clamps_outside_the_axes():
    rpm_idx, map_idx = nearest_cell_indices_fast(
        np.array([1.0, 1e6]), np.array([1.0, 1e6])
    )
    assert list(rpm_idx) == [0, len(DEFAULT_RPM_AXIS) - 1]
    assert list(map_idx) == [0, len(DEFAULT_MAP_AXIS) - 1]


def test_fast_assignment_rejects_a_degenerate_axis():
    with pytest.raises(ValueError):
        nearest_cell_indices_fast(np.array([3000.0]), np.array([50.0]), rpm_axis=[1000.0])


def test_fast_assignment_uses_less_memory(samples):
    """The broadcasting version allocates a full sample-by-breakpoint matrix."""
    rpm, map_kpa = samples
    _, reference_peak = peak_memory_bytes(nearest_cell_indices, rpm, map_kpa)
    _, fast_peak = peak_memory_bytes(nearest_cell_indices_fast, rpm, map_kpa)
    assert fast_peak < reference_peak


def test_timing_orders_best_median_and_worst():
    timing = time_call(lambda: sum(range(10_000)), repeats=5)
    assert timing.best_ms <= timing.median_ms <= timing.worst_ms
    assert timing.repeats == 5


def test_timing_detects_a_slower_function():
    fast = time_call(lambda: sum(range(1_000)), repeats=5)
    slow = time_call(lambda: sum(range(200_000)), repeats=5)
    assert slow.best_ms > fast.best_ms


def test_timing_rejects_zero_repeats():
    with pytest.raises(ValueError):
        time_call(lambda: None, repeats=0)


def test_every_method_has_a_flop_estimate():
    for method in SURFACE_METHODS:
        assert estimate_flops(method, 5000, 192, 34) > 0.0


def test_unknown_method_has_no_estimate():
    with pytest.raises(ValueError):
        estimate_flops("magic", 5000, 192, 34)


def test_linear_methods_scale_linearly_with_samples():
    small = estimate_flops("bilinear", 1_000, 192, 34)
    large = estimate_flops("bilinear", 10_000, 192, 34)
    assert large / small == pytest.approx(10.0, rel=0.01)


def test_gaussian_process_cost_grows_faster_than_linearly_with_visited_cells():
    """The cubic solve dominates once enough cells are visited.

    The estimate also contains a term linear in sample count, so the ratio
    falls short of the factor of eight that a purely cubic cost would give at
    twice the cell count.
    """
    small = estimate_flops("gaussian_process", 5000, 192, 20)
    large = estimate_flops("gaussian_process", 5000, 192, 40)
    assert large / small > 4.0

    bigger = estimate_flops("gaussian_process", 5000, 192, 80)
    assert bigger / large > large / small


def test_gaussian_process_costs_far_more_than_bilinear():
    cheap = estimate_flops("bilinear", 5000, 192, 34)
    expensive = estimate_flops("gaussian_process", 5000, 192, 34)
    assert expensive / cheap > 10.0


def test_arithmetic_intensity_divides_flops_by_bytes():
    assert arithmetic_intensity(1000.0, 250.0) == pytest.approx(4.0)


def test_arithmetic_intensity_rejects_zero_bytes():
    with pytest.raises(ValueError):
        arithmetic_intensity(1000.0, 0.0)