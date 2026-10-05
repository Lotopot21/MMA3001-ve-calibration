"""Tests for the scattered data surface fitting methods."""

import numpy as np
import pytest

from vetuner.surface_fit import (
    SURFACE_METHODS,
    aggregate_to_cells,
    bilinear_scatter,
    blend_by_confidence,
    fit_surface,
    gaussian_process_surface,
    polynomial_surface,
    table_node_coords,
    thin_plate_spline,
    to_cell_coords,
)
from vetuner.ve_surface import DEFAULT_MAP_AXIS, DEFAULT_RPM_AXIS


@pytest.fixture
def scattered():
    """Sample of a known smooth surface over part of the table."""
    rng = np.random.default_rng(0)
    rpm = rng.uniform(1500.0, 6000.0, 4000)
    map_kpa = rng.uniform(30.0, 95.0, 4000)
    truth = 1.0 + 0.10 * np.sin(rpm / 1200.0) + 0.05 * (map_kpa / 100.0)
    return rpm, map_kpa, truth


def test_breakpoints_map_to_integer_coordinates():
    coords = to_cell_coords(
        np.array([DEFAULT_RPM_AXIS[6]]), np.array([DEFAULT_MAP_AXIS[2]])
    )
    assert coords[0, 0] == pytest.approx(6.0)
    assert coords[0, 1] == pytest.approx(2.0)


def test_midpoint_maps_to_half_a_cell():
    midpoint = 0.5 * (DEFAULT_RPM_AXIS[3] + DEFAULT_RPM_AXIS[4])
    coords = to_cell_coords(np.array([midpoint]), np.array([DEFAULT_MAP_AXIS[0]]))
    assert coords[0, 0] == pytest.approx(3.5)


def test_node_coords_cover_the_whole_table():
    nodes = table_node_coords()
    assert nodes.shape == (len(DEFAULT_RPM_AXIS) * len(DEFAULT_MAP_AXIS), 2)
    assert nodes[:, 0].max() == len(DEFAULT_RPM_AXIS) - 1


def test_aggregation_preserves_total_sample_count(scattered):
    rpm, map_kpa, values = scattered
    cells = aggregate_to_cells(values, rpm, map_kpa)
    assert cells.count.sum() == len(values)


def test_aggregation_reduces_point_count_by_orders_of_magnitude(scattered):
    rpm, map_kpa, values = scattered
    cells = aggregate_to_cells(values, rpm, map_kpa)
    assert len(cells.mean) < len(values) / 20


def test_aggregation_of_a_constant_has_zero_variance():
    rng = np.random.default_rng(1)
    rpm = rng.uniform(1500.0, 6000.0, 500)
    map_kpa = rng.uniform(30.0, 95.0, 500)
    cells = aggregate_to_cells(np.full(500, 1.25), rpm, map_kpa)
    assert np.allclose(cells.mean, 1.25)
    assert np.allclose(cells.variance, 0.0, atol=1e-12)


def test_bilinear_weights_sum_to_the_sample_count(scattered):
    rpm, map_kpa, values = scattered
    _, weights = bilinear_scatter(values, rpm, map_kpa)
    assert weights.sum() == pytest.approx(len(values))


def test_bilinear_recovers_a_constant(scattered):
    rpm, map_kpa, _ = scattered
    surface, weights = bilinear_scatter(np.full(len(rpm), 0.93), rpm, map_kpa)
    assert np.allclose(surface[weights > 1e-9], 0.93)


def test_bilinear_beats_nearest_node_on_a_linear_field():
    """Bilinear scatter preserves sub-cell position; nearest-node binning discards it.

    Compared on RMSE over interior nodes. Edge nodes are excluded because
    bilinear weighting is one-sided there: a boundary node receives no
    contribution from beyond the table, so the weighted mean is pulled inward.
    That is a boundary effect rather than a property of the method.
    """
    rng = np.random.default_rng(2)
    rpm = rng.uniform(DEFAULT_RPM_AXIS[0], DEFAULT_RPM_AXIS[-1], 20_000)
    map_kpa = rng.uniform(DEFAULT_MAP_AXIS[0], DEFAULT_MAP_AXIS[-1], 20_000)
    coords = to_cell_coords(rpm, map_kpa)
    values = 1.0 + 0.01 * coords[:, 0] + 0.02 * coords[:, 1]

    nodes = table_node_coords()
    exact = (1.0 + 0.01 * nodes[:, 0] + 0.02 * nodes[:, 1]).reshape(
        len(DEFAULT_RPM_AXIS), len(DEFAULT_MAP_AXIS)
    )

    bilinear, _ = bilinear_scatter(values, rpm, map_kpa)
    nearest = fit_surface("per_cell", values, rpm, map_kpa)

    interior = np.zeros(exact.shape, dtype=bool)
    interior[1:-1, 1:-1] = True

    def rmse(surface):
        return float(np.sqrt(np.mean((surface - exact)[interior] ** 2)))

    assert rmse(bilinear) < rmse(nearest)


def test_bilinear_is_one_sided_at_table_edges():
    """Edge nodes receive weight from one side only, so they are biased inward."""
    rng = np.random.default_rng(3)
    rpm = rng.uniform(DEFAULT_RPM_AXIS[0], DEFAULT_RPM_AXIS[-1], 20_000)
    map_kpa = rng.uniform(DEFAULT_MAP_AXIS[0], DEFAULT_MAP_AXIS[-1], 20_000)
    coords = to_cell_coords(rpm, map_kpa)
    values = 1.0 + 0.01 * coords[:, 0] + 0.02 * coords[:, 1]

    nodes = table_node_coords()
    exact = (1.0 + 0.01 * nodes[:, 0] + 0.02 * nodes[:, 1]).reshape(
        len(DEFAULT_RPM_AXIS), len(DEFAULT_MAP_AXIS)
    )

    bilinear, _ = bilinear_scatter(values, rpm, map_kpa)
    error = np.abs(bilinear - exact)

    interior = np.zeros(exact.shape, dtype=bool)
    interior[1:-1, 1:-1] = True

    assert error[~interior].max() > error[interior].max()


def test_polynomial_recovers_a_polynomial_field():
    rng = np.random.default_rng(3)
    rpm = rng.uniform(DEFAULT_RPM_AXIS[0], DEFAULT_RPM_AXIS[-1], 2000)
    map_kpa = rng.uniform(DEFAULT_MAP_AXIS[0], DEFAULT_MAP_AXIS[-1], 2000)
    coords = to_cell_coords(rpm, map_kpa)
    values = 1.0 + 0.02 * coords[:, 0] - 0.003 * coords[:, 0] ** 2 + 0.01 * coords[:, 1]

    nodes = table_node_coords()
    exact = (
        1.0 + 0.02 * nodes[:, 0] - 0.003 * nodes[:, 0] ** 2 + 0.01 * nodes[:, 1]
    ).reshape(len(DEFAULT_RPM_AXIS), len(DEFAULT_MAP_AXIS))

    fitted = polynomial_surface(values, rpm, map_kpa, degree=2)
    assert np.abs(fitted - exact).max() < 1e-9


def test_polynomial_needs_enough_samples():
    with pytest.raises(ValueError):
        polynomial_surface(np.ones(3), np.full(3, 3000.0), np.full(3, 60.0), degree=3)


def test_thin_plate_spline_fits_a_smooth_field(scattered):
    rpm, map_kpa, values = scattered
    surface = thin_plate_spline(values, rpm, map_kpa, smoothing=0.5)
    assert np.isfinite(surface).all()
    assert 0.8 < surface.mean() < 1.3


def test_gaussian_process_returns_finite_mean_and_positive_std(scattered):
    rpm, map_kpa, values = scattered
    surface, std = gaussian_process_surface(values, rpm, map_kpa)
    assert np.isfinite(surface).all()
    assert np.all(std >= 0.0)


def test_uncertainty_is_larger_away_from_the_data(scattered):
    """The property that makes the Gaussian process useful here."""
    rpm, map_kpa, values = scattered
    _, std = gaussian_process_surface(values, rpm, map_kpa)

    coords = to_cell_coords(rpm, map_kpa)
    covered = np.zeros(std.shape, dtype=bool)
    covered[
        np.clip(np.round(coords[:, 0]).astype(int), 0, std.shape[0] - 1),
        np.clip(np.round(coords[:, 1]).astype(int), 0, std.shape[1] - 1),
    ] = True

    assert std[~covered].mean() > 2.0 * std[covered].mean()


def test_blending_returns_the_estimate_where_confident():
    surface = np.full((3, 3), 1.20)
    std = np.full((3, 3), 1e-6)
    assert np.allclose(blend_by_confidence(surface, std), 1.20, atol=1e-6)


def test_blending_returns_the_default_where_uncertain():
    surface = np.full((3, 3), 1.20)
    std = np.full((3, 3), 10.0)
    assert np.allclose(blend_by_confidence(surface, std), 1.0, atol=1e-3)


def test_blending_rejects_mismatched_shapes():
    with pytest.raises(ValueError):
        blend_by_confidence(np.ones((2, 2)), np.ones((3, 3)))


def test_every_method_returns_a_correctly_shaped_surface(scattered):
    rpm, map_kpa, values = scattered
    for method in SURFACE_METHODS:
        surface = fit_surface(method, values, rpm, map_kpa)
        assert surface.shape == (len(DEFAULT_RPM_AXIS), len(DEFAULT_MAP_AXIS))
        assert np.isfinite(surface).all(), method


def test_unknown_method_raises(scattered):
    rpm, map_kpa, values = scattered
    with pytest.raises(ValueError):
        fit_surface("magic", values, rpm, map_kpa)