"""Tests for the analytic VE surfaces."""

import numpy as np
import pytest

from vetuner.ve_surface import (
    BASE_MAP_SURFACE,
    DEFAULT_MAP_AXIS,
    DEFAULT_RPM_AXIS,
    TRUE_SURFACE,
    volumetric_efficiency,
    ve_table,
)


def test_scalar_input_returns_plausible_ve():
    ve = float(volumetric_efficiency(4200.0, 100.0))
    assert 0.85 < ve < 0.95


def test_ve_stays_within_physical_bounds_over_grid():
    table = ve_table()
    assert np.all(table > 0.4)
    assert np.all(table < 1.0)


def test_ve_peaks_near_stated_resonance_speed():
    sweep = np.linspace(800.0, 7000.0, 2000)
    ve = volumetric_efficiency(sweep, 100.0)
    peak_rpm = sweep[int(np.argmax(ve))]
    assert abs(peak_rpm - TRUE_SURFACE.peak_rpm) < 200.0


def test_ve_increases_with_manifold_pressure():
    map_sweep = np.linspace(20.0, 100.0, 50)
    ve = volumetric_efficiency(3000.0, map_sweep)
    assert np.all(np.diff(ve) > 0.0)


def test_table_shape_and_orientation():
    table = ve_table()
    assert table.shape == (len(DEFAULT_RPM_AXIS), len(DEFAULT_MAP_AXIS))
    assert table[0, -1] > table[0, 0]


def test_base_map_is_wrong_but_plausibly_wrong():
    truth = ve_table(params=TRUE_SURFACE)
    base = ve_table(params=BASE_MAP_SURFACE)
    relative_error = np.abs(base - truth) / truth
    assert relative_error.max() > 0.05
    assert relative_error.max() < 0.25


def test_base_map_error_varies_smoothly():
    truth = ve_table(params=TRUE_SURFACE)
    base = ve_table(params=BASE_MAP_SURFACE)
    error = base / truth
    assert np.abs(np.diff(error, axis=0)).max() < 0.10
    assert np.abs(np.diff(error, axis=1)).max() < 0.05


def test_negative_rpm_raises():
    with pytest.raises(ValueError):
        volumetric_efficiency(-100.0, 50.0)


def test_zero_map_raises():
    with pytest.raises(ValueError):
        volumetric_efficiency(3000.0, 0.0)