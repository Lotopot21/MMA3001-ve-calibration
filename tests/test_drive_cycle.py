"""Tests for synthetic drive cycle generation."""

import numpy as np
import pytest

from vetuner.drive_cycle import (
    DEFAULT_CYCLE,
    MAP_FLOOR_KPA,
    P_ATM_KPA,
    Segment,
    first_order_lag,
    generate_drive_cycle,
    manifold_pressure,
)
from vetuner.ve_surface import DEFAULT_MAP_AXIS, DEFAULT_RPM_AXIS


def test_sample_count_matches_requested_duration():
    cycle = generate_drive_cycle(sample_rate_hz=50.0)
    expected = int(round(sum(seg.duration_s for seg in DEFAULT_CYCLE) * 50.0))
    assert abs(len(cycle) - expected) <= len(DEFAULT_CYCLE)


def test_time_axis_is_uniform_and_starts_at_zero():
    cycle = generate_drive_cycle(sample_rate_hz=50.0)
    assert cycle.time_s[0] == 0.0
    assert np.allclose(np.diff(cycle.time_s), 0.02)


def test_all_channels_have_equal_length():
    cycle = generate_drive_cycle()
    n = len(cycle.time_s)
    assert len(cycle.rpm) == n
    assert len(cycle.tps) == n
    assert len(cycle.map_kpa) == n
    assert len(cycle.iat_c) == n
    assert len(cycle.batt_v) == n


def test_channels_stay_within_physical_ranges():
    cycle = generate_drive_cycle()
    assert np.all(cycle.rpm >= 0.0)
    assert np.all(cycle.rpm <= 7100.0)
    assert np.all((cycle.tps >= 0.0) & (cycle.tps <= 1.0))
    assert np.all(cycle.map_kpa >= MAP_FLOOR_KPA)
    assert np.all(cycle.map_kpa <= P_ATM_KPA)
    assert np.all((cycle.batt_v > 13.0) & (cycle.batt_v < 14.5))


def test_map_increases_with_throttle_at_fixed_speed():
    tps_sweep = np.linspace(0.0, 1.0, 50)
    pressure = manifold_pressure(3000.0, tps_sweep)
    assert np.all(np.diff(pressure) > 0.0)


def test_map_decreases_with_speed_at_fixed_throttle():
    rpm_sweep = np.linspace(800.0, 7000.0, 50)
    pressure = manifold_pressure(rpm_sweep, 0.25)
    assert np.all(np.diff(pressure) < 0.0)


def test_wide_open_throttle_approaches_ambient():
    assert manifold_pressure(6000.0, 1.0) > 0.93 * P_ATM_KPA


def test_idle_produces_realistic_vacuum():
    idle_map = float(manifold_pressure(800.0, 0.0))
    assert 25.0 < idle_map < 45.0


def test_lag_smooths_a_step_without_overshoot():
    step = np.concatenate([np.zeros(50), np.ones(200)])
    lagged = first_order_lag(step, dt=0.02, tau_s=0.2)
    assert lagged.max() <= 1.0
    assert lagged[50] < 0.5
    assert lagged[-1] > 0.95


def test_lag_reaches_63_percent_after_one_time_constant():
    dt, tau = 0.001, 0.5
    step = np.ones(int(2.0 / dt))
    step[0] = 0.0
    lagged = first_order_lag(step, dt, tau)
    at_tau = lagged[int(tau / dt)]
    assert 0.60 < at_tau < 0.66


def test_cycle_covers_some_but_not_all_table_cells():
    cycle = generate_drive_cycle()
    rpm_bin = np.digitize(cycle.rpm, DEFAULT_RPM_AXIS)
    map_bin = np.digitize(cycle.map_kpa, DEFAULT_MAP_AXIS)
    visited = {(int(r), int(m)) for r, m in zip(rpm_bin, map_bin, strict=True)}
    total_cells = (len(DEFAULT_RPM_AXIS) + 1) * (len(DEFAULT_MAP_AXIS) + 1)
    coverage = len(visited) / total_cells
    assert 0.10 < coverage < 0.80


def test_empty_cycle_raises():
    with pytest.raises(ValueError):
        generate_drive_cycle(cycle=())


def test_zero_duration_segment_raises():
    bad = (Segment(0.0, 1000.0, 1000.0, 0.1, 0.1),)
    with pytest.raises(ValueError):
        generate_drive_cycle(cycle=bad)


def test_throttle_outside_range_raises():
    with pytest.raises(ValueError):
        manifold_pressure(3000.0, 1.5)