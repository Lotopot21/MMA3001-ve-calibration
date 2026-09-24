"""Tests for injector pulse width output."""

import numpy as np
import pytest

from vetuner.engine_model import EngineGeometry, Injector
from vetuner.pulse_width import (
    export_table_csv,
    pulse_width_error,
    pulse_width_map,
    required_injector_flow,
)
from vetuner.ve_surface import (
    BASE_MAP_SURFACE,
    DEFAULT_MAP_AXIS,
    DEFAULT_RPM_AXIS,
    TRUE_SURFACE,
    ve_table,
)


@pytest.fixture
def truth():
    return ve_table(params=TRUE_SURFACE)


def test_map_matches_the_table_shape(truth):
    result = pulse_width_map(truth)
    assert result.pulse_width_ms.shape == truth.shape
    assert result.duty.shape == truth.shape


def test_pulse_width_always_exceeds_opening_delay(truth):
    injector = Injector()
    result = pulse_width_map(truth, injector=injector)
    assert np.all(result.pulse_width_ms > injector.deadtime_ms_at_14v)


def test_pulse_width_rises_with_load(truth):
    result = pulse_width_map(truth)
    assert np.all(np.diff(result.pulse_width_ms, axis=1) > 0.0)


def test_duty_rises_with_engine_speed_at_full_load(truth):
    """Available time per cycle shrinks with speed, so duty climbs."""
    result = pulse_width_map(truth)
    assert result.duty[-1, -1] > result.duty[0, -1]


def test_duty_is_pulse_width_over_cycle_period(truth):
    engine = EngineGeometry()
    result = pulse_width_map(truth, engine=engine)
    period_at_redline = 60.0 * 1e3 * engine.revolutions_per_cycle / DEFAULT_RPM_AXIS[-1]
    assert result.duty[-1, -1] == pytest.approx(
        result.pulse_width_ms[-1, -1] / period_at_redline
    )


def test_default_injector_is_adequately_sized(truth):
    result = pulse_width_map(truth)
    assert result.feasible.all()
    assert result.max_duty < 0.85


def test_worst_cell_is_at_high_speed_and_load(truth):
    rpm, map_kpa = pulse_width_map(truth).worst_cell
    assert rpm > 0.7 * DEFAULT_RPM_AXIS[-1]
    assert map_kpa > 0.7 * DEFAULT_MAP_AXIS[-1]


def test_hotter_air_reduces_pulse_width(truth):
    cold = pulse_width_map(truth, iat_c=10.0)
    hot = pulse_width_map(truth, iat_c=50.0)
    assert np.all(hot.pulse_width_ms < cold.pulse_width_ms)


def test_lower_voltage_lengthens_pulse_width(truth):
    low = pulse_width_map(truth, batt_v=12.0)
    high = pulse_width_map(truth, batt_v=14.0)
    assert np.all(low.pulse_width_ms > high.pulse_width_ms)


def test_larger_engine_needs_more_fuel(truth):
    small = pulse_width_map(truth, engine=EngineGeometry(displacement_l=1.0))
    large = pulse_width_map(truth, engine=EngineGeometry(displacement_l=2.0))
    assert np.all(large.fuel_mass_kg > small.fuel_mass_kg)


def test_required_flow_is_below_the_fitted_injector(truth):
    """Confirms the sizing margin rather than just that it fits."""
    required = required_injector_flow(truth, target_duty=0.80)
    assert 0.0 < required < Injector().flow_cc_min


def test_required_flow_rises_with_displacement(truth):
    small = required_injector_flow(truth, engine=EngineGeometry(displacement_l=1.6))
    large = required_injector_flow(truth, engine=EngineGeometry(displacement_l=2.4))
    assert large > small


def test_invalid_target_duty_raises(truth):
    with pytest.raises(ValueError):
        required_injector_flow(truth, target_duty=0.0)


def test_identical_tables_give_zero_pulse_width_error(truth):
    errors = pulse_width_error(truth, truth)
    assert errors["max_abs_ms"] == pytest.approx(0.0)
    assert errors["fraction_within_1pct"] == pytest.approx(1.0)


def test_miscalibrated_table_gives_substantial_pulse_width_error(truth):
    errors = pulse_width_error(ve_table(params=BASE_MAP_SURFACE), truth)
    assert errors["mean_abs_pct"] > 3.0


def test_mismatched_table_shape_raises():
    with pytest.raises(ValueError):
        pulse_width_map(np.ones((3, 3)))


def test_non_positive_ve_raises(truth):
    bad = truth.copy()
    bad[0, 0] = 0.0
    with pytest.raises(ValueError):
        pulse_width_map(bad)


def test_csv_export_round_trips(tmp_path, truth):
    destination = export_table_csv(tmp_path / "ve.csv", truth)
    rows = destination.read_text(encoding="utf-8").strip().splitlines()

    assert len(rows) == len(DEFAULT_RPM_AXIS) + 1
    assert rows[0].startswith("VE %")

    recovered = np.array(
        [[float(v) for v in row.split(",")[1:]] for row in rows[1:]]
    )
    assert np.allclose(recovered / 100.0, truth, atol=1e-4)


def test_csv_export_rejects_wrong_shape(tmp_path):
    with pytest.raises(ValueError):
        export_table_csv(tmp_path / "bad.csv", np.ones((2, 2)))