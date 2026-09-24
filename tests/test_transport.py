"""Tests for exhaust transport delay."""

import numpy as np
import pytest

from vetuner.drive_cycle import generate_drive_cycle
from vetuner.engine_model import simulate_engine
from vetuner.sensors import apply_sensor_model
from vetuner.transport import (
    ExhaustGeometry,
    apply_variable_delay,
    compensate_delay,
    estimate_delay_from_log,
    exhaust_mass_flow,
    exhaust_transport_delay,
)
from vetuner.ve_surface import BASE_MAP_SURFACE, TRUE_SURFACE, ve_table


@pytest.fixture
def log():
    cycle = generate_drive_cycle()
    sim = simulate_engine(cycle, ve_table(params=BASE_MAP_SURFACE))
    return apply_sensor_model(sim, seed=0)


def test_exhaust_gas_is_far_less_dense_than_intake_air():
    """Hot exhaust at ambient pressure has roughly a third of intake density."""
    assert 0.3 < ExhaustGeometry().gas_density_kg_m3 < 0.45


def test_mass_flow_doubles_when_speed_doubles():
    low = exhaust_mass_flow(0.85, 3000.0, 95.0, 25.0, 13.0)
    high = exhaust_mass_flow(0.85, 6000.0, 95.0, 25.0, 13.0)
    assert float(high) == pytest.approx(2.0 * float(low))


def test_mass_flow_exceeds_air_flow_by_the_fuel_fraction():
    stoich = exhaust_mass_flow(0.85, 3000.0, 95.0, 25.0, 14.7)
    rich = exhaust_mass_flow(0.85, 3000.0, 95.0, 25.0, 12.0)
    assert float(rich) > float(stoich)


def test_zero_afr_raises():
    with pytest.raises(ValueError):
        exhaust_mass_flow(0.85, 3000.0, 95.0, 25.0, 0.0)


def test_idle_delay_is_of_order_a_tenth_of_a_second():
    delay = float(exhaust_transport_delay(0.56, 800.0, 35.0, 30.0, 14.7))
    assert 0.05 < delay < 0.25


def test_full_load_delay_is_of_order_milliseconds():
    delay = float(exhaust_transport_delay(0.85, 6000.0, 97.0, 26.0, 12.8))
    assert 0.002 <= delay < 0.020


def test_delay_falls_with_engine_speed():
    speeds = np.linspace(800.0, 7000.0, 30)
    delay = exhaust_transport_delay(0.8, speeds, 90.0, 25.0, 13.0)
    assert np.all(np.diff(delay) < 0.0)


def test_delay_falls_with_load():
    loads = np.linspace(25.0, 100.0, 30)
    delay = exhaust_transport_delay(0.8, 3000.0, loads, 25.0, 13.5)
    assert np.all(np.diff(delay) < 0.0)


def test_delay_spans_more_than_an_order_of_magnitude(log):
    delay = estimate_delay_from_log(log, ve_table(params=TRUE_SURFACE))
    assert delay.max() / delay.min() > 10.0


def test_delay_is_clamped_to_configured_bounds():
    geometry = ExhaustGeometry(min_delay_s=0.01, max_delay_s=0.05)
    delay = exhaust_transport_delay(
        0.8, np.array([100.0, 20000.0]), 90.0, 25.0, 13.0, exhaust=geometry
    )
    assert delay.min() >= 0.01
    assert delay.max() <= 0.05


def test_zero_delay_leaves_a_signal_unchanged():
    time_s = np.linspace(0.0, 5.0, 251)
    signal = np.sin(time_s)
    delayed = apply_variable_delay(signal, time_s, np.zeros_like(time_s))
    assert np.allclose(delayed, signal)


def test_constant_delay_shifts_a_ramp_by_that_amount():
    time_s = np.linspace(0.0, 10.0, 1001)
    ramp = 3.0 * time_s
    delayed = apply_variable_delay(ramp, time_s, np.full_like(time_s, 0.5))
    interior = time_s > 0.5
    assert np.allclose(delayed[interior], 3.0 * (time_s[interior] - 0.5))


def test_sub_sample_delay_is_representable():
    """Whole-sample shifting cannot express a delay shorter than the interval."""
    time_s = np.arange(0.0, 2.0, 0.02)
    ramp = 100.0 * time_s
    delayed = apply_variable_delay(ramp, time_s, np.full_like(time_s, 0.005))
    interior = time_s > 0.1
    assert np.allclose(delayed[interior], 100.0 * (time_s[interior] - 0.005))


def test_negative_delay_raises():
    time_s = np.linspace(0.0, 1.0, 51)
    with pytest.raises(ValueError):
        apply_variable_delay(np.zeros_like(time_s), time_s, -np.ones_like(time_s))


def test_compensation_leaves_the_afr_channel_untouched(log):
    delay = estimate_delay_from_log(log, ve_table(params=TRUE_SURFACE))
    compensated = compensate_delay(log, delay)
    assert np.array_equal(compensated.afr, log.afr)
    assert np.array_equal(compensated.time_s, log.time_s)


def test_compensation_shifts_the_operating_point(log):
    delay = estimate_delay_from_log(log, ve_table(params=TRUE_SURFACE))
    compensated = compensate_delay(log, delay)
    assert not np.allclose(compensated.rpm, log.rpm)


def test_compensation_matters_only_during_transients(log):
    """Where the operating point is steady, the delay changes nothing."""
    delay = estimate_delay_from_log(log, ve_table(params=TRUE_SURFACE))
    compensated = compensate_delay(log, delay)

    rate = np.abs(np.gradient(log.rpm, log.time_s))
    steady = rate < 5.0
    moving = rate > 200.0

    assert np.abs(compensated.rpm - log.rpm)[steady].max() < 5.0
    assert np.abs(compensated.rpm - log.rpm)[moving].max() > 5.0


def test_mismatched_delay_length_raises(log):
    with pytest.raises(ValueError):
        compensate_delay(log, np.zeros(5))


def test_delay_estimate_is_robust_to_table_error(log):
    """The pipeline estimates delay from a table it knows to be wrong."""
    from_truth = estimate_delay_from_log(log, ve_table(params=TRUE_SURFACE))
    from_base = estimate_delay_from_log(log, ve_table(params=BASE_MAP_SURFACE))
    relative = np.abs(from_base - from_truth) / from_truth
    assert np.median(relative) < 0.20