"""Tests for the sensor model."""

import numpy as np
import pytest

from vetuner.drive_cycle import generate_drive_cycle
from vetuner.engine_model import simulate_engine
from vetuner.sensors import (
    NOISE_FREE,
    SensorConfig,
    SensorLog,
    add_noise,
    apply_sensor_model,
    quantise,
    transport_delay,
)
from vetuner.ve_surface import BASE_MAP_SURFACE, ve_table


@pytest.fixture
def simulation():
    cycle = generate_drive_cycle()
    return simulate_engine(cycle, ve_table(params=BASE_MAP_SURFACE))


def test_transport_delay_shifts_by_expected_samples():
    ramp = np.arange(100, dtype=float)
    delayed = transport_delay(ramp, dt=0.02, delay_s=0.10)
    assert np.allclose(delayed[5:], ramp[:-5])
    assert np.all(delayed[:5] == ramp[0])


def test_zero_transport_delay_is_identity():
    ramp = np.arange(50, dtype=float)
    assert np.array_equal(transport_delay(ramp, 0.02, 0.0), ramp)


def test_delay_longer_than_signal_returns_first_value():
    ramp = np.arange(10, dtype=float)
    delayed = transport_delay(ramp, 0.02, 5.0)
    assert np.all(delayed == ramp[0])


def test_quantised_values_are_multiples_of_step():
    values = np.linspace(0.0, 10.0, 500)
    step = 0.25
    residual = np.abs(quantise(values, step) / step - np.round(quantise(values, step) / step))
    assert np.all(residual < 1e-9)


def test_quantisation_error_never_exceeds_half_a_step():
    values = np.linspace(-5.0, 5.0, 1000)
    step = 0.1
    assert np.all(np.abs(quantise(values, step) - values) <= step / 2 + 1e-12)


def test_noise_has_requested_standard_deviation():
    rng = np.random.default_rng(0)
    noisy = add_noise(np.zeros(200_000), 0.25, rng)
    assert float(np.std(noisy)) == pytest.approx(0.25, rel=0.02)
    assert abs(float(np.mean(noisy))) < 0.01


def test_same_seed_gives_identical_logs(simulation):
    a = apply_sensor_model(simulation, seed=42)
    b = apply_sensor_model(simulation, seed=42)
    assert np.array_equal(a.afr, b.afr)
    assert np.array_equal(a.map_kpa, b.map_kpa)


def test_different_seeds_give_different_logs(simulation):
    a = apply_sensor_model(simulation, seed=1)
    b = apply_sensor_model(simulation, seed=2)
    assert not np.array_equal(a.afr, b.afr)


def test_noise_free_config_preserves_measured_channels(simulation):
    """The central verification: with no imperfections, the log is exact."""
    log = apply_sensor_model(simulation, config=NOISE_FREE)
    assert np.allclose(log.rpm, simulation.rpm)
    assert np.allclose(log.map_kpa, simulation.map_kpa)
    assert np.allclose(log.iat_c, simulation.iat_c)
    assert np.allclose(log.tps, simulation.tps)
    assert np.allclose(log.batt_v, simulation.batt_v)

    fuelled = ~simulation.fuel_cut
    assert np.allclose(log.afr[fuelled], simulation.afr_actual[fuelled])


def test_log_contains_no_missing_values(simulation):
    log = apply_sensor_model(simulation)
    for channel in (log.rpm, log.map_kpa, log.iat_c, log.tps, log.afr, log.batt_v):
        assert np.all(np.isfinite(channel))


def test_fuel_cut_reads_lean_rail_not_missing(simulation):
    config = SensorConfig(afr_noise_std=0.0, afr_quantisation=0.0, afr_tau_s=0.0)
    log = apply_sensor_model(simulation, config=config)
    assert np.allclose(log.afr[simulation.fuel_cut], config.afr_lean_rail)


def test_log_does_not_expose_true_ve(simulation):
    """Structural check that the answer key cannot leak into the pipeline."""
    log = apply_sensor_model(simulation)
    assert not hasattr(log, "ve_true")
    assert not hasattr(log, "afr_target")
    assert set(vars(log)) == {
        "time_s", "rpm", "map_kpa", "iat_c", "tps", "afr", "batt_v"
    }


def test_measured_channels_stay_physically_plausible(simulation):
    log = apply_sensor_model(simulation)
    assert np.all(log.rpm >= 0.0)
    assert np.all(log.map_kpa >= 0.0)
    assert np.all((log.tps >= 0.0) & (log.tps <= 1.0))
    assert np.all(log.afr > 5.0)
    assert np.all(log.afr <= 22.1)


def test_map_bias_shifts_the_whole_channel(simulation):
    unbiased = apply_sensor_model(
        simulation, config=SensorConfig(map_noise_std_kpa=0.0, map_quantisation_kpa=0.0)
    )
    biased = apply_sensor_model(
        simulation,
        config=SensorConfig(
            map_noise_std_kpa=0.0, map_quantisation_kpa=0.0, map_bias_kpa=3.0
        ),
    )
    assert np.allclose(biased.map_kpa - unbiased.map_kpa, 3.0)


def test_sensor_lag_reduces_high_frequency_content(simulation):
    fast = apply_sensor_model(simulation, config=SensorConfig(afr_tau_s=0.0), seed=7)
    slow = apply_sensor_model(simulation, config=SensorConfig(afr_tau_s=0.5), seed=7)
    assert np.std(np.diff(slow.afr)) < np.std(np.diff(fast.afr))


def test_single_sample_simulation_raises(simulation):
    from dataclasses import replace

    one = replace(
        simulation,
        time_s=simulation.time_s[:1],
        afr_actual=simulation.afr_actual[:1],
        fuel_cut=simulation.fuel_cut[:1],
    )
    with pytest.raises(ValueError):
        apply_sensor_model(one)


def test_returns_sensor_log_type(simulation):
    assert isinstance(apply_sensor_model(simulation), SensorLog)