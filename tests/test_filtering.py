"""Tests for sensor channel filtering."""

import numpy as np
import pytest

from vetuner.filtering import (
    FILTER_METHODS,
    filter_log,
    kalman_filter,
    kalman_steady_state,
    measure_phase_lag,
    moving_average,
    savitzky_golay,
)


@pytest.fixture
def noisy_ramp():
    rng = np.random.default_rng(0)
    clean = np.linspace(12.0, 15.0, 2000)
    return clean, clean + rng.normal(0.0, 0.2, clean.size)


def test_moving_average_preserves_a_constant():
    assert np.allclose(moving_average(np.full(500, 14.7), 21), 14.7)


def test_moving_average_preserves_a_linear_ramp_in_the_interior():
    """A centred average of a linear signal returns the signal exactly."""
    ramp = np.linspace(0.0, 100.0, 1001)
    smoothed = moving_average(ramp, 21)
    interior = slice(20, -20)
    assert np.allclose(smoothed[interior], ramp[interior])


def test_moving_average_reduces_noise_by_the_square_root_of_the_window():
    rng = np.random.default_rng(1)
    noise = rng.normal(0.0, 1.0, 200_000)
    window = 25
    reduction = noise.std() / moving_average(noise, window).std()
    assert reduction == pytest.approx(np.sqrt(window), rel=0.05)


def test_moving_average_rejects_an_oversized_window():
    with pytest.raises(ValueError):
        moving_average(np.zeros(10), 50)


def test_savitzky_golay_preserves_a_polynomial_exactly():
    """Verification: a polynomial of degree at most polyorder passes unchanged."""
    x = np.linspace(-1.0, 1.0, 501)
    cubic = 2.0 + 3.0 * x - 1.5 * x**2 + 0.7 * x**3
    smoothed = savitzky_golay(cubic, window=31, polyorder=3)
    interior = slice(30, -30)
    assert np.allclose(smoothed[interior], cubic[interior], atol=1e-10)


def test_moving_average_does_not_preserve_curvature():
    """The contrast that justifies Savitzky-Golay over a plain average."""
    x = np.linspace(-1.0, 1.0, 501)
    quadratic = 1.0 - x**2
    interior = slice(30, -30)

    average_error = np.abs(moving_average(quadratic, 31) - quadratic)[interior].max()
    golay_error = np.abs(savitzky_golay(quadratic, 31, 2) - quadratic)[interior].max()

    assert golay_error < 1e-10
    assert average_error > 100.0 * golay_error


def test_savitzky_golay_requires_an_odd_window():
    with pytest.raises(ValueError):
        savitzky_golay(np.zeros(100), window=20, polyorder=2)


def test_savitzky_golay_requires_window_above_polyorder():
    with pytest.raises(ValueError):
        savitzky_golay(np.zeros(100), window=3, polyorder=5)


def test_kalman_gain_matches_the_analytic_steady_state():
    """Verification against the closed-form solution of the Riccati equation."""
    q, r = 1e-4, 4e-2
    state = kalman_steady_state(q, r)

    variance = r
    for _ in range(5000):
        variance += q
        gain = variance / (variance + r)
        variance *= 1.0 - gain

    assert gain == pytest.approx(state.gain, rel=1e-9)


def test_kalman_preserves_a_constant():
    assert np.allclose(kalman_filter(np.full(500, 14.7), 1e-4, 1e-2), 14.7)


def test_kalman_reduces_noise():
    rng = np.random.default_rng(2)
    noise = rng.normal(0.0, 1.0, 20_000)
    assert kalman_filter(noise, 1e-3, 1.0).std() < 0.5 * noise.std()


def test_smaller_process_variance_smooths_more():
    rng = np.random.default_rng(3)
    noise = rng.normal(0.0, 1.0, 20_000)
    trusting = kalman_filter(noise, 1e-1, 1.0)
    sceptical = kalman_filter(noise, 1e-4, 1.0)
    assert sceptical.std() < trusting.std()


def test_kalman_rejects_zero_measurement_variance():
    with pytest.raises(ValueError):
        kalman_filter(np.zeros(100), 1e-3, 0.0)


def test_steady_state_gain_lies_between_zero_and_one():
    for q in (1e-6, 1e-4, 1e-2, 1.0):
        assert 0.0 < kalman_steady_state(q, 0.04).gain < 1.0


def test_centred_filters_introduce_no_lag(noisy_ramp):
    """The property that makes them usable offline but not in real time."""
    clean, noisy = noisy_ramp
    wave = clean + 0.5 * np.sin(np.linspace(0.0, 40.0, clean.size))

    assert measure_phase_lag(wave, moving_average(wave, 21)) == 0
    assert measure_phase_lag(wave, savitzky_golay(wave, 21, 2)) == 0


def test_kalman_lags_more_than_a_centred_filter(noisy_ramp):
    """A causal filter cannot use future samples, so it trails the signal.

    The lag is measured against a step rather than a slow wave: at the gains
    used here it is a few samples, below the resolution of whole-sample
    cross-correlation on a smooth signal.
    """
    step = np.concatenate([np.zeros(200), np.ones(400)])

    causal = kalman_filter(step, 1e-3, 1e-2)
    centred = moving_average(step, 11)

    assert causal[205] < 0.9
    assert centred[205] > 0.9


def test_phase_lag_rejects_mismatched_lengths():
    with pytest.raises(ValueError):
        measure_phase_lag(np.zeros(10), np.zeros(20))


def test_every_filter_preserves_the_log_length():
    from vetuner.drive_cycle import generate_drive_cycle
    from vetuner.engine_model import simulate_engine
    from vetuner.sensors import apply_sensor_model
    from vetuner.ve_surface import BASE_MAP_SURFACE, ve_table

    sim = simulate_engine(generate_drive_cycle(), ve_table(params=BASE_MAP_SURFACE))
    log = apply_sensor_model(sim, seed=0)

    settings = {
        "none": {},
        "moving_average": {"window": 11},
        "savitzky_golay": {"window": 11, "polyorder": 2},
        "kalman": {"process_variance": 1e-3, "measurement_variance": 1e-2},
    }

    for method in FILTER_METHODS:
        filtered = filter_log(log, method, **settings[method])
        assert len(filtered) == len(log)
        assert np.isfinite(filtered.afr).all()


def test_unknown_filter_raises():
    from vetuner.drive_cycle import generate_drive_cycle
    from vetuner.engine_model import simulate_engine
    from vetuner.sensors import apply_sensor_model
    from vetuner.ve_surface import BASE_MAP_SURFACE, ve_table

    sim = simulate_engine(generate_drive_cycle(), ve_table(params=BASE_MAP_SURFACE))
    log = apply_sensor_model(sim, seed=0)

    with pytest.raises(ValueError):
        filter_log(log, "wavelet")
        

