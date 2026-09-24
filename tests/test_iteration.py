"""Tests for iterative calibration and its convergence behaviour."""

import numpy as np
import pytest

from vetuner.drive_cycle import generate_drive_cycle
from vetuner.iteration import iterate_calibration, theoretical_error_decay
from vetuner.metrics import evaluate_table
from vetuner.sensors import NOISE_FREE
from vetuner.ve_surface import BASE_MAP_SURFACE, TRUE_SURFACE, ve_table


@pytest.fixture(scope="module")
def cycle():
    return generate_drive_cycle()


@pytest.fixture(scope="module")
def tables():
    return ve_table(params=BASE_MAP_SURFACE), ve_table(params=TRUE_SURFACE)


def test_decay_starts_at_full_error():
    assert theoretical_error_decay(0.5, 4)[0] == pytest.approx(1.0)


def test_unit_gain_removes_all_error_in_one_pass():
    decay = theoretical_error_decay(1.0, 3)
    assert decay[1] == pytest.approx(0.0)


def test_half_gain_halves_error_each_pass():
    assert np.allclose(theoretical_error_decay(0.5, 3), [1.0, 0.5, 0.25, 0.125])


def test_negative_pass_count_raises():
    with pytest.raises(ValueError):
        theoretical_error_decay(0.5, -1)


def test_invalid_gain_raises(cycle, tables):
    base, _ = tables
    with pytest.raises(ValueError):
        iterate_calibration(cycle, base, gain=1.5, max_passes=1)


def test_zero_passes_raises(cycle, tables):
    base, _ = tables
    with pytest.raises(ValueError):
        iterate_calibration(cycle, base, max_passes=0)


def test_history_lengths_are_consistent(cycle, tables):
    base, _ = tables
    history = iterate_calibration(
        cycle, base, gain=0.7, max_passes=3, method="bilinear",
        sensor_config=NOISE_FREE,
    )
    assert len(history.tables) == history.n_passes + 1
    assert len(history.max_change) == history.n_passes
    assert len(history.coverage) == history.n_passes


def test_initial_table_is_preserved_in_history(cycle, tables):
    base, _ = tables
    history = iterate_calibration(
        cycle, base, max_passes=2, method="bilinear", sensor_config=NOISE_FREE
    )
    assert np.array_equal(history.tables[0], base)


def test_iteration_does_not_mutate_the_input_table(cycle, tables):
    base, _ = tables
    original = base.copy()
    iterate_calibration(cycle, base, max_passes=2, method="bilinear", sensor_config=NOISE_FREE)
    assert np.array_equal(base, original)


def test_changes_shrink_as_the_iteration_proceeds(cycle, tables):
    base, _ = tables
    history = iterate_calibration(
        cycle, base, gain=0.7, max_passes=5, method="bilinear",
        sensor_config=NOISE_FREE, tolerance=1e-9,
    )
    assert history.max_change[-1] < history.max_change[0]


def test_noise_free_iteration_approaches_a_fixed_point(cycle, tables):
    """Without measurement noise the per-pass change decays substantially.

    The iteration does not reach an arbitrarily tight tolerance, because the
    surface fit couples neighbouring cells: each pass redistributes a small
    residual between cells with data and cells without, leaving a floor below
    which the table continues to shift slightly.
    """
    base, _ = tables
    history = iterate_calibration(
        cycle, base, gain=0.8, max_passes=15, method="bilinear",
        sensor_config=NOISE_FREE, tolerance=1e-12,
    )
    assert history.max_change[-1] < 0.05 * history.max_change[0]


def test_lower_gain_converges_more_slowly(cycle, tables):
    """Damping trades convergence speed for noise rejection."""
    base, _ = tables
    fast = iterate_calibration(
        cycle, base, gain=0.9, max_passes=6, method="bilinear",
        sensor_config=NOISE_FREE, tolerance=1e-12,
    )
    slow = iterate_calibration(
        cycle, base, gain=0.3, max_passes=6, method="bilinear",
        sensor_config=NOISE_FREE, tolerance=1e-12,
    )
    assert slow.max_change[-1] > fast.max_change[-1]


def test_iteration_improves_on_a_single_pass(cycle, tables):
    """The justification for iterating at all."""
    base, truth = tables
    history = iterate_calibration(
        cycle, base, gain=0.7, max_passes=6, method="bilinear",
        sensor_config=NOISE_FREE, tolerance=1e-9,
    )
    one_pass = evaluate_table(history.tables[1], truth).rmse
    converged = evaluate_table(history.final_table, truth).rmse
    assert converged < one_pass





def test_noise_free_decay_follows_the_theoretical_rate(cycle, tables):
    """Verification against the analytic fixed-point prediction.

    With noise disabled and a method that does not smooth across cells, the
    per-pass change should fall geometrically at rate (1 - gain), matching the
    linear fixed-point theory.
    """
    base, _ = tables
    gain = 0.5
    history = iterate_calibration(
        cycle, base, gain=gain, max_passes=6, method="bilinear",
        sensor_config=NOISE_FREE, tolerance=1e-12,
    )
    ratios = np.array(history.rms_change[1:4]) / np.array(history.rms_change[0:3])
    assert np.allclose(ratios, 1.0 - gain, atol=0.15)


def test_iteration_stops_at_the_pass_limit(cycle, tables):
    base, _ = tables
    history = iterate_calibration(
        cycle, base, gain=0.1, max_passes=2, method="bilinear", tolerance=1e-12
    )
    assert history.n_passes == 2
    assert not history.converged