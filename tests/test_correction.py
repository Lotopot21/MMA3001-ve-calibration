"""Tests for VE table correction."""

from dataclasses import replace

import numpy as np
import pytest

from vetuner.correction import apply_correction, baseline_correction, sample_correction_factors
from vetuner.drive_cycle import generate_drive_cycle
from vetuner.engine_model import simulate_engine
from vetuner.metrics import evaluate_table
from vetuner.sensors import NOISE_FREE, apply_sensor_model
from vetuner.ve_surface import BASE_MAP_SURFACE, TRUE_SURFACE, ve_table


def test_unit_factors_leave_table_unchanged():
    table = ve_table()
    assert np.allclose(apply_correction(table, np.ones_like(table)), table)


def test_gain_scales_the_applied_correction():
    table = np.full((3, 3), 0.80)
    factors = np.full((3, 3), 1.10)
    half = apply_correction(table, factors, gain=0.5)
    assert np.allclose(half, 0.80 * 1.05)


def test_correction_respects_physical_clamps():
    table = np.full((2, 2), 1.10)
    result = apply_correction(table, np.full((2, 2), 2.0), ve_max=1.15)
    assert np.all(result <= 1.15)


def test_invalid_gain_raises():
    with pytest.raises(ValueError):
        apply_correction(np.ones((2, 2)), np.ones((2, 2)), gain=0.0)


def test_lean_reading_gives_factor_above_one():
    cycle = generate_drive_cycle()
    sim = simulate_engine(cycle, 0.85 * ve_table(params=TRUE_SURFACE))
    log = apply_sensor_model(sim, config=NOISE_FREE)
    factors = sample_correction_factors(log)
    assert np.median(factors[~sim.fuel_cut]) > 1.0


def _drop_samples(log, keep):
    """Return a log containing only the samples where `keep` is True."""
    return replace(
        log,
        time_s=log.time_s[keep],
        rpm=log.rpm[keep],
        map_kpa=log.map_kpa[keep],
        iat_c=log.iat_c[keep],
        tps=log.tps[keep],
        afr=log.afr[keep],
        batt_v=log.batt_v[keep],
    )


def test_correction_recovers_truth_when_fuel_cut_is_excluded():
    """Verification: the correction arithmetic is sound given valid samples."""
    cycle = generate_drive_cycle()
    base = ve_table(params=BASE_MAP_SURFACE)
    truth = ve_table(params=TRUE_SURFACE)

    sim = simulate_engine(cycle, base)
    log = _drop_samples(apply_sensor_model(sim, config=NOISE_FREE), ~sim.fuel_cut)
    result = baseline_correction(log, base)

    before = evaluate_table(base, truth, result.counts)
    after = evaluate_table(result.table, truth, result.counts)
    assert after.rmse_visited < 0.3 * before.rmse_visited


def test_naive_baseline_degrades_the_table():
    """The failure that motivates gating: overrun samples are not fuelling errors.

    A wideband sensor saturates lean during fuel cut, giving a correction
    factor near 1.5 in the low-load cells those samples land in. Including
    them makes the corrected table worse than the starting map.
    """
    cycle = generate_drive_cycle()
    base = ve_table(params=BASE_MAP_SURFACE)
    truth = ve_table(params=TRUE_SURFACE)

    sim = simulate_engine(cycle, base)
    log = apply_sensor_model(sim, config=NOISE_FREE)
    result = baseline_correction(log, base)

    before = evaluate_table(base, truth, result.counts)
    after = evaluate_table(result.table, truth, result.counts)
    assert after.rmse_visited > before.rmse_visited


def test_unvisited_cells_are_left_alone():
    cycle = generate_drive_cycle()
    base = ve_table(params=BASE_MAP_SURFACE)
    sim = simulate_engine(cycle, base)
    log = apply_sensor_model(sim, config=NOISE_FREE)
    result = baseline_correction(log, base)

    untouched = result.counts == 0
    assert untouched.any()
    assert np.allclose(result.table[untouched], base[untouched])


def test_overrun_corrupts_the_baseline():
    """The failure that motivates gating: fuel cut is not a fuelling error."""
    cycle = generate_drive_cycle()
    base = ve_table(params=BASE_MAP_SURFACE)
    sim = simulate_engine(cycle, base)
    log = apply_sensor_model(sim, config=NOISE_FREE)
    result = baseline_correction(log, base)
    assert result.factors.max() > 1.25