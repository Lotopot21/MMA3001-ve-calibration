"""Tests for log sample gating."""

from dataclasses import replace

import numpy as np
import pytest

from vetuner.correction import baseline_correction, gated_correction
from vetuner.drive_cycle import generate_drive_cycle
from vetuner.engine_model import simulate_engine
from vetuner.gating import (
    GatingConfig,
    extend_forward,
    gate_samples,
    implausible_afr_mask,
    out_of_domain_mask,
    overrun_mask,
    signal_rate,
    transient_mask,
)
from vetuner.metrics import evaluate_table
from vetuner.sensors import NOISE_FREE, apply_sensor_model
from vetuner.ve_surface import (
    BASE_MAP_SURFACE,
    DEFAULT_MAP_AXIS,
    DEFAULT_RPM_AXIS,
    TRUE_SURFACE,
    ve_table,
)


@pytest.fixture
def log():
    cycle = generate_drive_cycle()
    sim = simulate_engine(cycle, ve_table(params=BASE_MAP_SURFACE))
    return apply_sensor_model(sim, seed=0)


def test_rate_of_a_linear_ramp_is_its_slope():
    time_s = np.linspace(0.0, 10.0, 501)
    assert np.allclose(signal_rate(3.5 * time_s + 2.0, time_s), 3.5)


def test_rate_of_a_constant_is_zero():
    time_s = np.linspace(0.0, 5.0, 251)
    assert np.allclose(signal_rate(np.full_like(time_s, 7.0), time_s), 0.0)


def test_rate_requires_two_samples():
    with pytest.raises(ValueError):
        signal_rate(np.array([1.0]), np.array([0.0]))


def test_extend_forward_flags_following_samples():
    mask = np.zeros(10, dtype=bool)
    mask[3] = True
    extended = extend_forward(mask, 2)
    assert list(np.flatnonzero(extended)) == [3, 4, 5]


def test_extend_forward_does_not_reach_backwards():
    mask = np.zeros(10, dtype=bool)
    mask[5] = True
    assert not extend_forward(mask, 3)[:5].any()


def test_extend_forward_by_zero_is_identity():
    mask = np.array([False, True, False, True])
    assert np.array_equal(extend_forward(mask, 0), mask)


def test_lean_rail_reading_is_implausible(log):
    config = GatingConfig()
    flagged = implausible_afr_mask(log, config)
    assert flagged.any()
    assert np.all(log.afr[flagged] > config.afr_max)


def test_overrun_detected_at_closed_throttle_and_speed(log):
    config = GatingConfig()
    flagged = overrun_mask(log, config)
    assert flagged.any()
    assert np.all(log.tps[flagged] < config.overrun_tps)
    assert np.all(log.rpm[flagged] > config.overrun_rpm)


def test_idle_is_not_treated_as_overrun(log):
    flagged = overrun_mask(log, GatingConfig())
    idle = log.rpm < 1000.0
    assert not flagged[idle].any()


def test_below_table_load_axis_is_out_of_domain(log):
    flagged = out_of_domain_mask(log)
    assert flagged.any()
    assert np.all(
        (log.map_kpa[flagged] < DEFAULT_MAP_AXIS[0])
        | (log.map_kpa[flagged] > DEFAULT_MAP_AXIS[-1])
        | (log.rpm[flagged] < DEFAULT_RPM_AXIS[0])
        | (log.rpm[flagged] > DEFAULT_RPM_AXIS[-1])
    )


def test_steady_cruise_is_not_flagged_as_transient(log):
    flagged = transient_mask(log, GatingConfig())
    steady = (log.time_s > 40.0) & (log.time_s < 55.0)
    assert not flagged[steady].any()


def test_throttle_step_is_flagged_as_transient(log):
    flagged = transient_mask(log, GatingConfig())
    rate = np.abs(signal_rate(log.tps, log.time_s))
    assert flagged[rate > 1.0].all()


def test_sustained_pull_is_mostly_accepted(log):
    """A ramp at constant load is valid data, not a transient."""
    flagged = transient_mask(log, GatingConfig())
    pull = (log.map_kpa > 90.0) & (log.rpm > 3500.0)
    assert pull.sum() > 100
    assert flagged[pull].mean() < 0.35


def test_gating_accepts_a_useful_fraction(log):
    gate = gate_samples(log)
    assert 0.4 < gate.acceptance_rate < 0.95


def test_rejection_summary_reports_every_criterion(log):
    summary = gate_samples(log).rejection_summary()
    assert set(summary) == {
        "implausible_afr", "overrun", "out_of_domain", "transient", "rejected_total"
    }
    assert summary["rejected_total"] > 0


def test_short_log_raises(log):
    one = replace(
        log,
        time_s=log.time_s[:1],
        rpm=log.rpm[:1],
        map_kpa=log.map_kpa[:1],
        iat_c=log.iat_c[:1],
        tps=log.tps[:1],
        afr=log.afr[:1],
        batt_v=log.batt_v[:1],
    )
    with pytest.raises(ValueError):
        gate_samples(one)


def test_gating_turns_a_damaging_pass_into_an_improving_one():
    """The headline result: gating is what makes the correction work.

    Each pass is scored over the cells it actually reached. The naive pass
    corrects many more cells but corrupts them with overrun and transient
    data; the gated pass corrects fewer cells and gets them nearly right.
    """
    cycle = generate_drive_cycle()
    base = ve_table(params=BASE_MAP_SURFACE)
    truth = ve_table(params=TRUE_SURFACE)

    sim = simulate_engine(cycle, base)
    log = apply_sensor_model(sim, config=NOISE_FREE)

    naive = baseline_correction(log, base)
    gated, _ = gated_correction(log, base)

    naive_before = evaluate_table(base, truth, naive.counts)
    naive_after = evaluate_table(naive.table, truth, naive.counts)
    gated_before = evaluate_table(base, truth, gated.counts)
    gated_after = evaluate_table(gated.table, truth, gated.counts)

    assert naive_after.rmse_visited > naive_before.rmse_visited
    assert gated_after.rmse_visited < 0.2 * gated_before.rmse_visited


def test_gating_reduces_coverage():
    """The trade-off: discarding samples leaves fewer cells with data."""
    cycle = generate_drive_cycle()
    base = ve_table(params=BASE_MAP_SURFACE)
    sim = simulate_engine(cycle, base)
    log = apply_sensor_model(sim, seed=0)

    naive = baseline_correction(log, base)
    gated, _ = gated_correction(log, base)

    assert (gated.counts > 0).sum() < (naive.counts > 0).sum()