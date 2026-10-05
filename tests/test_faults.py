"""Tests for deliberate sensor fault injection."""

import numpy as np
import pytest

from vetuner.drive_cycle import generate_drive_cycle
from vetuner.engine_model import simulate_engine
from vetuner.faults import (
    FAULT_CASES,
    FAULTY_CHANNELS,
    drop_samples,
    fault_case,
    freeze,
    invalidate,
    offset,
    saturate,
    sluggish,
)
from vetuner.sensor_health import OK, longest_unchanged_run, screen_log
from vetuner.sensors import SensorLog, apply_sensor_model
from vetuner.ve_surface import BASE_MAP_SURFACE, ve_table


@pytest.fixture
def log():
    sim = simulate_engine(generate_drive_cycle(), ve_table(params=BASE_MAP_SURFACE))
    return apply_sensor_model(sim, seed=0)


def channels(log):
    """Return every channel of a log as a name-to-array mapping."""
    return {name: getattr(log, name) for name in FAULTY_CHANNELS}


# --- injectors leave the rest of the log alone ------------------------------


@pytest.mark.parametrize("channel", FAULTY_CHANNELS)
def test_freezing_one_channel_changes_only_that_channel(log, channel):
    faulty = freeze(log, channel, 60.0, 5.0)
    for name, values in channels(log).items():
        same = np.array_equal(getattr(faulty, name), values)
        assert same is (name != channel)


def test_injecting_a_fault_does_not_modify_the_original(log):
    before = log.map_kpa.copy()
    freeze(log, "map_kpa", 60.0, 5.0)
    offset(log, "map_kpa", 5.0)
    saturate(log, "map_kpa", 0.0, 60.0, 5.0)
    invalidate(log, "map_kpa", 10)
    assert np.array_equal(log.map_kpa, before)


@pytest.mark.parametrize(
    "inject",
    [
        lambda log: freeze(log, "wheel_speed", 60.0, 5.0),
        lambda log: offset(log, "wheel_speed", 1.0),
        lambda log: saturate(log, "wheel_speed", 1.0, 60.0, 5.0),
        lambda log: invalidate(log, "wheel_speed", 5),
        lambda log: sluggish(log, "wheel_speed", 0.2),
    ],
)
def test_an_unknown_channel_is_rejected(log, inject):
    with pytest.raises(ValueError, match="Unknown channel"):
        inject(log)


# --- freeze ----------------------------------------------------------------


def test_a_frozen_channel_holds_one_value_for_the_whole_window(log):
    faulty = freeze(log, "map_kpa", 100.0, 20.0)
    window = (faulty.time_s >= 100.0) & (faulty.time_s < 120.0)
    assert len(np.unique(faulty.map_kpa[window])) == 1


def test_a_frozen_channel_holds_the_value_it_had_when_the_fault_began(log):
    faulty = freeze(log, "map_kpa", 100.0, 20.0)
    first = np.flatnonzero(log.time_s >= 100.0)[0]
    assert faulty.map_kpa[first] == log.map_kpa[first]


def test_freezing_produces_a_run_of_the_requested_length(log):
    faulty = freeze(log, "iat_c", 50.0, 8.0)
    length, _ = longest_unchanged_run(faulty.iat_c)
    dt = float(np.median(np.diff(log.time_s)))
    assert length * dt == pytest.approx(8.0, abs=dt)


def test_a_window_outside_the_log_is_rejected(log):
    with pytest.raises(ValueError, match="selects no samples"):
        freeze(log, "map_kpa", 10_000.0, 5.0)


def test_a_non_positive_duration_is_rejected(log):
    with pytest.raises(ValueError, match="duration must be positive"):
        freeze(log, "map_kpa", 10.0, 0.0)


# --- offset ----------------------------------------------------------------


def test_an_offset_shifts_every_sample_equally(log):
    faulty = offset(log, "map_kpa", 3.5)
    assert np.allclose(faulty.map_kpa - log.map_kpa, 3.5)


def test_an_offset_leaves_the_channel_varying_normally(log):
    """A biased sensor still tracks the engine; only its zero is wrong.

    This is what makes the fault hard: no local test on the readings can see
    it, which is why detection needs an absolute reference.
    """
    faulty = offset(log, "map_kpa", 3.5)
    assert np.allclose(np.diff(faulty.map_kpa), np.diff(log.map_kpa))


# --- saturate --------------------------------------------------------------


def test_saturation_pins_the_channel_at_the_requested_value(log):
    faulty = saturate(log, "afr", 21.0, 20.0, 10.0)
    window = (faulty.time_s >= 20.0) & (faulty.time_s < 30.0)
    assert np.all(faulty.afr[window] == 21.0)
    assert np.array_equal(faulty.afr[~window], log.afr[~window])


# --- drop_samples ----------------------------------------------------------


def test_dropping_samples_shortens_every_channel_equally(log):
    faulty = drop_samples(log, 90.0, 4.0)
    assert len(faulty) < len(log)
    assert all(len(values) == len(faulty) for values in channels(faulty).values())


def test_dropping_samples_leaves_a_gap_in_the_time_base(log):
    faulty = drop_samples(log, 90.0, 4.0)
    assert float(np.diff(faulty.time_s).max()) == pytest.approx(4.0, abs=0.05)


def test_dropping_samples_removes_the_requested_window(log):
    faulty = drop_samples(log, 90.0, 4.0)
    assert not ((faulty.time_s >= 90.0) & (faulty.time_s < 94.0)).any()


# --- invalidate ------------------------------------------------------------


def test_invalidating_marks_exactly_the_requested_count(log):
    faulty = invalidate(log, "afr", 25)
    assert int(np.sum(~np.isfinite(faulty.afr))) == 25


def test_invalidating_is_reproducible_for_a_given_seed(log):
    first = invalidate(log, "afr", 25, seed=7).afr
    second = invalidate(log, "afr", 25, seed=7).afr
    assert np.array_equal(np.isnan(first), np.isnan(second))


def test_a_different_seed_chooses_different_samples(log):
    first = invalidate(log, "afr", 25, seed=1).afr
    second = invalidate(log, "afr", 25, seed=2).afr
    assert not np.array_equal(np.isnan(first), np.isnan(second))


@pytest.mark.parametrize("count", [0, -1, 10**9])
def test_an_impossible_invalid_count_is_rejected(log, count):
    with pytest.raises(ValueError, match="Cannot invalidate"):
        invalidate(log, "afr", count)


# --- sluggish --------------------------------------------------------------


def test_slowing_a_channel_reduces_its_rate_of_change(log):
    faulty = sluggish(log, "afr", 0.36)
    assert np.abs(np.diff(faulty.afr)).mean() < np.abs(np.diff(log.afr)).mean()


def test_slowing_a_channel_keeps_it_within_its_original_range(log):
    """A sluggish sensor reads plausibly, which is why screening cannot see it."""
    faulty = sluggish(log, "afr", 0.36)
    assert faulty.afr.min() >= log.afr.min() - 1e-9
    assert faulty.afr.max() <= log.afr.max() + 1e-9


def test_a_zero_time_constant_leaves_the_channel_alone(log):
    assert np.allclose(sluggish(log, "afr", 0.0).afr, log.afr)


def test_slowing_needs_at_least_two_samples():
    one = np.array([1.0])
    single = SensorLog(
        time_s=one, rpm=one, map_kpa=one, iat_c=one, tps=one, afr=one, batt_v=one
    )
    with pytest.raises(ValueError, match="at least two samples"):
        sluggish(single, "afr", 0.1)


# --- the catalogue ---------------------------------------------------------


def test_every_case_has_a_distinct_name():
    names = [case.name for case in FAULT_CASES]
    assert len(names) == len(set(names))


def test_the_catalogue_includes_a_healthy_control():
    assert any(case.name == "healthy" for case in FAULT_CASES)


def test_the_catalogue_includes_a_known_blind_spot():
    """A catalogue of only detectable faults would overstate the screening."""
    undetected = [case for case in FAULT_CASES if not case.detected]
    assert any(case.name != "healthy" for case in undetected)


def test_a_case_can_be_looked_up_by_name():
    assert fault_case("logger_gap").expected_channel == "log"


def test_an_unknown_case_name_is_rejected():
    with pytest.raises(KeyError, match="No fault case"):
        fault_case("exploded")


@pytest.mark.parametrize("case", FAULT_CASES, ids=lambda c: c.name)
def test_screening_reaches_the_severity_the_catalogue_declares(log, case):
    assert screen_log(case.apply(log)).worst == case.expected_severity


@pytest.mark.parametrize("case", FAULT_CASES, ids=lambda c: c.name)
def test_screening_names_the_channel_the_catalogue_declares(log, case):
    flagged = {f.channel for f in screen_log(case.apply(log)).problems()}
    if case.expected_channel is None:
        assert flagged == set()
    else:
        assert case.expected_channel in flagged


@pytest.mark.parametrize(
    "case", [c for c in FAULT_CASES if c.detected], ids=lambda c: c.name
)
def test_every_detected_fault_carries_a_quantified_estimate(log, case):
    """A finding should say how large the fault is, not only that one exists."""
    worst = screen_log(case.apply(log)).problems()[0]
    assert worst.estimate is not None
    assert np.isfinite(worst.estimate)


def test_an_undetected_fault_still_damages_the_log(log):
    """The blind spot is a real fault, not a no-op dressed up as one.

    Without this the catalogue could claim a blind spot while injecting
    nothing, and the suite would pass on a screening that works perfectly.
    """
    case = fault_case("mixture_sluggish")
    faulty = case.apply(log)
    assert not np.array_equal(faulty.afr, log.afr)
    assert screen_log(faulty).worst == OK
