"""Tests for sensor health screening."""

from dataclasses import replace

import numpy as np
import pytest

from vetuner.drive_cycle import VALIDATION_CYCLE, generate_drive_cycle
from vetuner.engine_model import simulate_engine
from vetuner.sensor_health import (
    CRITICAL,
    OK,
    PLAUSIBLE_RANGES,
    STUCK_CHECK_CHANNELS,
    WARNING,
    Finding,
    HealthConfig,
    HealthReport,
    SensorFaultError,
    assert_usable,
    check_afr_saturation,
    check_channel_ranges,
    check_map_barometric,
    check_stuck_channels,
    check_time_base,
    longest_unchanged_run,
    screen_log,
    unchanged_runs,
)
from vetuner.sensors import SensorConfig, SensorLog, apply_sensor_model
from vetuner.ve_surface import BASE_MAP_SURFACE, ve_table


@pytest.fixture
def sim():
    return simulate_engine(generate_drive_cycle(), ve_table(params=BASE_MAP_SURFACE))


@pytest.fixture
def log(sim):
    return apply_sensor_model(sim, seed=0)


def finding_for(findings, channel, check):
    """Return the single finding matching a channel and check name."""
    matches = [f for f in findings if f.channel == channel and f.check == check]
    assert len(matches) == 1, f"expected one {check} finding for {channel}"
    return matches[0]


# --- longest_unchanged_run -------------------------------------------------


def test_run_length_of_a_strictly_changing_signal_is_one():
    length, start = longest_unchanged_run(np.array([1.0, 2.0, 3.0, 4.0]))
    assert (length, start) == (1, 0)


def test_run_length_of_a_constant_signal_is_its_length():
    length, start = longest_unchanged_run(np.full(7, 3.0))
    assert (length, start) == (7, 0)


def test_run_length_locates_an_interior_plateau():
    values = np.array([1.0, 2.0, 5.0, 5.0, 5.0, 5.0, 9.0])
    assert longest_unchanged_run(values) == (4, 2)


def test_run_length_finds_a_plateau_that_reaches_the_end():
    values = np.array([1.0, 2.0, 7.0, 7.0, 7.0])
    assert longest_unchanged_run(values) == (3, 2)


def test_run_length_of_an_empty_signal_is_zero():
    assert longest_unchanged_run(np.empty(0)) == (0, 0)


def test_run_length_reports_the_first_of_two_equal_longest_runs():
    values = np.array([1.0, 1.0, 2.0, 3.0, 3.0])
    assert longest_unchanged_run(values) == (2, 0)


def test_runs_partition_the_signal():
    values = np.array([1.0, 1.0, 2.0, 2.0, 2.0, 9.0])
    lengths, starts = unchanged_runs(values)
    assert lengths.sum() == len(values)
    assert list(starts) == [0, 2, 5]
    assert list(lengths) == [2, 3, 1]


def test_runs_of_an_empty_signal_are_empty():
    lengths, starts = unchanged_runs(np.empty(0))
    assert len(lengths) == 0
    assert len(starts) == 0


def test_an_ignored_sample_breaks_a_run_in_two():
    values = np.full(9, 4.0)
    ignore = np.zeros(9, dtype=bool)
    ignore[4] = True
    length, start = longest_unchanged_run(values, ignore)
    assert length == 4
    assert start in (0, 5)


def test_ignoring_nothing_matches_the_plain_run():
    values = np.array([1.0, 1.0, 1.0, 2.0])
    ignore = np.zeros(4, dtype=bool)
    assert longest_unchanged_run(values, ignore) == longest_unchanged_run(values)


# --- report aggregation ----------------------------------------------------


def test_empty_report_is_ok_and_usable():
    report = HealthReport()
    assert report.worst == OK
    assert report.usable


def test_report_takes_the_worst_severity_regardless_of_order():
    findings = [
        Finding("a", "x", OK, "fine"),
        Finding("b", "y", CRITICAL, "broken"),
        Finding("c", "z", WARNING, "iffy"),
    ]
    assert HealthReport(findings).worst == CRITICAL
    assert HealthReport(list(reversed(findings))).worst == CRITICAL


def test_report_is_unusable_only_when_a_finding_is_critical():
    assert HealthReport([Finding("a", "x", WARNING, "iffy")]).usable
    assert not HealthReport([Finding("a", "x", CRITICAL, "broken")]).usable


def test_problems_excludes_passes_and_orders_worst_first():
    report = HealthReport(
        [
            Finding("a", "x", OK, "fine"),
            Finding("b", "y", WARNING, "iffy"),
            Finding("c", "z", CRITICAL, "broken"),
        ]
    )
    assert [f.severity for f in report.problems()] == [CRITICAL, WARNING]


def test_report_text_says_so_when_nothing_is_wrong():
    report = HealthReport([Finding("a", "x", OK, "fine")])
    assert "no faults detected" in str(report)


def test_report_text_lists_each_problem():
    report = HealthReport(
        [Finding("map_kpa", "barometric", WARNING, "biased high by 2 kPa")]
    )
    text = str(report)
    assert "WARNING" in text
    assert "biased high by 2 kPa" in text


def test_assert_usable_passes_a_warning_but_stops_on_critical():
    assert_usable(HealthReport([Finding("a", "x", WARNING, "iffy")]))
    with pytest.raises(SensorFaultError, match="critical sensor fault"):
        assert_usable(HealthReport([Finding("a", "x", CRITICAL, "broken")]))


def test_assert_usable_names_the_faulty_channel():
    report = HealthReport([Finding("map_kpa", "stuck", CRITICAL, "held 32.1 kPa")])
    with pytest.raises(SensorFaultError, match="map_kpa"):
        assert_usable(report)


# --- healthy logs pass -----------------------------------------------------


@pytest.mark.parametrize("seed", range(6))
def test_a_healthy_log_raises_no_findings(sim, seed):
    assert screen_log(apply_sensor_model(sim, seed=seed)).problems() == []


def test_the_validation_cycle_is_also_clean():
    sim = simulate_engine(
        generate_drive_cycle(VALIDATION_CYCLE), ve_table(params=BASE_MAP_SURFACE)
    )
    assert screen_log(apply_sensor_model(sim, seed=0)).worst == OK


def test_screening_covers_every_channel(log):
    report = screen_log(log)
    checked = {(f.channel, f.check) for f in report.findings}
    for channel in PLAUSIBLE_RANGES:
        assert (channel, "range") in checked
    for channel in STUCK_CHECK_CHANNELS:
        assert (channel, "stuck") in checked
    assert ("log", "time_base") in checked
    assert ("map_kpa", "barometric") in checked
    assert ("afr", "saturation") in checked


def test_screening_an_empty_log_is_an_error():
    nothing = np.empty(0)
    empty = SensorLog(
        time_s=nothing,
        rpm=nothing,
        map_kpa=nothing,
        iat_c=nothing,
        tps=nothing,
        afr=nothing,
        batt_v=nothing,
    )
    with pytest.raises(ValueError, match="empty log"):
        screen_log(empty)


# --- time base -------------------------------------------------------------


def test_uniform_sampling_reports_the_logging_rate(log):
    finding = check_time_base(log)
    assert finding.severity == OK
    assert finding.estimate == pytest.approx(50.0, abs=0.1)


def test_a_dropped_block_of_samples_is_reported(log):
    keep = np.ones(len(log), dtype=bool)
    keep[2000:2100] = False
    gapped = replace(log, time_s=log.time_s[keep])
    finding = check_time_base(gapped)
    assert finding.severity == WARNING
    assert "gap" in finding.message


def test_non_increasing_timestamps_are_critical(log):
    time_s = log.time_s.copy()
    time_s[500] = time_s[499]
    finding = check_time_base(replace(log, time_s=time_s))
    assert finding.severity == CRITICAL
    assert "increasing" in finding.message


def test_too_few_samples_to_judge_the_time_base(log):
    finding = check_time_base(replace(log, time_s=np.array([0.0, 0.02])))
    assert finding.severity == CRITICAL


# --- channel ranges --------------------------------------------------------


def test_a_healthy_log_is_within_every_plausible_range(log):
    assert all(f.severity == OK for f in check_channel_ranges(log))


def test_non_finite_samples_are_critical(log):
    afr = log.afr.copy()
    afr[5] = np.nan
    finding = finding_for(check_channel_ranges(replace(log, afr=afr)), "afr", "range")
    assert finding.severity == CRITICAL
    assert "non-finite sample" in finding.message


def test_a_few_impossible_samples_are_a_warning(log):
    rpm = log.rpm.copy()
    rpm[:20] = 15000.0
    finding = finding_for(check_channel_ranges(replace(log, rpm=rpm)), "rpm", "range")
    assert finding.severity == WARNING


def test_many_impossible_samples_are_critical(log):
    rpm = log.rpm.copy()
    rpm[:300] = 15000.0
    finding = finding_for(check_channel_ranges(replace(log, rpm=rpm)), "rpm", "range")
    assert finding.severity == CRITICAL
    assert finding.estimate == pytest.approx(300 / len(log))


# --- stuck channels --------------------------------------------------------


def test_healthy_channels_never_hold_a_value_for_long(log):
    for finding in check_stuck_channels(log):
        assert finding.severity == OK
        assert finding.estimate < 1.0


def test_driver_input_and_regulated_channels_are_not_tested(log):
    """Throttle and battery voltage are constant by design, not by fault.

    Throttle rests wherever the driver holds it and battery voltage is pinned
    by the alternator's regulator, so a flat stretch on either is normal. They
    are excluded rather than given a looser threshold, because on a log quiet
    enough for their plateaus to show through the noise no threshold separates
    them from a genuine fault.
    """
    assert "tps" not in STUCK_CHECK_CHANNELS
    assert "batt_v" not in STUCK_CHECK_CHANNELS
    tested = {f.channel for f in check_stuck_channels(log)}
    assert tested == set(STUCK_CHECK_CHANNELS)


def test_a_channel_frozen_mid_log_is_critical(log):
    frozen = log.time_s >= 100.0
    stuck = replace(log, map_kpa=np.where(frozen, log.map_kpa[frozen][0], log.map_kpa))
    finding = finding_for(check_stuck_channels(stuck), "map_kpa", "stuck")
    assert finding.severity == CRITICAL
    assert finding.estimate == pytest.approx(110.0, abs=0.1)
    assert "t=100.0 s" in finding.message


def test_a_brief_freeze_is_a_warning_not_a_fault(log):
    frozen = (log.time_s >= 50.0) & (log.time_s < 54.0)
    stuck = replace(log, afr=np.where(frozen, log.afr[frozen][0], log.afr))
    finding = finding_for(check_stuck_channels(stuck), "afr", "stuck")
    assert finding.severity == WARNING
    assert finding.estimate == pytest.approx(4.0, abs=0.1)


def test_a_channel_constant_throughout_is_called_out_as_carrying_nothing(log):
    stuck = replace(log, iat_c=np.full_like(log.iat_c, 30.0))
    finding = finding_for(check_stuck_channels(stuck), "iat_c", "stuck")
    assert finding.severity == CRITICAL
    assert "whole log" in finding.message


def test_the_freeze_threshold_separates_healthy_logs_by_a_wide_margin(log):
    """The longest healthy stretch should sit far below the warning threshold.

    A threshold that only just clears real data would fire on a different
    drive cycle or noise seed, so the margin is what makes the check usable
    rather than the threshold value itself.
    """
    worst = max(f.estimate for f in check_stuck_channels(log))
    assert worst < HealthConfig().stuck_seconds_warning / 5.0


QUIETER_SENSORS = SensorConfig(
    afr_noise_std=0.01,
    map_noise_std_kpa=0.06,
    iat_noise_std_c=0.03,
    rpm_noise_std=0.3,
    tps_noise_std=0.0003,
    batt_noise_std_v=0.005,
)
"""A tenth of the realistic noise on every channel.

Quieter instruments are the hard case for the freeze check: with less noise a
quantised reading holds each grid value longer, which is what the
resolution-limited precondition exists to recognise.
"""


@pytest.mark.parametrize("seed", range(3))
def test_quieter_sensors_do_not_look_stuck(sim, seed):
    log = apply_sensor_model(sim, QUIETER_SENSORS, seed=seed)
    assert screen_log(log).problems() == []


def test_a_resolution_limited_channel_declines_to_judge_rather_than_guessing(sim):
    """Temperature moves slower than its own ADC step on a quiet log.

    Intake temperature changes by a few degrees over minutes while its reading
    is quantised in quarter-degree steps, so a working sensor already holds one
    value for seconds. The check says it cannot tell instead of reporting a
    fault it has no evidence for.
    """
    log = apply_sensor_model(sim, QUIETER_SENSORS, seed=0)
    finding = finding_for(check_stuck_channels(log), "iat_c", "stuck")
    assert finding.severity == OK
    assert "resolution-limited" in finding.message


def test_a_rail_is_excluded_so_saturation_is_not_mistaken_for_a_freeze():
    """Fuel cut pins the wideband sensor at its lean rail for the whole event.

    That stretch is unchanging for a legitimate reason, so counting it would
    report a fault every time the engine coasted.
    """
    time_s = np.arange(0, 20.0, 0.02)
    afr = 13.0 + 0.1 * np.sin(time_s * 7.0)
    coasting = (time_s >= 5.0) & (time_s < 15.0)
    afr[coasting] = 22.0
    log = SensorLog(
        time_s=time_s,
        rpm=2000.0 + 50.0 * np.sin(time_s * 3.0),
        map_kpa=50.0 + np.sin(time_s * 2.0),
        iat_c=30.0 + 0.1 * np.sin(time_s),
        tps=np.full_like(time_s, 0.3),
        afr=afr,
        batt_v=np.full_like(time_s, 14.0),
    )
    assert finding_for(check_stuck_channels(log), "afr", "stuck").severity == OK


# --- manifold pressure against ambient -------------------------------------


def test_a_healthy_manifold_approaches_ambient_at_wide_open_throttle(log):
    finding = check_map_barometric(log)
    assert finding.severity == OK
    assert finding.estimate == pytest.approx(0.0, abs=1.5)


@pytest.mark.parametrize(
    ("bias", "severity"),
    [(2.0, WARNING), (6.0, CRITICAL), (-15.0, WARNING)],
)
def test_a_biased_pressure_sensor_is_detected(sim, bias, severity):
    log = apply_sensor_model(sim, SensorConfig(map_bias_kpa=bias), seed=0)
    assert check_map_barometric(log).severity == severity


def test_the_reported_bias_tracks_the_injected_one(sim):
    for bias in (2.0, 4.0, 6.0):
        log = apply_sensor_model(sim, SensorConfig(map_bias_kpa=bias), seed=0)
        assert check_map_barometric(log).estimate == pytest.approx(bias, abs=0.5)


def test_pressure_above_ambient_is_called_impossible(sim):
    log = apply_sensor_model(sim, SensorConfig(map_bias_kpa=8.0), seed=0)
    assert "impossible" in check_map_barometric(log).message


def test_a_low_reading_sensor_and_a_blocked_intake_are_not_separated(sim):
    """A shortfall against ambient has two causes the log cannot distinguish."""
    log = apply_sensor_model(sim, SensorConfig(map_bias_kpa=-15.0), seed=0)
    assert "restricted" in check_map_barometric(log).message


def test_the_check_is_skipped_without_wide_open_throttle(log):
    part_throttle = replace(log, tps=np.minimum(log.tps, 0.5))
    finding = check_map_barometric(part_throttle)
    assert finding.severity == OK
    assert "not tested" in finding.message


def test_a_biased_sensor_hides_from_every_other_check(sim):
    """Pressure bias is why this check exists: nothing else sees it.

    A 4 kPa offset stays inside the sensor's plausible range at every sample,
    so without an absolute reference the calibration absorbs it into the table.
    """
    log = apply_sensor_model(sim, SensorConfig(map_bias_kpa=4.0), seed=0)
    others = [
        f
        for f in screen_log(log).findings
        if f.check != "barometric" and f.severity != OK
    ]
    assert others == []
    assert check_map_barometric(log).severity != OK


# --- air-fuel ratio saturation ---------------------------------------------


def test_overrun_alone_leaves_the_sensor_mostly_unsaturated(log):
    finding = check_afr_saturation(log)
    assert finding.severity == OK
    assert finding.estimate < 0.2


@pytest.mark.parametrize(
    ("fraction", "severity"), [(0.3, WARNING), (0.6, CRITICAL)]
)
def test_widespread_saturation_is_reported(log, fraction, severity):
    afr = log.afr.copy()
    afr[: int(len(afr) * fraction)] = 21.0
    assert check_afr_saturation(replace(log, afr=afr)).severity == severity


def test_a_rich_rail_counts_as_saturation_too(log):
    afr = log.afr.copy()
    afr[: int(len(afr) * 0.6)] = 7.0
    assert check_afr_saturation(replace(log, afr=afr)).severity == CRITICAL


# --- configuration ---------------------------------------------------------


def test_thresholds_can_be_tightened_to_fire_on_a_healthy_log(log):
    assert check_map_barometric(log).severity == OK
    strict = HealthConfig(barometric_shortfall_kpa=0.1)
    assert check_map_barometric(log, strict).severity == WARNING


def test_thresholds_can_be_relaxed_to_accept_a_known_fault(sim):
    log = apply_sensor_model(sim, SensorConfig(map_bias_kpa=6.0), seed=0)
    assert check_map_barometric(log).severity == CRITICAL
    tolerant = HealthConfig(barometric_warning_kpa=10.0, barometric_critical_kpa=20.0)
    assert check_map_barometric(log, tolerant).severity == OK
