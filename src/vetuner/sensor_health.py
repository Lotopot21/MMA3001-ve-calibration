"""Sensor health screening for logged engine data.

Gating decides whether an individual reading is usable. This module asks a
different question: whether the instrument that produced the readings can be
trusted at all.

The distinction matters because the two failure modes look nothing alike. A
reading outside the sensor's working range is obviously unusable and gating
rejects it. A sensor that is biased, stuck or sluggish produces entirely
plausible readings that happen to be wrong, every criterion accepts them, and
the calibration faithfully compensates for the instrument fault by bending the
volumetric efficiency table. The resulting table looks exactly as trustworthy
as a correct one.

Each check returns a `Finding` with a severity rather than silently repairing
the data. Repair would require assuming the fault's form, trusting an inferred
magnitude, and would leave no trace in the output. Reporting the fault with a
quantified estimate leaves the decision, and the record of it, with the person
running the tool.

Notes
-----
Two fault classes are not detectable from a log alone, and both are left
unreported rather than guessed at.

A wideband sensor with a gain error produces measurements indistinguishable
from a volumetric efficiency table that is uniformly wrong by the same factor,
because both yield the same air-fuel ratio at every operating point. Nothing in
the log separates them; the sensor needs an external reference, which is why
real wideband controllers provide free-air calibration.

A sluggish wideband sensor is the more troubling gap, because it is the most
damaging fault measured against this pipeline: quadrupling the sensor time
constant raises table error from 1.18 to 4.29 VE points, worse than the
pressure bias this module refuses outright, while every check here passes.
Detection was attempted by cross-correlating the air-fuel ratio channel against
manifold pressure to measure its effective lag, and abandoned: a fourfold
increase in time constant moved the measured lag by 0.06 s against a
cycle-dependent baseline of 0.46 s, and an exhaust transport delay of 0.1 s
shifted it by the same amount. A sluggish sensor and a long exhaust path are
both lags, so no statistic of a single log distinguishes them. Checking sensor
response belongs with the sensor, against a known step, not with the
calibration.
"""

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

from vetuner.sensors import SensorLog

OK = "ok"
"""Severity: the check passed."""

WARNING = "warning"
"""Severity: the data is degraded but still usable."""

CRITICAL = "critical"
"""Severity: the data cannot support a trustworthy calibration."""

_SEVERITY_RANK = {OK: 0, WARNING: 1, CRITICAL: 2}

PLAUSIBLE_RANGES: dict[str, tuple[float, float]] = {
    "rpm": (0.0, 12000.0),
    "map_kpa": (0.0, 300.0),
    "iat_c": (-40.0, 150.0),
    "tps": (0.0, 1.0),
    "afr": (5.0, 30.0),
    "batt_v": (6.0, 18.0),
}
"""Physically possible range of each channel.

These are the limits of what an instrument could report at all, deliberately
wider than the range in which a reading is useful for calibration. A value
outside them indicates a sensor or logger fault rather than an unusual
operating condition.
"""

CLIPPING_RAILS: dict[str, tuple[float, ...]] = {
    "rpm": (0.0,),
    "map_kpa": (0.0,),
    "tps": (0.0, 1.0),
    "afr": (22.0,),
}
"""Values at which each channel is pinned by the end of its own range.

A reading held at one of these is legitimately constant: the quantity has
moved outside what the instrument can resolve, so the reading stops varying
without anything being wrong. Throttle rests at its closed and wide-open
stops, and a wideband sensor saturates on ambient air whenever fuel is cut.
Channels absent from this mapping have no rail the engine reaches in normal
use.
"""

STUCK_CHECK_CHANNELS: tuple[str, ...] = ("rpm", "map_kpa", "iat_c", "afr")
"""Channels on which an unchanging reading is evidence of a fault.

Throttle position and battery voltage are left out because constancy is their
normal behaviour rather than a symptom: throttle is a driver input held at
whatever setting is wanted, and battery voltage is held near constant by the
alternator's regulator. Both would be reported as stuck on any log quiet
enough for their plateaus to show through the noise. The remaining four all
respond continuously to engine state, so a reading that stops moving on one of
them has stopped tracking something that is still changing.
"""


class SensorFaultError(RuntimeError):
    """Raised when a log carries a fault severe enough to invalidate calibration."""


@dataclass(frozen=True)
class Finding:
    """One check's verdict on one channel.

    Attributes
    ----------
    channel : str
        Name of the log channel examined, or ``"log"`` for whole-record checks.
    check : str
        Short identifier of the check that produced this finding.
    severity : str
        One of `OK`, `WARNING` or `CRITICAL`.
    message : str
        Human-readable description of what was found.
    estimate : float, optional
        Quantified magnitude of the fault where the check can supply one, such
        as an apparent sensor bias in kPa. None when the check is qualitative.
    """

    channel: str
    check: str
    severity: str
    message: str
    estimate: float | None = None

    def __str__(self) -> str:
        """Return a single formatted line describing this finding."""
        mark = {OK: "ok  ", WARNING: "WARN", CRITICAL: "CRIT"}[self.severity]
        return f"[{mark}] {self.channel:<9} {self.message}"


@dataclass(frozen=True)
class HealthReport:
    """Collected findings from screening one log.

    Attributes
    ----------
    findings : list of Finding
        Every check's verdict, including those that passed.
    """

    findings: list[Finding] = field(default_factory=list)

    @property
    def worst(self) -> str:
        """Return the highest severity among all findings."""
        if not self.findings:
            return OK
        return max(self.findings, key=lambda f: _SEVERITY_RANK[f.severity]).severity

    @property
    def usable(self) -> bool:
        """Return whether the log is free of critical faults."""
        return self.worst != CRITICAL

    def problems(self) -> list[Finding]:
        """Return only the findings that are not `OK`.

        Returns
        -------
        list of Finding
            Warnings and critical findings, worst first.
        """
        flagged = [f for f in self.findings if f.severity != OK]
        return sorted(flagged, key=lambda f: -_SEVERITY_RANK[f.severity])

    def __str__(self) -> str:
        """Return a readable multi-line summary of the screening."""
        header = f"Sensor health: {self.worst.upper()} ({len(self.findings)} checks run)"
        flagged = self.problems()
        if not flagged:
            return header + "\n  no faults detected"
        return "\n".join([header, *(f"  {f}" for f in flagged)])


@dataclass(frozen=True)
class HealthConfig:
    """Thresholds controlling the sensor health checks.

    Attributes
    ----------
    barometric_kpa : float
        Ambient pressure the manifold is expected to approach at wide-open
        throttle, in kPa absolute.
    barometric_warning_kpa, barometric_critical_kpa : float
        Excess above ambient at which a manifold pressure reading is treated
        as biased, in kPa.
    barometric_shortfall_kpa : float
        Shortfall below ambient at wide-open throttle that suggests either a
        low-reading sensor or a restricted intake, in kPa.
    wot_threshold : float
        Throttle position above which the manifold is expected to approach
        ambient pressure, as a fraction.
    min_wot_samples : int
        Fewest wide-open-throttle samples needed before the barometric check
        can be attempted.
    stuck_seconds_warning, stuck_seconds_critical : float
        Duration for which a channel may hold an unchanging value before it is
        reported as stuck, in seconds.
    resolution_limit_fraction : float
        Fraction of `stuck_seconds_warning` that a channel's typical dwell on
        one ADC value may reach before the stuck check is abandoned as unable
        to distinguish a fault.
    saturation_warning, saturation_critical : float
        Proportion of air-fuel ratio samples at a sensor rail above which the
        reading is reported as saturated.
    afr_rail_low, afr_rail_high : float
        Air-fuel ratio values treated as the sensor's rails.
    timing_jitter : float
        Permitted variation in the sample interval, as a fraction of its
        median.
    """

    barometric_kpa: float = 101.3
    barometric_warning_kpa: float = 1.5
    barometric_critical_kpa: float = 5.0
    barometric_shortfall_kpa: float = 12.0
    wot_threshold: float = 0.90
    min_wot_samples: int = 50

    stuck_seconds_warning: float = 2.0
    stuck_seconds_critical: float = 10.0
    resolution_limit_fraction: float = 0.25

    saturation_warning: float = 0.20
    saturation_critical: float = 0.50
    afr_rail_low: float = 8.0
    afr_rail_high: float = 20.0

    timing_jitter: float = 0.05


def unchanged_runs(
    values: NDArray[np.float64], ignore: NDArray[np.bool_] | None = None
) -> tuple[NDArray[np.int64], NDArray[np.int64]]:
    """Split a signal into stretches over which it does not change at all.

    Parameters
    ----------
    values : numpy.ndarray
        Samples to examine.
    ignore : numpy.ndarray, optional
        Boolean mask of samples that may not take part in a run. Each is
        treated as a run of its own, so a stretch spanning one is broken in
        two rather than counted whole.

    Returns
    -------
    lengths : numpy.ndarray
        Length in samples of each run, in the order they occur.
    starts : numpy.ndarray
        Index at which each run begins.

    Notes
    -----
    Comparison is exact rather than tolerance-based, which is appropriate
    because every channel is quantised onto an ADC grid: successive readings
    are either the same grid value or a different one, with nothing in between.
    """
    if len(values) == 0:
        empty = np.empty(0, dtype=np.int64)
        return empty, empty

    steady = np.diff(values) == 0.0
    if ignore is not None:
        steady &= ~(ignore[1:] | ignore[:-1])

    # Run boundaries sit either side of each change, with the signal's ends
    # closing the first and last runs.
    edges = np.concatenate(([-1], np.flatnonzero(~steady), [len(values) - 1]))
    return np.diff(edges), edges[:-1] + 1


def longest_unchanged_run(
    values: NDArray[np.float64], ignore: NDArray[np.bool_] | None = None
) -> tuple[int, int]:
    """Find the longest stretch over which a signal does not change at all.

    Parameters
    ----------
    values : numpy.ndarray
        Samples to examine.
    ignore : numpy.ndarray, optional
        Boolean mask of samples that may not take part in a run, as for
        `unchanged_runs`.

    Returns
    -------
    length : int
        Number of samples in the longest run of identical consecutive values.
        One for a signal that changes at every step, and `len(values)` for a
        constant signal with nothing ignored.
    start : int
        Index at which that run begins. Zero when `values` is empty.
    """
    lengths, starts = unchanged_runs(values, ignore)
    if len(lengths) == 0:
        return 0, 0
    best = int(np.argmax(lengths))
    return int(lengths[best]), int(starts[best])


def check_time_base(log: SensorLog, config: HealthConfig = HealthConfig()) -> Finding:
    """Verify that the log is sampled at a steady rate with no gaps.

    An irregular time base invalidates every rate calculation downstream,
    including the transient gating and the sensor lag estimate, because those
    assume a uniform interval.

    Parameters
    ----------
    log : SensorLog
        Measured engine data.
    config : HealthConfig, optional
        Screening thresholds.

    Returns
    -------
    Finding
        Verdict on the log's time base.
    """
    if len(log) < 3:
        return Finding("log", "time_base", CRITICAL, "fewer than three samples")

    dt = np.diff(log.time_s)
    median = float(np.median(dt))

    if median <= 0.0 or np.any(dt <= 0.0):
        return Finding(
            "log", "time_base", CRITICAL, "timestamps are not strictly increasing"
        )

    jitter = float(np.max(np.abs(dt - median)) / median)
    if jitter > config.timing_jitter:
        n_gaps = int(np.sum(dt > median * (1.0 + config.timing_jitter)))
        longest = float(dt.max())
        return Finding(
            "log",
            "time_base",
            WARNING,
            f"{n_gaps} gap{'s' if n_gaps != 1 else ''} in a {median * 1e3:.1f} ms "
            f"time base, the longest {longest:.2f} s at "
            f"t={log.time_s[int(np.argmax(dt))]:.1f} s",
            estimate=longest,
        )

    return Finding(
        "log",
        "time_base",
        OK,
        f"uniform at {1.0 / median:.0f} Hz over {log.time_s[-1]:.0f} s",
        estimate=1.0 / median,
    )


def check_channel_ranges(
    log: SensorLog, config: HealthConfig = HealthConfig()
) -> list[Finding]:
    """Verify that every channel stays within physically possible limits.

    A value outside `PLAUSIBLE_RANGES` cannot be produced by a working sensor
    on a running engine, so it indicates an instrument or logger fault rather
    than an unusual operating condition.

    Parameters
    ----------
    log : SensorLog
        Measured engine data.
    config : HealthConfig, optional
        Screening thresholds. Unused by this check, accepted for consistency.

    Returns
    -------
    list of Finding
        One verdict per channel.
    """
    findings: list[Finding] = []
    for channel, (low, high) in PLAUSIBLE_RANGES.items():
        values = getattr(log, channel)

        if not np.all(np.isfinite(values)):
            n_bad = int(np.sum(~np.isfinite(values)))
            findings.append(
                Finding(
                    channel,
                    "range",
                    CRITICAL,
                    f"{n_bad} non-finite sample{'s' if n_bad != 1 else ''}",
                    estimate=float(n_bad),
                )
            )
            continue

        outside = (values < low) | (values > high)
        n_out = int(outside.sum())
        if n_out == 0:
            findings.append(
                Finding(
                    channel,
                    "range",
                    OK,
                    f"within {low:g} to {high:g} "
                    f"(observed {values.min():.1f} to {values.max():.1f})",
                )
            )
            continue

        fraction = n_out / len(values)
        severity = CRITICAL if fraction > 0.01 else WARNING
        findings.append(
            Finding(
                channel,
                "range",
                severity,
                f"{n_out} samples ({fraction:.1%}) outside {low:g} to {high:g}",
                estimate=fraction,
            )
        )
    return findings


def check_stuck_channels(
    log: SensorLog, config: HealthConfig = HealthConfig()
) -> list[Finding]:
    """Detect channels that latch at a fixed value and stop reporting.

    Every channel carries noise and sits on an ADC grid, so a working sensor
    changes its reading constantly even when the engine is held steady: on a
    healthy log the longest unchanging stretch on any channel is a fraction of
    a second. A sensor that has failed short, open or frozen holds one value
    for as long as the fault lasts, which separates the two cases by orders of
    magnitude rather than by a finely tuned threshold.

    Parameters
    ----------
    log : SensorLog
        Measured engine data.
    config : HealthConfig, optional
        Screening thresholds.

    Returns
    -------
    list of Finding
        One verdict per channel in `STUCK_CHECK_CHANNELS`, carrying the
        duration of its longest unchanging stretch in seconds.

    Notes
    -----
    Samples sitting at a channel's `CLIPPING_RAILS` are excluded, because a
    pinned reading is constant for a legitimate reason: a wideband sensor
    saturates on ambient air whenever fuel is cut. The cost is that a sensor
    which fails to a rail value escapes this particular check; a total failure
    is still caught as a whole-log constant, and an impossible value by
    `check_channel_ranges`.

    The check also relies on the channel resolving its own movement. Where a
    quantity changes by less than one ADC step over the fault threshold, a
    working sensor is already flat for that long and the two cases cannot be
    separated at all: on a log with a tenth of realistic noise, a ten-second
    freeze on the temperature channel is statistically indistinguishable from
    healthy data. The check detects that situation and declines to judge rather
    than reporting a fault it cannot support.
    """
    findings: list[Finding] = []
    dt = float(np.median(np.diff(log.time_s))) if len(log) > 1 else 0.0

    for channel in STUCK_CHECK_CHANNELS:
        values = getattr(log, channel)

        if len(values) and float(np.ptp(values)) == 0.0:
            findings.append(
                Finding(
                    channel,
                    "stuck",
                    CRITICAL,
                    f"constant at {values[0]:.2f} for the whole log; "
                    f"the channel carries no information",
                    estimate=len(values) * dt,
                )
            )
            continue

        rails = CLIPPING_RAILS.get(channel, ())
        at_rail = np.isin(values, rails) if rails else None
        lengths, starts = unchanged_runs(values, at_rail)

        # A channel whose quantity moves slower than one ADC step per
        # threshold holds each grid value for a comparable time anyway, so a
        # long run carries no evidence either way.
        typical_s = float(np.percentile(lengths, 99)) * dt
        if typical_s > config.resolution_limit_fraction * config.stuck_seconds_warning:
            findings.append(
                Finding(
                    channel,
                    "stuck",
                    OK,
                    f"not tested: resolution-limited, holding each value "
                    f"{typical_s:.1f} s even when working",
                    estimate=typical_s,
                )
            )
            continue

        best = int(np.argmax(lengths))
        length, start = int(lengths[best]), int(starts[best])
        held_s = length * dt

        if held_s > config.stuck_seconds_critical:
            severity: str | None = CRITICAL
        elif held_s > config.stuck_seconds_warning:
            severity = WARNING
        else:
            severity = None

        if severity is not None:
            findings.append(
                Finding(
                    channel,
                    "stuck",
                    severity,
                    f"held {values[start]:.2f} for {held_s:.1f} s from "
                    f"t={log.time_s[start]:.1f} s; sensor appears stuck",
                    estimate=held_s,
                )
            )
        else:
            findings.append(
                Finding(
                    channel,
                    "stuck",
                    OK,
                    f"never held one value longer than {held_s:.2f} s",
                    estimate=held_s,
                )
            )
    return findings


def check_map_barometric(
    log: SensorLog, config: HealthConfig = HealthConfig()
) -> Finding:
    """Check manifold pressure against ambient pressure at wide-open throttle.

    A naturally aspirated engine cannot raise manifold pressure above ambient,
    so with the throttle fully open the reading should approach barometric
    pressure without exceeding it. That gives an absolute reference against
    which a pressure sensor can be checked using the log alone.

    The check matters because manifold pressure bias is the single largest
    sensitivity in the calibration and is otherwise invisible: the readings
    stay inside every plausible range, gating accepts them, and the bias is
    absorbed into the volumetric efficiency table.

    Parameters
    ----------
    log : SensorLog
        Measured engine data.
    config : HealthConfig, optional
        Screening thresholds.

    Returns
    -------
    Finding
        Verdict on the manifold pressure sensor, carrying the apparent bias in
        kPa where one is detected.

    Notes
    -----
    A reading that falls well short of ambient is reported as a warning rather
    than a fault, because a low-reading sensor and a restricted intake produce
    the same symptom and the log cannot separate them.
    """
    wot = log.tps > config.wot_threshold
    if int(wot.sum()) < config.min_wot_samples:
        return Finding(
            "map_kpa",
            "barometric",
            OK,
            f"not tested: only {int(wot.sum())} wide-open-throttle samples",
        )

    observed = float(np.percentile(log.map_kpa[wot], 99))
    excess = observed - config.barometric_kpa

    if excess > config.barometric_critical_kpa:
        return Finding(
            "map_kpa",
            "barometric",
            CRITICAL,
            f"reads {observed:.1f} kPa at wide-open throttle, "
            f"{excess:+.1f} kPa above ambient and physically impossible",
            estimate=excess,
        )

    if excess > config.barometric_warning_kpa:
        return Finding(
            "map_kpa",
            "barometric",
            WARNING,
            f"reads {observed:.1f} kPa at wide-open throttle, "
            f"{excess:+.1f} kPa above ambient; sensor appears biased high",
            estimate=excess,
        )

    if excess < -config.barometric_shortfall_kpa:
        return Finding(
            "map_kpa",
            "barometric",
            WARNING,
            f"reaches only {observed:.1f} kPa at wide-open throttle, "
            f"{excess:+.1f} kPa below ambient; sensor may read low, "
            f"or the intake may be restricted",
            estimate=excess,
        )

    return Finding(
        "map_kpa",
        "barometric",
        OK,
        f"reaches {observed:.1f} kPa at wide-open throttle "
        f"({excess:+.1f} kPa against ambient)",
        estimate=excess,
    )


def check_afr_saturation(
    log: SensorLog, config: HealthConfig = HealthConfig()
) -> Finding:
    """Measure how much of the log sits at an air-fuel ratio sensor rail.

    A saturated reading states that the mixture is outside the sensor's range
    but not by how much, so it carries no information the correction can use.
    Some saturation is expected during overrun fuel cut; a large proportion
    means the starting calibration is too far out for this log to correct.

    Parameters
    ----------
    log : SensorLog
        Measured engine data.
    config : HealthConfig, optional
        Screening thresholds.

    Returns
    -------
    Finding
        Verdict on how much usable air-fuel ratio data the log contains.
    """
    at_rail = (log.afr >= config.afr_rail_high) | (log.afr <= config.afr_rail_low)
    fraction = float(at_rail.mean())

    if fraction > config.saturation_critical:
        return Finding(
            "afr",
            "saturation",
            CRITICAL,
            f"{fraction:.0%} of samples at a sensor rail; too little usable "
            f"data to calibrate, and the starting map is likely far out",
            estimate=fraction,
        )

    if fraction > config.saturation_warning:
        return Finding(
            "afr",
            "saturation",
            WARNING,
            f"{fraction:.0%} of samples at a sensor rail, beyond what overrun "
            f"alone explains",
            estimate=fraction,
        )

    return Finding(
        "afr",
        "saturation",
        OK,
        f"{fraction:.1%} of samples at a rail, consistent with overrun",
        estimate=fraction,
    )


def screen_log(log: SensorLog, config: HealthConfig = HealthConfig()) -> HealthReport:
    """Run every sensor health check against a log.

    Parameters
    ----------
    log : SensorLog
        Measured engine data.
    config : HealthConfig, optional
        Screening thresholds.

    Returns
    -------
    HealthReport
        Every check's verdict, whose `worst` severity determines whether the
        log should be calibrated against.

    Raises
    ------
    ValueError
        If the log contains no samples.
    """
    if len(log) == 0:
        raise ValueError("Cannot screen an empty log")

    findings = [check_time_base(log, config)]
    findings.extend(check_channel_ranges(log, config))
    findings.extend(check_stuck_channels(log, config))
    findings.append(check_map_barometric(log, config))
    findings.append(check_afr_saturation(log, config))
    return HealthReport(findings)


def assert_usable(report: HealthReport) -> None:
    """Stop the calibration if the screening found a critical fault.

    Calibrating against a faulty instrument produces a table that compensates
    for the fault, and nothing in the output distinguishes it from a correct
    one. Refusing is therefore safer than proceeding with a caveat.

    Parameters
    ----------
    report : HealthReport
        Output of `screen_log`.

    Raises
    ------
    SensorFaultError
        If any finding is `CRITICAL`.
    """
    if not report.usable:
        faults = "\n".join(
            f"  {f}" for f in report.problems() if f.severity == CRITICAL
        )
        raise SensorFaultError(
            "Log carries a critical sensor fault; calibration would compensate "
            f"for the instrument rather than the engine.\n{faults}"
        )
