"""Deliberate sensor faults, for testing the screening that detects them.

A detector can only be trusted against data whose faults are known, so this
module builds the faulty half of a matched pair: one log straight from the
sensor model, and the same log with one named fault injected. Screening the
pair shows both that a fault is caught and that a healthy log is left alone,
which a faulty log on its own cannot demonstrate.

Faults are applied to the log rather than to the engine, because that is where
they belong physically. A sensor reporting the wrong number does not change
what the engine did; it changes only what was recorded about it. The forward
model therefore runs untouched and the corruption is applied afterwards, in the
same way `vetuner.sensors.SensorConfig` already applies measurement bias.

Every injector returns a new log, takes its parameters in seconds rather than
samples, and is deterministic given its arguments.
"""

from collections.abc import Callable
from dataclasses import dataclass, replace

import numpy as np
from numpy.typing import NDArray

from vetuner.drive_cycle import first_order_lag
from vetuner.sensor_health import CRITICAL, OK, WARNING
from vetuner.sensors import SensorLog

FAULTY_CHANNELS: tuple[str, ...] = ("rpm", "map_kpa", "iat_c", "tps", "afr", "batt_v")
"""Channels a fault may be injected on."""


def _sample_interval(log: SensorLog) -> float:
    """Return the log's sample interval in seconds.

    Parameters
    ----------
    log : SensorLog
        Measured engine data, with at least two samples.

    Returns
    -------
    float
        Median interval between samples.

    Raises
    ------
    ValueError
        If the log has fewer than two samples.
    """
    if len(log) < 2:
        raise ValueError("Need at least two samples to determine a sample interval")
    return float(np.median(np.diff(log.time_s)))


def _window(log: SensorLog, start_s: float, duration_s: float) -> NDArray[np.bool_]:
    """Return a mask selecting a time window of the log.

    Parameters
    ----------
    log : SensorLog
        Measured engine data.
    start_s : float
        Window start, in seconds from the beginning of the log.
    duration_s : float
        Window length in seconds. Must be positive.

    Returns
    -------
    numpy.ndarray
        Boolean mask over samples.

    Raises
    ------
    ValueError
        If `duration_s` is not positive, or the window selects no samples.
    """
    if duration_s <= 0.0:
        raise ValueError("Fault duration must be positive")

    mask = (log.time_s >= start_s) & (log.time_s < start_s + duration_s)
    if not mask.any():
        raise ValueError(
            f"Window {start_s} to {start_s + duration_s} s selects no samples; "
            f"the log spans {log.time_s[0]} to {log.time_s[-1]} s"
        )
    return mask


def _check_channel(channel: str) -> None:
    """Reject a channel name a fault cannot be injected on.

    Parameters
    ----------
    channel : str
        Name to validate.

    Raises
    ------
    ValueError
        If `channel` is not one of `FAULTY_CHANNELS`.
    """
    if channel not in FAULTY_CHANNELS:
        raise ValueError(
            f"Unknown channel {channel!r}; expected one of {', '.join(FAULTY_CHANNELS)}"
        )


def freeze(
    log: SensorLog, channel: str, start_s: float, duration_s: float
) -> SensorLog:
    """Latch a channel at one value, as a failed sensor does.

    Represents a sensor that has failed short, open or frozen: it keeps
    reporting whatever it read when the fault began. The value is taken from
    the log itself rather than invented, so the reading stays plausible and
    only its refusal to move gives the fault away.

    Parameters
    ----------
    log : SensorLog
        Measured engine data.
    channel : str
        Channel to freeze.
    start_s : float
        Time at which the sensor fails, in seconds.
    duration_s : float
        How long the fault lasts, in seconds.

    Returns
    -------
    SensorLog
        Log with the channel held constant over the window.
    """
    _check_channel(channel)
    mask = _window(log, start_s, duration_s)
    values = getattr(log, channel).copy()
    values[mask] = values[mask][0]
    return replace(log, **{channel: values})


def offset(log: SensorLog, channel: str, amount: float) -> SensorLog:
    """Shift a whole channel by a constant, as a miscalibrated sensor does.

    This is the fault the screening exists for. The readings stay inside every
    plausible range and vary normally, so nothing local marks them as wrong;
    only a comparison against an absolute reference reveals the offset.

    Parameters
    ----------
    log : SensorLog
        Measured engine data.
    channel : str
        Channel to shift.
    amount : float
        Offset to add, in the channel's own units.

    Returns
    -------
    SensorLog
        Log with the channel shifted throughout.
    """
    _check_channel(channel)
    return replace(log, **{channel: getattr(log, channel) + amount})


def saturate(
    log: SensorLog, channel: str, value: float, start_s: float, duration_s: float
) -> SensorLog:
    """Pin a channel at a fixed reading, as a sensor at the end of its range does.

    Represents a wideband sensor whose mixture has left the range it can
    resolve. The reading states that the quantity is beyond the limit but not
    by how much, so it carries nothing the correction can use.

    Parameters
    ----------
    log : SensorLog
        Measured engine data.
    channel : str
        Channel to pin.
    value : float
        Reading to hold, in the channel's own units.
    start_s : float
        Time at which saturation begins, in seconds.
    duration_s : float
        How long it lasts, in seconds.

    Returns
    -------
    SensorLog
        Log with the channel pinned over the window.
    """
    _check_channel(channel)
    mask = _window(log, start_s, duration_s)
    values = getattr(log, channel).copy()
    values[mask] = value
    return replace(log, **{channel: values})


def drop_samples(log: SensorLog, start_s: float, duration_s: float) -> SensorLog:
    """Remove a block of samples, as a logger that stopped writing does.

    Every channel loses the same samples, so the result is a log with a hole in
    its time base rather than a gap in one signal. This matters because the
    rate calculations downstream assume a uniform sample interval, and nothing
    in the remaining values reveals that the assumption has been broken.

    Parameters
    ----------
    log : SensorLog
        Measured engine data.
    start_s : float
        Start of the dropped block, in seconds.
    duration_s : float
        Length of the dropped block, in seconds.

    Returns
    -------
    SensorLog
        Shorter log with the block removed from every channel.
    """
    keep = ~_window(log, start_s, duration_s)
    return SensorLog(
        time_s=log.time_s[keep],
        rpm=log.rpm[keep],
        map_kpa=log.map_kpa[keep],
        iat_c=log.iat_c[keep],
        tps=log.tps[keep],
        afr=log.afr[keep],
        batt_v=log.batt_v[keep],
    )


def invalidate(
    log: SensorLog, channel: str, count: int, seed: int = 0
) -> SensorLog:
    """Replace scattered samples with not-a-number, as a dropped reading does.

    Represents a channel whose value was missing or unparseable in the logged
    file. Arithmetic on a non-finite sample silently contaminates whatever it
    reaches, so these are worth finding before any of it runs.

    Parameters
    ----------
    log : SensorLog
        Measured engine data.
    channel : str
        Channel to damage.
    count : int
        How many samples to invalidate. Must be positive and no more than the
        log length.
    seed : int, optional
        Random seed selecting which samples, by default 0. Fixing it makes the
        fault reproducible.

    Returns
    -------
    SensorLog
        Log with `count` samples of the channel set to NaN.

    Raises
    ------
    ValueError
        If `count` is not positive or exceeds the number of samples.
    """
    _check_channel(channel)
    if not 0 < count <= len(log):
        raise ValueError(f"Cannot invalidate {count} of {len(log)} samples")

    rng = np.random.default_rng(seed)
    chosen = rng.choice(len(log), size=count, replace=False)
    values = getattr(log, channel).copy()
    values[chosen] = np.nan
    return replace(log, **{channel: values})


def sluggish(log: SensorLog, channel: str, tau_s: float) -> SensorLog:
    """Slow a channel's response, as a contaminated or ageing sensor does.

    Applies a further first-order lag on top of whatever the sensor model
    already applied. The reading remains smooth, plausible and correctly
    ranged; it simply arrives late, which makes this the fault the screening
    cannot see.

    Parameters
    ----------
    log : SensorLog
        Measured engine data.
    channel : str
        Channel to slow.
    tau_s : float
        Additional time constant, in seconds.

    Returns
    -------
    SensorLog
        Log with the channel's response slowed.
    """
    _check_channel(channel)
    lagged = first_order_lag(getattr(log, channel), _sample_interval(log), tau_s)
    return replace(log, **{channel: lagged})


@dataclass(frozen=True)
class FaultCase:
    """One named fault, with the verdict screening is expected to reach.

    Pairing the injector with its expected outcome turns the catalogue into a
    specification: the tests assert that screening reaches these verdicts, so a
    check that stops working, or starts firing on healthy data, fails the suite
    rather than going unnoticed.

    Attributes
    ----------
    name : str
        Short identifier.
    description : str
        What the fault represents physically.
    apply : callable
        Takes a healthy log and returns the faulty one.
    expected_severity : str
        Worst severity screening should report: `OK`, `WARNING` or `CRITICAL`.
    expected_channel : str or None
        Channel the finding should name, or None where the fault is expected to
        go undetected.
    """

    name: str
    description: str
    apply: Callable[[SensorLog], SensorLog]
    expected_severity: str
    expected_channel: str | None

    @property
    def detected(self) -> bool:
        """Return whether screening is expected to report this fault at all."""
        return self.expected_severity != OK


FAULT_CASES: tuple[FaultCase, ...] = (
    FaultCase(
        name="healthy",
        description="no fault; the control case",
        apply=lambda log: log,
        expected_severity=OK,
        expected_channel=None,
    ),
    FaultCase(
        name="pressure_bias_small",
        description="manifold pressure sensor reading 2 kPa high",
        apply=lambda log: offset(log, "map_kpa", 2.0),
        expected_severity=WARNING,
        expected_channel="map_kpa",
    ),
    FaultCase(
        name="pressure_bias_large",
        description="manifold pressure sensor reading 6 kPa high",
        apply=lambda log: offset(log, "map_kpa", 6.0),
        expected_severity=CRITICAL,
        expected_channel="map_kpa",
    ),
    FaultCase(
        name="pressure_frozen",
        description="manifold pressure sensor latching partway through the log",
        apply=lambda log: freeze(log, "map_kpa", 120.0, 60.0),
        expected_severity=CRITICAL,
        expected_channel="map_kpa",
    ),
    FaultCase(
        name="temperature_frozen_briefly",
        description="intake temperature sensor dropping out for three seconds",
        apply=lambda log: freeze(log, "iat_c", 60.0, 3.0),
        expected_severity=WARNING,
        expected_channel="iat_c",
    ),
    FaultCase(
        name="temperature_dead",
        description="intake temperature sensor reporting one value throughout",
        apply=lambda log: freeze(log, "iat_c", 0.0, 1e6),
        expected_severity=CRITICAL,
        expected_channel="iat_c",
    ),
    FaultCase(
        name="mixture_unreadable",
        description="wideband sensor pinned lean for half the log",
        apply=lambda log: saturate(log, "afr", 21.0, 0.0, 120.0),
        expected_severity=CRITICAL,
        expected_channel="afr",
    ),
    FaultCase(
        name="logger_gap",
        description="datalogger stopping for four seconds mid-run",
        apply=lambda log: drop_samples(log, 90.0, 4.0),
        expected_severity=WARNING,
        expected_channel="log",
    ),
    FaultCase(
        name="missing_readings",
        description="scattered air-fuel ratio samples missing from the file",
        apply=lambda log: invalidate(log, "afr", 40),
        expected_severity=CRITICAL,
        expected_channel="afr",
    ),
    FaultCase(
        name="speed_impossible",
        description="engine speed channel glitching far beyond redline",
        apply=lambda log: saturate(log, "rpm", 15000.0, 30.0, 10.0),
        expected_severity=CRITICAL,
        expected_channel="rpm",
    ),
    FaultCase(
        name="mixture_sluggish",
        description="ageing wideband sensor responding slowly; known blind spot",
        apply=lambda log: sluggish(log, "afr", 0.36),
        expected_severity=OK,
        expected_channel=None,
    ),
)
"""Faults used to demonstrate and test the screening.

The catalogue deliberately includes a healthy control and a fault known to be
undetectable, so that the summary it produces reports the screening's blind
spot alongside its successes rather than only the cases that work.
"""


def fault_case(name: str) -> FaultCase:
    """Look up one fault case by name.

    Parameters
    ----------
    name : str
        Name of the case, as given in `FAULT_CASES`.

    Returns
    -------
    FaultCase
        The matching case.

    Raises
    ------
    KeyError
        If no case has that name.
    """
    for case in FAULT_CASES:
        if case.name == name:
            return case
    available = ", ".join(c.name for c in FAULT_CASES)
    raise KeyError(f"No fault case named {name!r}; available: {available}")
