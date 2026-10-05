"""Sample selection for VE table calibration.

A logged sample is only useful for calibration if its air-fuel ratio reading
reflects a fuelling error at a known operating point. Several conditions break
that link:

- overrun fuel cut, where the sensor reads ambient air and no fuelling error
  exists at all;
- transients, where the operating point moves appreciably while the sensor is
  still responding, so the reading belongs to a different table cell;
- operation outside the table axes, where the ECU clamps its table read and
  no cell represents the condition;
- implausible readings outside the sensor's working range.

This module identifies and rejects those samples. Each criterion is exposed
separately so its individual contribution can be measured.
"""

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from vetuner.sensors import SensorLog
from vetuner.ve_surface import DEFAULT_MAP_AXIS, DEFAULT_RPM_AXIS


@dataclass(frozen=True)
class GatingConfig:
    """Thresholds controlling which log samples are accepted.

    Attributes
    ----------
    afr_min, afr_max : float
        Plausible air-fuel ratio range. Readings outside it indicate a
        saturated or faulty sensor rather than a fuelling error.
    overrun_tps : float
        Throttle position below which overrun is suspected, as a fraction.
    overrun_rpm : float
        Engine speed above which a closed throttle implies overrun, in rpm.
    sensor_tau_s : float
        Assumed air-fuel ratio sensor response time, in seconds. Sets how far
        the operating point may drift before a reading is mis-attributed.
    max_cell_displacement : float
        Largest operating point movement tolerated during one sensor response
        time, in table cell widths. Values below one keep the reading within
        the cell it came from.
    max_tps_rate : float
        Largest throttle movement rate accepted, in fraction per second.
        Guards against fuel film transients that manifold pressure alone does
        not capture.
    settle_time_s : float
        Period after any transient during which samples remain rejected,
        allowing the sensor reading to catch up.
        overrun_recovery_s : float
        Period after fuel cut during which the air-fuel ratio sensor is still
        decaying from its lean saturation, in seconds. Roughly five sensor
        time constants.
    """

    afr_min: float = 8.0
    afr_max: float = 20.0
    overrun_tps: float = 0.03
    overrun_rpm: float = 1500.0
    overrun_recovery_s: float = 0.6
    sensor_tau_s: float = 0.12
    max_cell_displacement: float = 0.2
    max_tps_rate: float = 0.5
    settle_time_s: float = 0.2
    


@dataclass(frozen=True)
class GatingResult:
    """Outcome of applying gating criteria to a log.

    The per-criterion masks overlap, since a sample may fail several tests at
    once. Their individual counts therefore sum to more than the total number
    rejected.

    Attributes
    ----------
    accepted : numpy.ndarray
        Boolean mask of samples usable for calibration.
    implausible_afr : numpy.ndarray
        Samples whose air-fuel ratio lies outside the sensor's working range.
    overrun : numpy.ndarray
        Samples at closed throttle and elevated engine speed.
    out_of_domain : numpy.ndarray
        Samples outside the table axes.
    transient : numpy.ndarray
        Samples where the operating point was moving too quickly, including
        the settling period that follows.
    n_total, n_accepted : int
        Sample counts.
    post_overrun : numpy.ndarray
        Samples during the sensor's recovery from overrun saturation.
    """

    accepted: NDArray[np.bool_]
    implausible_afr: NDArray[np.bool_]
    overrun: NDArray[np.bool_]
    post_overrun: NDArray[np.bool_]
    out_of_domain: NDArray[np.bool_]
    transient: NDArray[np.bool_]
    n_total: int
    n_accepted: int

    @property
    def acceptance_rate(self) -> float:
        """Fraction of samples accepted."""
        return self.n_accepted / self.n_total if self.n_total else 0.0

    def rejection_summary(self) -> dict[str, int]:
        """Return the number of samples failing each criterion.

        Returns
        -------
        dict
            Criterion name to count. Counts overlap and do not sum to the
            total rejected.
        """
        return {
            "implausible_afr": int(self.implausible_afr.sum()),
            "overrun": int(self.overrun.sum()),
            "post_overrun": int(self.post_overrun.sum()),
            "out_of_domain": int(self.out_of_domain.sum()),
            "transient": int(self.transient.sum()),
            "rejected_total": int(self.n_total - self.n_accepted),
        }


def signal_rate(values: ArrayLike, time_s: ArrayLike) -> NDArray[np.float64]:
    """Differentiate a logged channel with respect to time.

    Uses a second-order central difference in the interior and a first-order
    difference at each end.

    Parameters
    ----------
    values : array_like
        Channel samples.
    time_s : array_like
        Sample times in seconds, strictly increasing.

    Returns
    -------
    numpy.ndarray
        Rate of change per second, same shape as `values`.

    Raises
    ------
    ValueError
        If fewer than two samples are supplied or the arrays differ in length.
    """
    vals = np.asarray(values, dtype=float)
    times = np.asarray(time_s, dtype=float)

    if vals.shape != times.shape:
        raise ValueError("Values and times must have the same shape")
    if vals.size < 2:
        raise ValueError("At least two samples are required to compute a rate")

    return np.gradient(vals, times)


def extend_forward(mask: ArrayLike, n_samples: int) -> NDArray[np.bool_]:
    """Extend a boolean mask forward in time.

    Each flagged sample also flags the `n_samples` that follow it, so that a
    rejection persists while a lagging sensor recovers.

    Parameters
    ----------
    mask : array_like
        Boolean mask to extend.
    n_samples : int
        Number of following samples to flag. Zero or negative returns a copy.

    Returns
    -------
    numpy.ndarray
        The extended mask.
    """
    base = np.asarray(mask, dtype=bool)
    if n_samples <= 0:
        return base.copy()

    out = base.copy()
    for shift in range(1, min(n_samples, len(base) - 1) + 1):
        out[shift:] |= base[:-shift]
    return out


def implausible_afr_mask(log: SensorLog, config: GatingConfig) -> NDArray[np.bool_]:
    """Flag air-fuel ratio readings outside the sensor's working range.

    Parameters
    ----------
    log : SensorLog
        Measured engine data.
    config : GatingConfig
        Gating thresholds.

    Returns
    -------
    numpy.ndarray
        Boolean mask, True where the reading is implausible.
    """
    return (log.afr < config.afr_min) | (log.afr > config.afr_max)


def overrun_mask(log: SensorLog, config: GatingConfig) -> NDArray[np.bool_]:
    """Flag samples where the engine is likely in overrun fuel cut.

    A closed throttle at elevated engine speed means the engine is being driven
    by the vehicle rather than driving it, and most ECUs cut fuel entirely.
    The sensor then reads ambient air, which is not a fuelling error.

    Parameters
    ----------
    log : SensorLog
        Measured engine data.
    config : GatingConfig
        Gating thresholds.

    Returns
    -------
    numpy.ndarray
        Boolean mask, True where overrun is suspected.

    Notes
    -----
    Inferred from throttle position and engine speed alone, since a datalogger
    does not record the ECU's internal fuel cut state.
    """
    return (log.tps < config.overrun_tps) & (log.rpm > config.overrun_rpm)


def out_of_domain_mask(
    log: SensorLog,
    rpm_axis: ArrayLike = DEFAULT_RPM_AXIS,
    map_axis: ArrayLike = DEFAULT_MAP_AXIS,
) -> NDArray[np.bool_]:
    """Flag samples lying outside the table axes.

    The ECU clamps a table read to the nearest edge node, so it applies a
    volumetric efficiency belonging to a different operating point. The
    resulting fuelling error is real but cannot be attributed to any cell.

    Parameters
    ----------
    log : SensorLog
        Measured engine data.
    rpm_axis, map_axis : array_like, optional
        Table breakpoints.

    Returns
    -------
    numpy.ndarray
        Boolean mask, True where the sample lies outside the table.
    """
    rpm_ax = np.asarray(rpm_axis, dtype=float)
    map_ax = np.asarray(map_axis, dtype=float)

    return (
        (log.rpm < rpm_ax[0])
        | (log.rpm > rpm_ax[-1])
        | (log.map_kpa < map_ax[0])
        | (log.map_kpa > map_ax[-1])
    )


def transient_mask(
    log: SensorLog,
    config: GatingConfig,
    rpm_axis: ArrayLike = DEFAULT_RPM_AXIS,
    map_axis: ArrayLike = DEFAULT_MAP_AXIS,
) -> NDArray[np.bool_]:
    """Flag samples where the operating point is moving too quickly.

    The test is how far the operating point travels during one sensor response
    time, expressed in table cell widths. A reading that arrives after the
    engine has moved to a different cell would be attributed to the wrong one.

    Expressing the criterion this way accepts a sustained acceleration at
    constant load, where the operating point drifts slowly across the table,
    while rejecting a throttle step, where it jumps several cells before the
    sensor responds.

    Parameters
    ----------
    log : SensorLog
        Measured engine data.
    config : GatingConfig
        Gating thresholds.
    rpm_axis, map_axis : array_like, optional
        Table breakpoints, used to determine cell widths.

    Returns
    -------
    numpy.ndarray
        Boolean mask, True during a transient or its settling period.

    Notes
    -----
    Throttle rate is tested separately because rapid throttle movement causes
    fuel film transients on the port walls that manifold pressure does not
    reveal. Wall wetting is not modelled, so those samples are rejected rather
    than corrected.
    """
    rpm_cell = float(np.mean(np.diff(np.asarray(rpm_axis, dtype=float))))
    map_cell = float(np.mean(np.diff(np.asarray(map_axis, dtype=float))))

    rpm_cells = (
        np.abs(smoothed_rate(log.rpm, log.time_s)) * config.sensor_tau_s / rpm_cell
    )
    map_cells = (
        np.abs(smoothed_rate(log.map_kpa, log.time_s)) * config.sensor_tau_s / map_cell
    )
    tps_rate = np.abs(smoothed_rate(log.tps, log.time_s))

    moving = (
        (rpm_cells > config.max_cell_displacement)
        | (map_cells > config.max_cell_displacement)
        | (tps_rate > config.max_tps_rate)
    )

    dt = float(log.time_s[1] - log.time_s[0])
    return extend_forward(moving, int(round(config.settle_time_s / dt)))


def gate_samples(
    log: SensorLog,
    config: GatingConfig = GatingConfig(),
    rpm_axis: ArrayLike = DEFAULT_RPM_AXIS,
    map_axis: ArrayLike = DEFAULT_MAP_AXIS,
) -> GatingResult:
    """Apply all gating criteria to a log.

    Parameters
    ----------
    log : SensorLog
        Measured engine data.
    config : GatingConfig, optional
        Gating thresholds.
    rpm_axis, map_axis : array_like, optional
        Table breakpoints.

    Returns
    -------
    GatingResult
        The combined acceptance mask and each criterion's mask.

    Raises
    ------
    ValueError
        If the log contains fewer than two samples.
    """
    if len(log) < 2:
        raise ValueError("At least two samples are required for gating")

    implausible = implausible_afr_mask(log, config)
    overrun = overrun_mask(log, config)
    post_overrun = post_overrun_mask(log, config)
    out_of_domain = out_of_domain_mask(log, rpm_axis, map_axis)
    transient = transient_mask(log, config, rpm_axis, map_axis)

    accepted = ~(implausible | overrun | post_overrun | out_of_domain | transient)

    return GatingResult(
        accepted=accepted,
        implausible_afr=implausible,
        overrun=overrun,
        post_overrun=post_overrun,
        out_of_domain=out_of_domain,
        transient=transient,
        n_total=len(log),
        n_accepted=int(accepted.sum()),
    )
    
def smoothed_rate(
    values: ArrayLike, time_s: ArrayLike, window_s: float = 0.2
) -> NDArray[np.float64]:
    """Differentiate a channel after smoothing, to suppress noise amplification.

    Differentiation amplifies high-frequency noise: a channel carrying noise of
    standard deviation sigma, sampled at interval dt, yields a spurious rate of
    order sigma/dt. Smoothing first removes that component while preserving the
    genuine rate of change, which varies over much longer timescales.

    Parameters
    ----------
    values : array_like
        Channel samples.
    time_s : array_like
        Sample times in seconds, uniformly spaced.
    window_s : float, optional
        Smoothing window in seconds, by default 0.2.

    Returns
    -------
    numpy.ndarray
        Rate of change per second, same shape as `values`.
    """
    vals = np.asarray(values, dtype=float)
    times = np.asarray(time_s, dtype=float)

    dt = float(times[1] - times[0])
    width = max(int(round(window_s / dt)), 1)
    if width > 1:
        kernel = np.ones(width) / width
        vals = np.convolve(vals, kernel, mode="same")
        vals[: width // 2] = vals[width // 2]
        vals[-(width // 2) :] = vals[-(width // 2) - 1]

    return np.gradient(vals, times)

def post_overrun_mask(
    log: SensorLog, config: GatingConfig
) -> NDArray[np.bool_]:
    """Flag samples immediately following overrun fuel cut.

    During fuel cut the sensor saturates on ambient air at its lean limit.
    When fuelling resumes the reading decays towards the true mixture over
    several sensor time constants, so the first samples after overrun report a
    mixture far leaner than the engine is actually running.

    These samples are hazardous because every other criterion accepts them:
    the throttle is open, the engine speed is ordinary, the reading lies
    within the sensor's working range and the operating point is inside the
    table. Only their history reveals them.

    Parameters
    ----------
    log : SensorLog
        Measured engine data.
    config : GatingConfig
        Gating thresholds.

    Returns
    -------
    numpy.ndarray
        Boolean mask, True during the recovery period after fuel cut.
    """
    dt = float(log.time_s[1] - log.time_s[0])
    recovery = int(round(config.overrun_recovery_s / dt))
    return extend_forward(overrun_mask(log, config), recovery)