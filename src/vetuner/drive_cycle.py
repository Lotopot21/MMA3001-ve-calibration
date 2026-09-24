"""Synthetic drive cycle generation.

Produces a time series of engine operating conditions (speed, throttle
position, manifold pressure, intake air temperature and battery voltage)
for feeding the forward engine model.

The cycle prescribes the operating point trajectory directly. Vehicle and
drivetrain dynamics are not modelled, because the calibration pipeline is
sensitive only to which operating points are visited and how quickly they
change, not to how the vehicle arrived at them.
"""

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

P_ATM_KPA = 101.3
"""Assumed ambient pressure at sea level, in kPa absolute."""

MAP_FLOOR_KPA = 8.0
"""Lower clamp on manifold pressure, representing sealing and leakage limits."""

THROTTLE_LEAK_AREA = 0.015
"""Effective flow area at closed throttle, as a fraction of the full-open area.

Represents the idle bypass and plate clearance. Sets the idle vacuum level.
"""

THROTTLE_RESTRICTION = 0.35
"""Dimensionless restriction coefficient relating engine speed to pressure drop.

Chosen so that wide-open throttle at redline gives a manifold pressure a few
kPa below ambient, as expected for a naturally aspirated engine.
"""

RPM_MAX = 7000.0
"""Reference speed used to non-dimensionalise engine speed in the throttle model."""


@dataclass(frozen=True)
class Segment:
    """One phase of a drive cycle, linearly ramped from start to end.

    Attributes
    ----------
    duration_s : float
        Length of the segment in seconds. Must be positive.
    rpm_start, rpm_end : float
        Engine speed at the start and end of the segment, in rpm.
    tps_start, tps_end : float
        Throttle position at the start and end of the segment, as a
        fraction in [0, 1].
    label : str
        Human-readable description, used for plotting and debugging.
    """

    duration_s: float
    rpm_start: float
    rpm_end: float
    tps_start: float
    tps_end: float
    label: str = ""


@dataclass(frozen=True)
class DriveCycle:
    """Sampled engine operating conditions over time.

    Attributes
    ----------
    time_s : numpy.ndarray
        Sample times in seconds, uniformly spaced and starting at zero.
    rpm : numpy.ndarray
        Engine speed in rpm.
    tps : numpy.ndarray
        Throttle position as a fraction in [0, 1].
    map_kpa : numpy.ndarray
        Manifold absolute pressure in kPa.
    iat_c : numpy.ndarray
        Intake air temperature in degrees Celsius.
    batt_v : numpy.ndarray
        Battery voltage in volts.
    """

    time_s: NDArray[np.float64]
    rpm: NDArray[np.float64]
    tps: NDArray[np.float64]
    map_kpa: NDArray[np.float64]
    iat_c: NDArray[np.float64]
    batt_v: NDArray[np.float64]

    def __len__(self) -> int:
        """Return the number of samples in the cycle."""
        return len(self.time_s)


DEFAULT_CYCLE: tuple[Segment, ...] = (
    Segment(20.0, 800.0, 800.0, 0.00, 0.00, "warm idle"),
    Segment(8.0, 800.0, 2000.0, 0.00, 0.15, "pull away"),
    Segment(25.0, 2000.0, 2050.0, 0.15, 0.16, "low cruise"),
    Segment(6.0, 2050.0, 3000.0, 0.16, 0.20, "accelerate"),
    Segment(30.0, 3000.0, 3020.0, 0.20, 0.20, "mid cruise"),
    Segment(25.0, 3500.0, 3520.0, 0.22, 0.22, "faster cruise"),
    Segment(8.0, 3520.0, 4500.0, 0.22, 0.30, "accelerate"),
    Segment(20.0, 4500.0, 4520.0, 0.30, 0.30, "high cruise"),
    Segment(6.0, 4520.0, 2000.0, 0.00, 0.00, "overrun"),
    Segment(12.0, 2500.0, 7000.0, 1.00, 1.00, "wide open throttle pull"),
    Segment(5.0, 7000.0, 2000.0, 0.00, 0.00, "overrun"),
    Segment(20.0, 2500.0, 2520.0, 0.18, 0.18, "cruise"),
    Segment(10.0, 3000.0, 6800.0, 1.00, 1.00, "second pull"),
    Segment(15.0, 1200.0, 800.0, 0.00, 0.00, "return to idle"),
)
"""A representative road cycle covering idle, cruise, acceleration and overrun."""

VALIDATION_CYCLE: tuple[Segment, ...] = (
    Segment(15.0, 800.0, 800.0, 0.00, 0.00, "idle"),
    Segment(10.0, 800.0, 2600.0, 0.00, 0.22, "pull away"),
    Segment(20.0, 2600.0, 2650.0, 0.22, 0.22, "cruise"),
    Segment(8.0, 2650.0, 3800.0, 0.22, 0.26, "accelerate"),
    Segment(22.0, 3800.0, 3850.0, 0.26, 0.26, "cruise"),
    Segment(10.0, 3200.0, 6400.0, 0.85, 0.85, "part throttle pull"),
    Segment(6.0, 6400.0, 2200.0, 0.00, 0.00, "overrun"),
    Segment(18.0, 2200.0, 2250.0, 0.14, 0.14, "light cruise"),
    Segment(12.0, 2800.0, 7000.0, 1.00, 1.00, "full pull"),
    Segment(12.0, 1500.0, 800.0, 0.00, 0.00, "return to idle"),
)
"""An independent cycle held out for testing, never used to fit a table.

Visits different operating points from `DEFAULT_CYCLE`, including a sustained
part-throttle pull, so that a table fitted on one cycle is scored on
conditions it was not tuned against.
"""

def first_order_lag(signal: ArrayLike, dt: float, tau_s: float) -> NDArray[np.float64]:
    """Apply a discrete first-order lag to a signal.

    Implements ``y[i] = y[i-1] + alpha * (x[i] - y[i-1])`` with
    ``alpha = dt / (tau + dt)``, the backward-Euler discretisation of
    ``tau * dy/dt + y = x``.

    Parameters
    ----------
    signal : array_like
        Input samples, uniformly spaced in time.
    dt : float
        Sample interval in seconds. Must be positive.
    tau_s : float
        Time constant in seconds. Zero or negative returns the input unchanged.

    Returns
    -------
    numpy.ndarray
        Lagged signal, same shape as `signal`.

    Raises
    ------
    ValueError
        If `dt` is not positive.
    """
    x = np.asarray(signal, dtype=float)
    if dt <= 0.0:
        raise ValueError("Sample interval must be positive")
    if tau_s <= 0.0:
        return x.copy()

    alpha = dt / (tau_s + dt)
    y = np.empty_like(x)
    y[0] = x[0]
    for i in range(1, len(x)):
        y[i] = y[i - 1] + alpha * (x[i] - y[i - 1])
    return y


def manifold_pressure(rpm: ArrayLike, tps: ArrayLike) -> NDArray[np.float64]:
    """Compute manifold absolute pressure from engine speed and throttle position.

    Models the throttle plate as a flow restriction. Manifold pressure settles
    where flow past the plate balances the engine's pumping demand, so opening
    the throttle raises it and increasing engine speed lowers it.

    Parameters
    ----------
    rpm : array_like
        Engine speed in rpm. Must be non-negative.
    tps : array_like
        Throttle position as a fraction in [0, 1].

    Returns
    -------
    numpy.ndarray
        Manifold absolute pressure in kPa, clamped below at `MAP_FLOOR_KPA`
        and above at `P_ATM_KPA`.

    Raises
    ------
    ValueError
        If any engine speed is negative or any throttle position lies
        outside [0, 1].

    Notes
    -----
    This is a static model. Manifold filling dynamics are not represented;
    the manifold is assumed to reach its steady-state pressure instantly.
    Since the drive cycle already lags throttle position, the resulting
    pressure trace still varies smoothly.
    """
    rpm_arr = np.asarray(rpm, dtype=float)
    tps_arr = np.asarray(tps, dtype=float)

    if np.any(rpm_arr < 0.0):
        raise ValueError("Engine speed must be non-negative")
    if np.any((tps_arr < 0.0) | (tps_arr > 1.0)):
        raise ValueError("Throttle position must lie in [0, 1]")

    area = THROTTLE_LEAK_AREA + (1.0 - THROTTLE_LEAK_AREA) * tps_arr**2
    restriction = THROTTLE_RESTRICTION * (rpm_arr / RPM_MAX) / area
    pressure = P_ATM_KPA / np.sqrt(1.0 + restriction**2)
    return np.clip(pressure, MAP_FLOOR_KPA, P_ATM_KPA)


def intake_air_temperature(
    map_kpa: ArrayLike,
    ambient_c: float = 25.0,
    soak_c: float = 18.0,
) -> NDArray[np.float64]:
    """Estimate intake air temperature from manifold pressure.

    At low airflow the charge spends longer in the manifold and picks up more
    underbonnet heat, so intake temperature rises as manifold pressure falls.

    Parameters
    ----------
    map_kpa : array_like
        Manifold absolute pressure in kPa.
    ambient_c : float, optional
        Ambient air temperature in degrees Celsius, by default 25.0.
    soak_c : float, optional
        Maximum heat pickup at zero airflow, in Kelvin, by default 18.0.

    Returns
    -------
    numpy.ndarray
        Intake air temperature in degrees Celsius.
    """
    map_arr = np.asarray(map_kpa, dtype=float)
    return ambient_c + soak_c * (1.0 - map_arr / P_ATM_KPA)


def battery_voltage(rpm: ArrayLike) -> NDArray[np.float64]:
    """Estimate charging system voltage from engine speed.

    Voltage is lower at idle, where the alternator is turning slowly, and
    reaches its regulated value once the engine is spinning.

    Parameters
    ----------
    rpm : array_like
        Engine speed in rpm.

    Returns
    -------
    numpy.ndarray
        Battery voltage in volts.
    """
    rpm_arr = np.asarray(rpm, dtype=float)
    return 13.6 + 0.6 * np.clip(rpm_arr / 2000.0, 0.0, 1.0)


def generate_drive_cycle(
    cycle: tuple[Segment, ...] = DEFAULT_CYCLE,
    sample_rate_hz: float = 50.0,
    rpm_tau_s: float = 0.35,
    tps_tau_s: float = 0.15,
) -> DriveCycle:
    """Generate a sampled drive cycle from a sequence of segments.

    Segment endpoints are joined by linear ramps, then both engine speed and
    throttle position are passed through a first-order lag so that segment
    boundaries produce smooth, physically plausible transients rather than
    instantaneous steps.

    Parameters
    ----------
    cycle : tuple of Segment, optional
        Segments to concatenate, by default `DEFAULT_CYCLE`.
    sample_rate_hz : float, optional
        Logging rate in Hz, by default 50.0, typical of aftermarket ECU logs.
    rpm_tau_s : float, optional
        Engine speed lag time constant in seconds, by default 0.35,
        representing rotating inertia.
    tps_tau_s : float, optional
        Throttle lag time constant in seconds, by default 0.15,
        representing driver and actuator response.

    Returns
    -------
    DriveCycle
        Sampled operating conditions.

    Raises
    ------
    ValueError
        If `cycle` is empty, `sample_rate_hz` is not positive, or any segment
        has a non-positive duration.
    """
    if not cycle:
        raise ValueError("Drive cycle must contain at least one segment")
    if sample_rate_hz <= 0.0:
        raise ValueError("Sample rate must be positive")
    if any(seg.duration_s <= 0.0 for seg in cycle):
        raise ValueError("Every segment must have a positive duration")

    dt = 1.0 / sample_rate_hz
    rpm_parts: list[NDArray[np.float64]] = []
    tps_parts: list[NDArray[np.float64]] = []

    for seg in cycle:
        n = max(int(round(seg.duration_s * sample_rate_hz)), 1)
        rpm_parts.append(np.linspace(seg.rpm_start, seg.rpm_end, n, endpoint=False))
        tps_parts.append(np.linspace(seg.tps_start, seg.tps_end, n, endpoint=False))

    rpm_raw = np.concatenate(rpm_parts)
    tps_raw = np.clip(np.concatenate(tps_parts), 0.0, 1.0)

    rpm = first_order_lag(rpm_raw, dt, rpm_tau_s)
    tps = np.clip(first_order_lag(tps_raw, dt, tps_tau_s), 0.0, 1.0)

    map_kpa = manifold_pressure(rpm, tps)
    time_s = np.arange(len(rpm), dtype=float) * dt

    return DriveCycle(
        time_s=time_s,
        rpm=rpm,
        tps=tps,
        map_kpa=map_kpa,
        iat_c=intake_air_temperature(map_kpa),
        batt_v=battery_voltage(rpm),
    )