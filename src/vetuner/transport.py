"""Exhaust transport delay modelling and compensation.

Exhaust gas takes time to travel from the exhaust valve to the wideband
sensor, so a reading taken at one instant describes combustion that occurred
earlier. The delay is the path volume divided by the volumetric flow rate of
exhaust, and since flow rises with both engine speed and load, the delay spans
more than an order of magnitude across the operating range: over a hundred
milliseconds at idle, a few milliseconds at full load.

A constant delay is therefore not a usable approximation. This module computes
the delay from operating conditions, applies it when generating synthetic
logs, and compensates for it when calibrating.
"""

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from vetuner.engine_model import (
    CELSIUS_TO_KELVIN,
    R_SPECIFIC_AIR,
    EngineGeometry,
    air_mass_per_cycle,
)
from vetuner.sensors import SensorLog


@dataclass(frozen=True)
class ExhaustGeometry:
    """Exhaust path properties governing transport delay.

    Attributes
    ----------
    path_volume_l : float
        Volume between the exhaust valve and the sensor, in litres. Covers the
        port, the header primary and any collector upstream of the sensor.
    gas_temperature_c : float
        Representative exhaust gas temperature, in degrees Celsius. Taken as
        constant; in reality it varies with load and spark timing.
    pressure_kpa : float
        Static pressure in the exhaust path, in kPa absolute. Close to ambient
        for a naturally aspirated engine without significant back pressure.
    min_delay_s, max_delay_s : float
        Bounds applied to the computed delay, in seconds. The lower bound
        represents the finite response of the sensor element; the upper bound
        guards against the flow estimate approaching zero at very low airflow.
    """

    path_volume_l: float = 1.2
    gas_temperature_c: float = 700.0
    pressure_kpa: float = 101.3
    min_delay_s: float = 0.002
    max_delay_s: float = 0.300

    @property
    def gas_density_kg_m3(self) -> float:
        """Exhaust gas density from the ideal gas law, in kg/m^3."""
        return (self.pressure_kpa * 1e3) / (
            R_SPECIFIC_AIR * (self.gas_temperature_c + CELSIUS_TO_KELVIN)
        )


def exhaust_mass_flow(
    ve: ArrayLike,
    rpm: ArrayLike,
    map_kpa: ArrayLike,
    iat_c: ArrayLike,
    afr: ArrayLike,
    engine: EngineGeometry = EngineGeometry(),
) -> NDArray[np.float64]:
    """Compute total exhaust mass flow rate.

    Exhaust mass is the air ingested plus the fuel burnt, so it exceeds air
    mass flow by a factor of ``1 + 1/AFR``, about seven percent at
    stoichiometric.

    Parameters
    ----------
    ve : array_like
        Volumetric efficiency as a fraction.
    rpm : array_like
        Engine speed in rpm.
    map_kpa : array_like
        Manifold absolute pressure in kPa.
    iat_c : array_like
        Intake air temperature in degrees Celsius.
    afr : array_like
        Air-fuel ratio by mass.
    engine : EngineGeometry, optional
        Engine geometry.

    Returns
    -------
    numpy.ndarray
        Exhaust mass flow rate in kg/s.

    Raises
    ------
    ValueError
        If any air-fuel ratio is non-positive.
    """
    afr_arr = np.asarray(afr, dtype=float)
    if np.any(afr_arr <= 0.0):
        raise ValueError("Air-fuel ratio must be positive")

    per_cycle = air_mass_per_cycle(ve, map_kpa, iat_c, engine)
    cycles_per_second = np.asarray(rpm, dtype=float) / 60.0 / engine.revolutions_per_cycle
    air_flow = per_cycle * engine.n_cylinders * cycles_per_second
    return air_flow * (1.0 + 1.0 / afr_arr)


def exhaust_transport_delay(
    ve: ArrayLike,
    rpm: ArrayLike,
    map_kpa: ArrayLike,
    iat_c: ArrayLike,
    afr: ArrayLike,
    engine: EngineGeometry = EngineGeometry(),
    exhaust: ExhaustGeometry = ExhaustGeometry(),
) -> NDArray[np.float64]:
    """Compute the exhaust transport delay at each operating point.

    The delay is the time for the exhaust path to be swept once::

        delay = path volume / volumetric flow rate

    Volumetric flow is mass flow divided by exhaust gas density, so the delay
    falls as either engine speed or load rises.

    Parameters
    ----------
    ve : array_like
        Volumetric efficiency as a fraction.
    rpm : array_like
        Engine speed in rpm.
    map_kpa : array_like
        Manifold absolute pressure in kPa.
    iat_c : array_like
        Intake air temperature in degrees Celsius.
    afr : array_like
        Air-fuel ratio by mass.
    engine : EngineGeometry, optional
        Engine geometry.
    exhaust : ExhaustGeometry, optional
        Exhaust path properties.

    Returns
    -------
    numpy.ndarray
        Transport delay in seconds, clamped to the configured bounds.

    Notes
    -----
    This is a plug flow model: the exhaust path is treated as displacing its
    entire contents before new gas reaches the sensor. Real exhaust flow is
    pulsating and partially mixed, so the true response is a distribution of
    arrival times rather than a pure delay. The plug flow estimate gives the
    centroid of that distribution.
    """
    mass_flow = exhaust_mass_flow(ve, rpm, map_kpa, iat_c, afr, engine)
    volumetric_flow = mass_flow / exhaust.gas_density_kg_m3

    with np.errstate(divide="ignore", invalid="ignore"):
        delay = (exhaust.path_volume_l * 1e-3) / volumetric_flow

    delay = np.where(np.isfinite(delay), delay, exhaust.max_delay_s)
    return np.clip(delay, exhaust.min_delay_s, exhaust.max_delay_s)


def apply_variable_delay(
    signal: ArrayLike, time_s: ArrayLike, delay_s: ArrayLike
) -> NDArray[np.float64]:
    """Delay a signal by an amount that varies sample by sample.

    The output at time ``t`` is the input evaluated at ``t - delay(t)``,
    obtained by linear interpolation. Interpolation is necessary rather than
    convenient: at typical logging rates the delay at high engine speed is
    shorter than one sample interval, so shifting by whole samples cannot
    represent it.

    Parameters
    ----------
    signal : array_like
        Input samples.
    time_s : array_like
        Sample times in seconds, strictly increasing.
    delay_s : array_like
        Delay at each sample, in seconds. Must be non-negative.

    Returns
    -------
    numpy.ndarray
        The delayed signal. Times falling before the start of the record take
        the first sample's value.

    Raises
    ------
    ValueError
        If the arrays differ in shape or any delay is negative.
    """
    values = np.asarray(signal, dtype=float)
    times = np.asarray(time_s, dtype=float)
    delays = np.asarray(delay_s, dtype=float)

    if not (values.shape == times.shape == delays.shape):
        raise ValueError("Signal, time and delay arrays must have the same shape")
    if np.any(delays < 0.0):
        raise ValueError("Delay must be non-negative")

    return np.interp(times - delays, times, values)


def compensate_delay(
    log: SensorLog, delay_s: ArrayLike
) -> SensorLog:
    """Re-attribute each air-fuel ratio reading to the operating point that produced it.

    Rather than shifting the air-fuel ratio signal, the operating point
    channels are resampled at ``t - delay(t)``. This is the correct direction:
    the reading itself is valid, it simply describes conditions at an earlier
    instant, so pairing it with the conditions at that instant recovers the
    intended association.

    Parameters
    ----------
    log : SensorLog
        Measured engine data.
    delay_s : array_like
        Estimated transport delay at each sample, in seconds.

    Returns
    -------
    SensorLog
        A log in which engine speed, manifold pressure, intake air temperature
        and throttle position are those in force when each reading's charge was
        burnt.

    Raises
    ------
    ValueError
        If `delay_s` does not match the log length.
    """
    from dataclasses import replace

    delays = np.asarray(delay_s, dtype=float)
    if delays.shape != log.time_s.shape:
        raise ValueError("Delay array must have the same length as the log")

    shifted = log.time_s - delays

    return replace(
        log,
        rpm=np.interp(shifted, log.time_s, log.rpm),
        map_kpa=np.interp(shifted, log.time_s, log.map_kpa),
        iat_c=np.interp(shifted, log.time_s, log.iat_c),
        tps=np.interp(shifted, log.time_s, log.tps),
    )


def estimate_delay_from_log(
    log: SensorLog,
    table: ArrayLike,
    engine: EngineGeometry = EngineGeometry(),
    exhaust: ExhaustGeometry = ExhaustGeometry(),
    rpm_axis: ArrayLike | None = None,
    map_axis: ArrayLike | None = None,
) -> NDArray[np.float64]:
    """Estimate transport delay using only quantities available from a log.

    A calibration pipeline does not know the true volumetric efficiency, so the
    delay is computed from the current table's estimate instead. Because the
    delay depends on the square root of nothing and only linearly on flow, a
    table error of ten percent produces a delay error of about ten percent,
    which is small against the delay's own thirty-fold variation across the
    operating range.

    Parameters
    ----------
    log : SensorLog
        Measured engine data.
    table : array_like
        Current VE table, used to estimate airflow.
    engine : EngineGeometry, optional
        Engine geometry.
    exhaust : ExhaustGeometry, optional
        Exhaust path properties. The path volume is a physical measurement of
        the installation, not something inferred from the log.
    rpm_axis, map_axis : array_like, optional
        Table breakpoints.

    Returns
    -------
    numpy.ndarray
        Estimated delay at each sample, in seconds.
    """
    from vetuner.engine_model import read_ve_table
    from vetuner.ve_surface import DEFAULT_MAP_AXIS, DEFAULT_RPM_AXIS

    rpm_ax = DEFAULT_RPM_AXIS if rpm_axis is None else rpm_axis
    map_ax = DEFAULT_MAP_AXIS if map_axis is None else map_axis

    ve = read_ve_table(table, log.rpm, log.map_kpa, rpm_ax, map_ax)
    return exhaust_transport_delay(
        ve, log.rpm, log.map_kpa, log.iat_c, log.afr, engine, exhaust
    )