"""Sensor model.

Corrupts simulated engine measurements to resemble real logged data: response
lag, electrical noise, ADC quantisation and, optionally, exhaust transport
delay.

Corruption is applied to measurements only. The true volumetric efficiency
surface is never passed through this module and is deliberately absent from
the resulting log, so the calibration pipeline cannot see it.
"""

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from vetuner.drive_cycle import first_order_lag
from vetuner.engine_model import SimulationResult


@dataclass(frozen=True)
class SensorConfig:
    """Sensor imperfection settings.

    Every term can be set to zero independently, allowing individual error
    sources to be isolated during sensitivity analysis. Setting all terms to
    zero reduces the sensor model to the identity, which is used to verify the
    calibration pipeline against a noise-free reference.

    Attributes
    ----------
    afr_tau_s : float
        Wideband sensor response time constant, in seconds.
    afr_transport_delay_s : float
        Exhaust gas travel time from valve to sensor, in seconds. Zero by
        default; a speed-dependent delay is a planned extension.
    afr_noise_std : float
        Standard deviation of AFR measurement noise, in AFR units.
    afr_quantisation : float
        ADC step size for the AFR channel, in AFR units.
    afr_lean_rail : float
        Reading returned when no fuel is being injected, in AFR units.
        Represents the sensor saturating on ambient air during overrun.
    map_tau_s, map_noise_std_kpa, map_quantisation_kpa : float
        Manifold pressure sensor lag (s), noise (kPa) and ADC step (kPa).
    map_bias_kpa : float
        Systematic manifold pressure offset in kPa, representing a
        calibration error. Zero by default.
    iat_tau_s, iat_noise_std_c, iat_quantisation_c : float
        Intake air temperature sensor lag (s), noise (K) and ADC step (K).
        The lag is long because the thermistor has real thermal mass.
    rpm_noise_std, rpm_quantisation : float
        Engine speed jitter and resolution, in rpm.
    tps_noise_std, tps_quantisation : float
        Throttle position noise and ADC step, as fractions.
    batt_noise_std_v, batt_quantisation_v : float
        Battery voltage noise and ADC step, in volts.
    """

    afr_tau_s: float = 0.12
    afr_transport_delay_s: float = 0.0
    afr_noise_std: float = 0.10
    afr_quantisation: float = 0.02
    afr_lean_rail: float = 22.0

    map_tau_s: float = 0.01
    map_noise_std_kpa: float = 0.6
    map_quantisation_kpa: float = 0.11
    map_bias_kpa: float = 0.0

    iat_tau_s: float = 8.0
    iat_noise_std_c: float = 0.3
    iat_quantisation_c: float = 0.25

    rpm_noise_std: float = 3.0
    rpm_quantisation: float = 1.0

    tps_noise_std: float = 0.003
    tps_quantisation: float = 0.001

    batt_noise_std_v: float = 0.05
    batt_quantisation_v: float = 0.02


NOISE_FREE = SensorConfig(
    afr_tau_s=0.0,
    afr_transport_delay_s=0.0,
    afr_noise_std=0.0,
    afr_quantisation=0.0,
    map_tau_s=0.0,
    map_noise_std_kpa=0.0,
    map_quantisation_kpa=0.0,
    map_bias_kpa=0.0,
    iat_tau_s=0.0,
    iat_noise_std_c=0.0,
    iat_quantisation_c=0.0,
    rpm_noise_std=0.0,
    rpm_quantisation=0.0,
    tps_noise_std=0.0,
    tps_quantisation=0.0,
    batt_noise_std_v=0.0,
    batt_quantisation_v=0.0,
)
"""Sensor configuration with every imperfection disabled.

Used to verify that the calibration pipeline recovers the true VE surface
exactly when given perfect measurements.
"""


@dataclass(frozen=True)
class SensorLog:
    """Measured engine data, as it would appear in a logged file.

    Contains only quantities a real datalogger can observe. The true
    volumetric efficiency is deliberately absent.

    Attributes
    ----------
    time_s : numpy.ndarray
        Sample times in seconds.
    rpm : numpy.ndarray
        Measured engine speed in rpm.
    map_kpa : numpy.ndarray
        Measured manifold absolute pressure in kPa.
    iat_c : numpy.ndarray
        Measured intake air temperature in degrees Celsius.
    tps : numpy.ndarray
        Measured throttle position as a fraction in [0, 1].
    afr : numpy.ndarray
        Measured air-fuel ratio by mass.
    batt_v : numpy.ndarray
        Measured battery voltage in volts.
    """

    time_s: NDArray[np.float64]
    rpm: NDArray[np.float64]
    map_kpa: NDArray[np.float64]
    iat_c: NDArray[np.float64]
    tps: NDArray[np.float64]
    afr: NDArray[np.float64]
    batt_v: NDArray[np.float64]

    def __len__(self) -> int:
        """Return the number of samples."""
        return len(self.time_s)


def transport_delay(signal: ArrayLike, dt: float, delay_s: float) -> NDArray[np.float64]:
    """Delay a signal by a whole number of samples.

    The leading samples are filled with the first value, representing a sensor
    that has been exposed to the initial condition before logging began.

    Parameters
    ----------
    signal : array_like
        Input samples, uniformly spaced in time.
    dt : float
        Sample interval in seconds. Must be positive.
    delay_s : float
        Delay in seconds. Rounded to the nearest whole sample. Zero or
        negative returns the input unchanged.

    Returns
    -------
    numpy.ndarray
        Delayed signal, same shape as `signal`.

    Raises
    ------
    ValueError
        If `dt` is not positive.

    Notes
    -----
    Rounding to whole samples limits delay resolution to the logging interval.
    At 50 Hz that is 20 ms, which is coarse compared to a typical exhaust
    transport delay at high engine speed. Sub-sample delay would require
    interpolation and is not currently implemented.
    """
    x = np.asarray(signal, dtype=float)
    if dt <= 0.0:
        raise ValueError("Sample interval must be positive")
    n = int(round(delay_s / dt))
    if n <= 0:
        return x.copy()
    if n >= len(x):
        return np.full_like(x, x[0])

    out = np.empty_like(x)
    out[:n] = x[0]
    out[n:] = x[:-n]
    return out


def quantise(signal: ArrayLike, step: float) -> NDArray[np.float64]:
    """Round a signal onto a uniform grid, as an ADC does.

    Parameters
    ----------
    signal : array_like
        Input samples.
    step : float
        Quantisation interval in the units of `signal`. Zero or negative
        returns the input unchanged.

    Returns
    -------
    numpy.ndarray
        Quantised signal, same shape as `signal`.
    """
    x = np.asarray(signal, dtype=float)
    if step <= 0.0:
        return x.copy()
    return np.round(x / step) * step


def add_noise(
    signal: ArrayLike, std: float, rng: np.random.Generator
) -> NDArray[np.float64]:
    """Add zero-mean Gaussian noise to a signal.

    Parameters
    ----------
    signal : array_like
        Input samples.
    std : float
        Noise standard deviation in the units of `signal`. Zero or negative
        returns the input unchanged.
    rng : numpy.random.Generator
        Seeded random number generator, for reproducibility.

    Returns
    -------
    numpy.ndarray
        Noisy signal, same shape as `signal`.
    """
    x = np.asarray(signal, dtype=float)
    if std <= 0.0:
        return x.copy()
    return x + rng.normal(0.0, std, size=x.shape)


def apply_sensor_model(
    result: SimulationResult,
    config: SensorConfig = SensorConfig(),
    seed: int = 0,
) -> SensorLog:
    """Convert a forward simulation into a realistic measured log.

    Each channel is corrupted in physical order: transport delay first (where
    applicable), then sensor response lag, then electrical noise, then ADC
    quantisation. Applying these in a different order would represent a
    different physical arrangement.

    Parameters
    ----------
    result : SimulationResult
        Output of the forward engine model.
    config : SensorConfig, optional
        Sensor imperfection settings, by default a realistic configuration.
    seed : int, optional
        Random seed, by default 0. Fixing it makes logs reproducible.

    Returns
    -------
    SensorLog
        Measured channels only. Contains no reference to the true VE surface.

    Raises
    ------
    ValueError
        If the simulation contains fewer than two samples, so no sample
        interval can be determined.

    Notes
    -----
    Air-fuel ratio is undefined during overrun fuel cut. A real wideband
    sensor saturates on ambient air in that condition, so those samples are
    set to `config.afr_lean_rail` before lag is applied. This keeps the log
    free of missing values and leaves the pipeline responsible for
    identifying and rejecting overrun, as it would be with real data.
    """
    if len(result) < 2:
        raise ValueError("Need at least two samples to determine a sample interval")

    dt = float(result.time_s[1] - result.time_s[0])
    rng = np.random.default_rng(seed)

    afr_physical = np.where(result.fuel_cut, config.afr_lean_rail, result.afr_actual)
    afr = transport_delay(afr_physical, dt, config.afr_transport_delay_s)
    afr = first_order_lag(afr, dt, config.afr_tau_s)
    afr = add_noise(afr, config.afr_noise_std, rng)
    afr = quantise(np.minimum(afr, config.afr_lean_rail), config.afr_quantisation)

    map_kpa = first_order_lag(result.map_kpa + config.map_bias_kpa, dt, config.map_tau_s)
    map_kpa = add_noise(map_kpa, config.map_noise_std_kpa, rng)
    map_kpa = quantise(np.maximum(map_kpa, 0.0), config.map_quantisation_kpa)

    iat_c = first_order_lag(result.iat_c, dt, config.iat_tau_s)
    iat_c = quantise(
        add_noise(iat_c, config.iat_noise_std_c, rng), config.iat_quantisation_c
    )

    rpm = quantise(
        np.maximum(add_noise(result.rpm, config.rpm_noise_std, rng), 0.0),
        config.rpm_quantisation,
    )

    tps = quantise(
        np.clip(add_noise(result.tps, config.tps_noise_std, rng), 0.0, 1.0),
        config.tps_quantisation,
    )

    batt_v = quantise(
        add_noise(result.batt_v, config.batt_noise_std_v, rng),
        config.batt_quantisation_v,
    )

    return SensorLog(
        time_s=result.time_s,
        rpm=rpm,
        map_kpa=map_kpa,
        iat_c=iat_c,
        tps=tps,
        afr=afr,
        batt_v=batt_v,
    )