"""Forward engine model.

Simulates what a fuel-injected engine actually does when its ECU runs a given
volumetric efficiency table. The true VE surface determines the air ingested;
the ECU's table determines the fuel injected; the mismatch between them
appears as an air-fuel ratio error.

This module generates the synthetic data used to validate the calibration
pipeline. The pipeline itself must never import the true surface.
"""

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.interpolate import RegularGridInterpolator

from vetuner.drive_cycle import DriveCycle
from vetuner.ve_surface import (
    DEFAULT_MAP_AXIS,
    DEFAULT_RPM_AXIS,
    TRUE_SURFACE,
    VESurfaceParams,
    volumetric_efficiency,
)

R_SPECIFIC_AIR = 287.05
"""Specific gas constant for dry air, in J/(kg*K)."""

CELSIUS_TO_KELVIN = 273.15
"""Offset between the Celsius and Kelvin scales."""


@dataclass(frozen=True)
class EngineGeometry:
    """Fixed geometric properties of the simulated engine.

    Attributes
    ----------
    displacement_l : float
        Total swept volume in litres.
    n_cylinders : int
        Number of cylinders.
    revolutions_per_cycle : float
        Crankshaft revolutions per complete engine cycle. Two for a
        four-stroke engine, one for a two-stroke.
    """

    displacement_l: float = 2.0
    n_cylinders: int = 4
    revolutions_per_cycle: float = 2.0

    @property
    def cylinder_volume_m3(self) -> float:
        """Swept volume of a single cylinder, in cubic metres."""
        return self.displacement_l * 1e-3 / self.n_cylinders


@dataclass(frozen=True)
class Injector:
    """Fuel injector characteristics.

    Attributes
    ----------
    flow_cc_min : float
        Static flow rate at rated fuel pressure, in cc per minute.
    fuel_density_kg_m3 : float
        Density of the fuel, in kg/m^3. About 745 for petrol.
    deadtime_ms_at_14v : float
        Opening delay at 14 volts, in milliseconds.
    deadtime_slope_ms_per_v : float
        Increase in opening delay per volt below 14, in ms/V.
    max_duty_fraction : float
        Highest usable fraction of the available cycle time. Above this the
        injector cannot close fully between pulses.
    """

    flow_cc_min: float = 440.0
    fuel_density_kg_m3: float = 745.0
    deadtime_ms_at_14v: float = 0.90
    deadtime_slope_ms_per_v: float = 0.20
    max_duty_fraction: float = 0.85

    @property
    def flow_kg_s(self) -> float:
        """Static flow rate in kg/s."""
        return self.flow_cc_min * 1e-6 * self.fuel_density_kg_m3 / 60.0


@dataclass(frozen=True)
class SimulationResult:
    """Output of a forward engine simulation.

    All arrays share the length of the input drive cycle.

    Attributes
    ----------
    time_s, rpm, tps, map_kpa, iat_c, batt_v : numpy.ndarray
        Operating conditions, carried through from the drive cycle.
    ve_true : numpy.ndarray
        Volumetric efficiency actually achieved, as a fraction. Hidden from
        the calibration pipeline.
    ve_commanded : numpy.ndarray
        Volumetric efficiency the ECU read from its table, as a fraction.
    air_mass_kg : numpy.ndarray
        Air ingested per cylinder per cycle, in kg.
    afr_target : numpy.ndarray
        Air-fuel ratio the ECU aimed for, by mass.
    fuel_mass_kg : numpy.ndarray
        Fuel delivered per cylinder per cycle, in kg.
    pulse_width_ms : numpy.ndarray
        Commanded injector pulse width, in milliseconds.
    injector_duty : numpy.ndarray
        Pulse width as a fraction of the available cycle time.
    afr_actual : numpy.ndarray
        Air-fuel ratio actually produced, by mass. NaN where fuel is cut.
    fuel_cut : numpy.ndarray
        Boolean flag marking overrun samples where no fuel was injected.
    """

    time_s: NDArray[np.float64]
    rpm: NDArray[np.float64]
    tps: NDArray[np.float64]
    map_kpa: NDArray[np.float64]
    iat_c: NDArray[np.float64]
    batt_v: NDArray[np.float64]
    ve_true: NDArray[np.float64]
    ve_commanded: NDArray[np.float64]
    air_mass_kg: NDArray[np.float64]
    afr_target: NDArray[np.float64]
    fuel_mass_kg: NDArray[np.float64]
    pulse_width_ms: NDArray[np.float64]
    injector_duty: NDArray[np.float64]
    afr_actual: NDArray[np.float64]
    fuel_cut: NDArray[np.bool_]

    def __len__(self) -> int:
        """Return the number of samples."""
        return len(self.time_s)


def air_density(map_kpa: ArrayLike, iat_c: ArrayLike) -> NDArray[np.float64]:
    """Compute manifold air density from the ideal gas law.

    Parameters
    ----------
    map_kpa : array_like
        Manifold absolute pressure in kPa. Must be positive.
    iat_c : array_like
        Intake air temperature in degrees Celsius. Must be above absolute zero.

    Returns
    -------
    numpy.ndarray
        Air density in kg/m^3.

    Raises
    ------
    ValueError
        If any pressure is non-positive or any temperature is at or below
        absolute zero.
    """
    map_arr = np.asarray(map_kpa, dtype=float)
    iat_arr = np.asarray(iat_c, dtype=float)

    if np.any(map_arr <= 0.0):
        raise ValueError("Manifold pressure must be positive")
    if np.any(iat_arr <= -CELSIUS_TO_KELVIN):
        raise ValueError("Intake air temperature must be above absolute zero")

    return (map_arr * 1e3) / (R_SPECIFIC_AIR * (iat_arr + CELSIUS_TO_KELVIN))


def air_mass_per_cycle(
    ve: ArrayLike,
    map_kpa: ArrayLike,
    iat_c: ArrayLike,
    engine: EngineGeometry,
) -> NDArray[np.float64]:
    """Compute air mass trapped per cylinder per engine cycle.

    Applies the speed-density relation: the cylinder swallows its swept volume
    scaled by volumetric efficiency, filled with air at manifold density.

    Parameters
    ----------
    ve : array_like
        Volumetric efficiency as a fraction.
    map_kpa : array_like
        Manifold absolute pressure in kPa.
    iat_c : array_like
        Intake air temperature in degrees Celsius.
    engine : EngineGeometry
        Engine geometry.

    Returns
    -------
    numpy.ndarray
        Air mass per cylinder per cycle, in kg.
    """
    return np.asarray(ve, dtype=float) * engine.cylinder_volume_m3 * air_density(
        map_kpa, iat_c
    )


def target_afr(
    map_kpa: ArrayLike,
    cruise_afr: float = 14.7,
    power_afr: float = 12.8,
    enrich_start_kpa: float = 65.0,
    enrich_full_kpa: float = 95.0,
) -> NDArray[np.float64]:
    """Compute the ECU's commanded air-fuel ratio target.

    Runs stoichiometric at light load for efficiency and emissions, then
    enriches progressively towards full load to control combustion
    temperature and detonation margin.

    Parameters
    ----------
    map_kpa : array_like
        Manifold absolute pressure in kPa.
    cruise_afr : float, optional
        Target below `enrich_start_kpa`, by default 14.7.
    power_afr : float, optional
        Target at and above `enrich_full_kpa`, by default 12.8.
    enrich_start_kpa, enrich_full_kpa : float, optional
        Pressures bounding the enrichment ramp, in kPa.

    Returns
    -------
    numpy.ndarray
        Target air-fuel ratio by mass.
    """
    map_arr = np.asarray(map_kpa, dtype=float)
    fraction = np.clip(
        (map_arr - enrich_start_kpa) / (enrich_full_kpa - enrich_start_kpa), 0.0, 1.0
    )
    return cruise_afr - (cruise_afr - power_afr) * fraction


def read_ve_table(
    table: ArrayLike,
    rpm: ArrayLike,
    map_kpa: ArrayLike,
    rpm_axis: ArrayLike = DEFAULT_RPM_AXIS,
    map_axis: ArrayLike = DEFAULT_MAP_AXIS,
) -> NDArray[np.float64]:
    """Read a VE table by bilinear interpolation, as an ECU would.

    Operating points outside the table axes are clamped to the nearest
    breakpoint rather than extrapolated, matching typical ECU behaviour.

    Parameters
    ----------
    table : array_like
        VE fractions with shape ``(len(rpm_axis), len(map_axis))``.
    rpm : array_like
        Engine speed in rpm.
    map_kpa : array_like
        Manifold absolute pressure in kPa.
    rpm_axis, map_axis : array_like, optional
        Table breakpoints.

    Returns
    -------
    numpy.ndarray
        Interpolated VE fractions, broadcast to the shape of the inputs.

    Raises
    ------
    ValueError
        If `table` does not match the axis lengths.

    Notes
    -----
    Bilinear interpolation is a linear operator mapping table nodes to
    operating points. Its transpose distributes a correction at an operating
    point back onto the four surrounding nodes, which is the basis of the
    bilinear scatter surface fitting method.
    """
    table_arr = np.asarray(table, dtype=float)
    rpm_ax = np.asarray(rpm_axis, dtype=float)
    map_ax = np.asarray(map_axis, dtype=float)

    if table_arr.shape != (len(rpm_ax), len(map_ax)):
        raise ValueError(
            f"Table shape {table_arr.shape} does not match axes "
            f"({len(rpm_ax)}, {len(map_ax)})"
        )

    interpolator = RegularGridInterpolator(
        (rpm_ax, map_ax), table_arr, method="linear", bounds_error=False, fill_value=None
    )

    rpm_b, map_b = np.broadcast_arrays(
        np.asarray(rpm, dtype=float), np.asarray(map_kpa, dtype=float)
    )
    points = np.column_stack(
        [
            np.clip(rpm_b.ravel(), rpm_ax[0], rpm_ax[-1]),
            np.clip(map_b.ravel(), map_ax[0], map_ax[-1]),
        ]
    )
    return interpolator(points).reshape(rpm_b.shape)


def injector_deadtime_ms(batt_v: ArrayLike, injector: Injector) -> NDArray[np.float64]:
    """Compute injector opening delay as a function of supply voltage.

    Parameters
    ----------
    batt_v : array_like
        Supply voltage in volts.
    injector : Injector
        Injector characteristics.

    Returns
    -------
    numpy.ndarray
        Opening delay in milliseconds.
    """
    volts = np.asarray(batt_v, dtype=float)
    return injector.deadtime_ms_at_14v + injector.deadtime_slope_ms_per_v * (14.0 - volts)


def pulse_width_from_fuel_mass(
    fuel_mass_kg: ArrayLike,
    batt_v: ArrayLike,
    injector: Injector,
) -> NDArray[np.float64]:
    """Convert a required fuel mass into a commanded pulse width.

    Parameters
    ----------
    fuel_mass_kg : array_like
        Fuel mass per injection event, in kg. Must be non-negative.
    batt_v : array_like
        Supply voltage in volts.
    injector : Injector
        Injector characteristics.

    Returns
    -------
    numpy.ndarray
        Commanded pulse width in milliseconds, including opening delay.

    Raises
    ------
    ValueError
        If any fuel mass is negative.
    """
    mass = np.asarray(fuel_mass_kg, dtype=float)
    if np.any(mass < 0.0):
        raise ValueError("Fuel mass must be non-negative")
    open_ms = 1e3 * mass / injector.flow_kg_s
    return open_ms + injector_deadtime_ms(batt_v, injector)


def fuel_mass_from_pulse_width(
    pulse_width_ms: ArrayLike,
    batt_v: ArrayLike,
    injector: Injector,
) -> NDArray[np.float64]:
    """Convert a commanded pulse width back into delivered fuel mass.

    The exact inverse of `pulse_width_from_fuel_mass`. Pulse widths shorter
    than the opening delay deliver no fuel.

    Parameters
    ----------
    pulse_width_ms : array_like
        Commanded pulse width in milliseconds.
    batt_v : array_like
        Supply voltage in volts.
    injector : Injector
        Injector characteristics.

    Returns
    -------
    numpy.ndarray
        Delivered fuel mass per injection event, in kg.
    """
    pw = np.asarray(pulse_width_ms, dtype=float)
    open_ms = np.maximum(pw - injector_deadtime_ms(batt_v, injector), 0.0)
    return open_ms * 1e-3 * injector.flow_kg_s


def cycle_period_ms(rpm: ArrayLike, engine: EngineGeometry) -> NDArray[np.float64]:
    """Compute the time available for one injection event.

    Parameters
    ----------
    rpm : array_like
        Engine speed in rpm. Must be positive.
    engine : EngineGeometry
        Engine geometry.

    Returns
    -------
    numpy.ndarray
        Duration of one complete engine cycle, in milliseconds.

    Raises
    ------
    ValueError
        If any engine speed is non-positive.
    """
    rpm_arr = np.asarray(rpm, dtype=float)
    if np.any(rpm_arr <= 0.0):
        raise ValueError("Engine speed must be positive to define a cycle period")
    return 60.0 * 1e3 * engine.revolutions_per_cycle / rpm_arr


def simulate_engine(
    cycle: DriveCycle,
    ecu_table: ArrayLike,
    engine: EngineGeometry = EngineGeometry(),
    injector: Injector = Injector(),
    true_params: VESurfaceParams = TRUE_SURFACE,
    rpm_axis: ArrayLike = DEFAULT_RPM_AXIS,
    map_axis: ArrayLike = DEFAULT_MAP_AXIS,
    fuel_cut_tps: float = 0.02,
    fuel_cut_rpm: float = 1500.0,
) -> SimulationResult:
    """Run the forward engine model over a drive cycle.

    The engine ingests air according to `true_params`; the ECU meters fuel
    according to `ecu_table`. Where the two disagree, the resulting air-fuel
    ratio departs from target in direct proportion.

    Parameters
    ----------
    cycle : DriveCycle
        Operating conditions over time.
    ecu_table : array_like
        VE table the simulated ECU is running, shaped to the axes.
    engine : EngineGeometry, optional
        Engine geometry.
    injector : Injector, optional
        Injector characteristics.
    true_params : VESurfaceParams, optional
        Parameters of the hidden true VE surface.
    rpm_axis, map_axis : array_like, optional
        Table breakpoints.
    fuel_cut_tps : float, optional
        Throttle position below which overrun fuel cut engages, by default 0.02.
    fuel_cut_rpm : float, optional
        Engine speed above which overrun fuel cut engages, by default 1500.

    Returns
    -------
    SimulationResult
        Simulated engine behaviour, including the hidden true VE.

    Notes
    -----
    Fuel film dynamics, exhaust transport delay and cylinder-to-cylinder
    variation are not modelled. Transport delay is handled separately in the
    sensor model; wall wetting is deliberately excluded and its effect is
    managed by steady-state gating in the calibration pipeline.
    """
    ve_true = volumetric_efficiency(cycle.rpm, cycle.map_kpa, true_params)
    ve_commanded = read_ve_table(ecu_table, cycle.rpm, cycle.map_kpa, rpm_axis, map_axis)

    air_mass = air_mass_per_cycle(ve_true, cycle.map_kpa, cycle.iat_c, engine)
    believed_air = air_mass_per_cycle(ve_commanded, cycle.map_kpa, cycle.iat_c, engine)

    afr_tgt = target_afr(cycle.map_kpa)
    fuel_cut = (cycle.tps < fuel_cut_tps) & (cycle.rpm > fuel_cut_rpm)

    requested_fuel = np.where(fuel_cut, 0.0, believed_air / afr_tgt)
    pulse_width = np.where(
        fuel_cut, 0.0, pulse_width_from_fuel_mass(requested_fuel, cycle.batt_v, injector)
    )

    period = cycle_period_ms(cycle.rpm, engine)
    max_pw = injector.max_duty_fraction * period
    pulse_width = np.minimum(pulse_width, max_pw)

    fuel_mass = np.where(
        fuel_cut, 0.0, fuel_mass_from_pulse_width(pulse_width, cycle.batt_v, injector)
    )

    with np.errstate(divide="ignore", invalid="ignore"):
        afr_actual = np.where(fuel_cut, np.nan, air_mass / fuel_mass)

    return SimulationResult(
        time_s=cycle.time_s,
        rpm=cycle.rpm,
        tps=cycle.tps,
        map_kpa=cycle.map_kpa,
        iat_c=cycle.iat_c,
        batt_v=cycle.batt_v,
        ve_true=ve_true,
        ve_commanded=ve_commanded,
        air_mass_kg=air_mass,
        afr_target=afr_tgt,
        fuel_mass_kg=fuel_mass,
        pulse_width_ms=pulse_width,
        injector_duty=pulse_width / period,
        afr_actual=afr_actual,
        fuel_cut=fuel_cut,
    )