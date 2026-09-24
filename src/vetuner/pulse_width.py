"""Injector pulse width output.

Converts a calibrated volumetric efficiency table into the injector pulse
widths it commands, and checks that those pulse widths are physically
deliverable.

An engine control unit does not store pulse widths. It stores volumetric
efficiency and computes pulse width at run time from the current manifold
pressure, intake air temperature and supply voltage. The maps produced here
are therefore evaluated at stated reference conditions, and serve to verify a
calibration and to expose injector sizing limits rather than to be programmed
into an ECU directly.
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import ArrayLike, NDArray

from vetuner.engine_model import (
    EngineGeometry,
    Injector,
    air_mass_per_cycle,
    cycle_period_ms,
    injector_deadtime_ms,
    pulse_width_from_fuel_mass,
    target_afr,
)
from vetuner.ve_surface import DEFAULT_MAP_AXIS, DEFAULT_RPM_AXIS

REFERENCE_IAT_C = 25.0
"""Intake air temperature at which pulse width maps are evaluated, in Celsius."""

REFERENCE_BATT_V = 14.0
"""Supply voltage at which pulse width maps are evaluated, in volts."""


@dataclass(frozen=True)
class PulseWidthMap:
    """Injector pulse widths commanded by a VE table at reference conditions.

    Attributes
    ----------
    rpm_axis, map_axis : numpy.ndarray
        Table breakpoints, in rpm and kPa absolute.
    pulse_width_ms : numpy.ndarray
        Commanded pulse width at each operating point, in milliseconds,
        including injector opening delay.
    duty : numpy.ndarray
        Pulse width as a fraction of the time available per engine cycle.
    air_mass_kg : numpy.ndarray
        Air ingested per cylinder per cycle, in kg.
    fuel_mass_kg : numpy.ndarray
        Fuel required per cylinder per cycle, in kg.
    afr_target : numpy.ndarray
        Commanded air-fuel ratio at each operating point.
    feasible : numpy.ndarray
        Boolean mask, True where the pulse width fits within the injector's
        usable duty limit.
    iat_c, batt_v : float
        Reference conditions at which the map was evaluated.

    Notes
    -----
    Pulse width scales with air density, so a map evaluated at one intake air
    temperature does not apply at another. A change from 25 to 45 degrees
    Celsius reduces density by about six percent and pulse width with it.
    """

    rpm_axis: NDArray[np.float64]
    map_axis: NDArray[np.float64]
    pulse_width_ms: NDArray[np.float64]
    duty: NDArray[np.float64]
    air_mass_kg: NDArray[np.float64]
    fuel_mass_kg: NDArray[np.float64]
    afr_target: NDArray[np.float64]
    feasible: NDArray[np.bool_]
    iat_c: float
    batt_v: float

    @property
    def max_duty(self) -> float:
        """Highest injector duty anywhere in the map."""
        return float(self.duty.max())

    @property
    def worst_cell(self) -> tuple[float, float]:
        """Engine speed and manifold pressure of the highest duty cell."""
        i, j = np.unravel_index(int(self.duty.argmax()), self.duty.shape)
        return float(self.rpm_axis[i]), float(self.map_axis[j])


def pulse_width_map(
    table: ArrayLike,
    rpm_axis: ArrayLike = DEFAULT_RPM_AXIS,
    map_axis: ArrayLike = DEFAULT_MAP_AXIS,
    engine: EngineGeometry = EngineGeometry(),
    injector: Injector = Injector(),
    iat_c: float = REFERENCE_IAT_C,
    batt_v: float = REFERENCE_BATT_V,
) -> PulseWidthMap:
    """Compute the pulse widths a VE table commands across the operating range.

    Parameters
    ----------
    table : array_like
        VE table as fractions, shaped to the axes.
    rpm_axis, map_axis : array_like, optional
        Table breakpoints.
    engine : EngineGeometry, optional
        Engine geometry.
    injector : Injector, optional
        Injector characteristics.
    iat_c : float, optional
        Reference intake air temperature in Celsius, by default 25.0.
    batt_v : float, optional
        Reference supply voltage in volts, by default 14.0.

    Returns
    -------
    PulseWidthMap
        Pulse widths, duty cycles and feasibility at every table node.

    Raises
    ------
    ValueError
        If `table` does not match the axis lengths, or contains
        non-positive volumetric efficiencies.
    """
    ve = np.asarray(table, dtype=float)
    rpm_ax = np.asarray(rpm_axis, dtype=float)
    map_ax = np.asarray(map_axis, dtype=float)

    if ve.shape != (len(rpm_ax), len(map_ax)):
        raise ValueError(
            f"Table shape {ve.shape} does not match axes ({len(rpm_ax)}, {len(map_ax)})"
        )
    if np.any(ve <= 0.0):
        raise ValueError("Volumetric efficiency must be positive")

    rpm_grid, map_grid = np.meshgrid(rpm_ax, map_ax, indexing="ij")

    air = air_mass_per_cycle(ve, map_grid, iat_c, engine)
    afr = target_afr(map_grid)
    fuel = air / afr

    pulse = pulse_width_from_fuel_mass(fuel, batt_v, injector)
    period = cycle_period_ms(rpm_grid, engine)
    duty = pulse / period

    return PulseWidthMap(
        rpm_axis=rpm_ax,
        map_axis=map_ax,
        pulse_width_ms=pulse,
        duty=duty,
        air_mass_kg=air,
        fuel_mass_kg=fuel,
        afr_target=afr,
        feasible=duty <= injector.max_duty_fraction,
        iat_c=iat_c,
        batt_v=batt_v,
    )


def required_injector_flow(
    table: ArrayLike,
    rpm_axis: ArrayLike = DEFAULT_RPM_AXIS,
    map_axis: ArrayLike = DEFAULT_MAP_AXIS,
    engine: EngineGeometry = EngineGeometry(),
    injector: Injector = Injector(),
    target_duty: float = 0.80,
    iat_c: float = REFERENCE_IAT_C,
    batt_v: float = REFERENCE_BATT_V,
) -> float:
    """Compute the injector flow rate needed to stay within a duty limit.

    Finds the operating point demanding the most fuel per unit time and returns
    the static flow rate at which that point would sit at `target_duty`.
    Comparing this against the fitted injector's rating shows how much sizing
    margin the calibration has.

    Parameters
    ----------
    table : array_like
        VE table as fractions.
    rpm_axis, map_axis : array_like, optional
        Table breakpoints.
    engine : EngineGeometry, optional
        Engine geometry.
    injector : Injector, optional
        Injector characteristics, used for its opening delay.
    target_duty : float, optional
        Duty fraction the worst cell should reach, by default 0.80.
    iat_c, batt_v : float, optional
        Reference conditions.

    Returns
    -------
    float
        Required static flow rate in cc per minute.

    Raises
    ------
    ValueError
        If `target_duty` lies outside (0, 1], or the opening delay alone
        exceeds the available time at any operating point.
    """
    if not 0.0 < target_duty <= 1.0:
        raise ValueError("Target duty must lie in (0, 1]")

    result = pulse_width_map(
        table, rpm_axis, map_axis, engine, injector, iat_c, batt_v
    )

    rpm_grid, _ = np.meshgrid(result.rpm_axis, result.map_axis, indexing="ij")
    period = cycle_period_ms(rpm_grid, engine)
    available_ms = target_duty * period - injector_deadtime_ms(batt_v, injector)

    if np.any(available_ms <= 0.0):
        raise ValueError(
            "Injector opening delay exceeds the available time at some operating point"
        )

    flow_kg_s = 1e3 * result.fuel_mass_kg / available_ms
    worst_kg_s = float(flow_kg_s.max())
    return worst_kg_s * 60.0 / injector.fuel_density_kg_m3 * 1e6


def pulse_width_error(
    estimated_table: ArrayLike,
    true_table: ArrayLike,
    rpm_axis: ArrayLike = DEFAULT_RPM_AXIS,
    map_axis: ArrayLike = DEFAULT_MAP_AXIS,
    **kwargs: object,
) -> dict[str, float]:
    """Compare the pulse widths commanded by an estimated and a true table.

    Expresses calibration accuracy in the units the injector actually receives,
    rather than in volumetric efficiency points. A pulse width error is what
    determines the mixture the engine runs, so it is the quantity with direct
    physical consequence.

    Parameters
    ----------
    estimated_table : array_like
        Calibrated VE table, as fractions.
    true_table : array_like
        Reference VE table, as fractions.
    rpm_axis, map_axis : array_like, optional
        Table breakpoints.
    **kwargs
        Forwarded to `pulse_width_map`.

    Returns
    -------
    dict
        Mean and maximum absolute error in milliseconds and as a percentage,
        and the proportion of cells within one percent.
    """
    estimated = pulse_width_map(estimated_table, rpm_axis, map_axis, **kwargs)
    reference = pulse_width_map(true_table, rpm_axis, map_axis, **kwargs)

    absolute = np.abs(estimated.pulse_width_ms - reference.pulse_width_ms)
    relative = 100.0 * absolute / reference.pulse_width_ms

    return {
        "mean_abs_ms": float(absolute.mean()),
        "max_abs_ms": float(absolute.max()),
        "mean_abs_pct": float(relative.mean()),
        "max_abs_pct": float(relative.max()),
        "fraction_within_1pct": float((relative < 1.0).mean()),
    }


def export_table_csv(
    path: str | Path,
    table: ArrayLike,
    rpm_axis: ArrayLike = DEFAULT_RPM_AXIS,
    map_axis: ArrayLike = DEFAULT_MAP_AXIS,
    label: str = "VE %",
    scale: float = 100.0,
    decimals: int = 2,
) -> Path:
    """Write a table to CSV with engine speed as rows and load as columns.

    The layout matches the convention used by engine calibration software, so
    the file can be inspected or imported directly rather than reshaped first.

    Parameters
    ----------
    path : str or pathlib.Path
        Destination file. Parent directories are created if absent.
    table : array_like
        Values shaped to the axes.
    rpm_axis, map_axis : array_like, optional
        Table breakpoints.
    label : str, optional
        Text placed in the top left corner, by default ``"VE %"``.
    scale : float, optional
        Factor applied to the values before writing, by default 100.0 to
        convert VE fractions to percentages.
    decimals : int, optional
        Decimal places to write, by default 2.

    Returns
    -------
    pathlib.Path
        The path written.

    Raises
    ------
    ValueError
        If `table` does not match the axis lengths.
    """
    values = np.asarray(table, dtype=float) * scale
    rpm_ax = np.asarray(rpm_axis, dtype=float)
    map_ax = np.asarray(map_axis, dtype=float)

    if values.shape != (len(rpm_ax), len(map_ax)):
        raise ValueError("Table shape does not match the axes")

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)

    lines = [label + "," + ",".join(f"{m:.0f}" for m in map_ax)]
    for i, rpm in enumerate(rpm_ax):
        row = ",".join(f"{v:.{decimals}f}" for v in values[i])
        lines.append(f"{rpm:.0f},{row}")

    destination.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return destination