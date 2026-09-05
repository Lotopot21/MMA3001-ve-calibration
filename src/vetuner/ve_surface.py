"""Analytic volumetric efficiency surfaces.

This module defines the ground-truth VE surface used to generate synthetic
logs, and a deliberately miscalibrated base map representing a realistic
starting calibration. The ground truth is never exposed to the calibration
pipeline; it is used only to score the pipeline's output.
"""

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray


@dataclass(frozen=True)
class VESurfaceParams:
    """Parameters of an analytic volumetric efficiency surface.

    The surface is the product of a speed term and a load term::

        VE(N, p) = f(N) * g(p)

    where ``f`` is a baseline plus a Gaussian resonance peak minus a Gaussian
    resonance dip, and ``g`` is a decaying deficit that reduces VE at low
    manifold pressure to represent increased residual gas fraction.

    Attributes
    ----------
    base : float
        Baseline VE far from the resonance peak, as a fraction.
    peak_gain : float
        Height of the resonance peak added to `base`, as a fraction.
    peak_rpm : float
        Engine speed at which the resonance peak is centred, in rpm.
    peak_width_rpm : float
        Standard deviation of the resonance peak, in rpm.
    dip_gain : float
        Depth of the secondary resonance dip, as a fraction.
    dip_rpm : float
        Engine speed at which the dip is centred, in rpm.
    dip_width_rpm : float
        Standard deviation of the dip, in rpm.
    load_deficit : float
        Maximum VE reduction at the reference manifold pressure, as a fraction.
    load_scale_kpa : float
        Decay length of the load deficit, in kPa.
    map_ref_kpa : float
        Manifold pressure at which the full load deficit applies, in kPa.
    ve_min, ve_max : float
        Clamping bounds applied to the returned surface, as fractions.
    """

    base: float = 0.62
    peak_gain: float = 0.30
    peak_rpm: float = 4200.0
    peak_width_rpm: float = 1600.0
    dip_gain: float = 0.05
    dip_rpm: float = 2600.0
    dip_width_rpm: float = 500.0
    load_deficit: float = 0.18
    load_scale_kpa: float = 35.0
    map_ref_kpa: float = 20.0
    ve_min: float = 0.20
    ve_max: float = 1.15


TRUE_SURFACE = VESurfaceParams()
"""Hidden ground-truth surface. The pipeline must never read this directly."""

BASE_MAP_SURFACE = VESurfaceParams(
    base=0.60,
    peak_gain=0.20,
    peak_rpm=3400.0,
    peak_width_rpm=2000.0,
    dip_gain=0.0,
    load_deficit=0.10,
)
"""Miscalibrated starting map.

Represents a base calibration carried over from a similar but not identical
engine: the resonance peak sits at the wrong speed, is too shallow, and the
load dependence is understated. The resulting error varies smoothly across
the operating range rather than randomly per cell, which is what a real
miscalibration looks like.
"""

DEFAULT_RPM_AXIS = np.linspace(800.0, 7000.0, 16)
"""Default table speed axis, in rpm."""

DEFAULT_MAP_AXIS = np.linspace(20.0, 100.0, 12)
"""Default table load axis, in kPa absolute."""


def volumetric_efficiency(
    rpm: ArrayLike,
    map_kpa: ArrayLike,
    params: VESurfaceParams = TRUE_SURFACE,
) -> NDArray[np.float64]:
    """Evaluate a VE surface at arbitrary operating points.

    Parameters
    ----------
    rpm : array_like
        Engine speed in rpm. Must be non-negative.
    map_kpa : array_like
        Manifold absolute pressure in kPa. Must be positive.
    params : VESurfaceParams, optional
        Surface parameters, by default the ground-truth surface.

    Returns
    -------
    numpy.ndarray
        Volumetric efficiency as a fraction, clamped to
        ``[params.ve_min, params.ve_max]``. Shape follows NumPy broadcasting
        of `rpm` and `map_kpa`.

    Raises
    ------
    ValueError
        If any engine speed is negative or any manifold pressure is
        non-positive.

    Examples
    --------
    >>> float(volumetric_efficiency(4200.0, 100.0))  # doctest: +ELLIPSIS
    0.90...
    """
    rpm_arr = np.asarray(rpm, dtype=float)
    map_arr = np.asarray(map_kpa, dtype=float)

    if np.any(rpm_arr < 0.0):
        raise ValueError("Engine speed must be non-negative")
    if np.any(map_arr <= 0.0):
        raise ValueError("Manifold pressure must be positive")

    peak = params.peak_gain * np.exp(
        -0.5 * ((rpm_arr - params.peak_rpm) / params.peak_width_rpm) ** 2
    )
    dip = params.dip_gain * np.exp(
        -0.5 * ((rpm_arr - params.dip_rpm) / params.dip_width_rpm) ** 2
    )
    speed_term = params.base + peak - dip

    load_term = 1.0 - params.load_deficit * np.exp(
        -(map_arr - params.map_ref_kpa) / params.load_scale_kpa
    )

    return np.clip(speed_term * load_term, params.ve_min, params.ve_max)


def ve_table(
    rpm_axis: ArrayLike = DEFAULT_RPM_AXIS,
    map_axis: ArrayLike = DEFAULT_MAP_AXIS,
    params: VESurfaceParams = TRUE_SURFACE,
) -> NDArray[np.float64]:
    """Sample a VE surface onto a regular table grid.

    Parameters
    ----------
    rpm_axis : array_like, optional
        Table speed breakpoints in rpm, by default `DEFAULT_RPM_AXIS`.
    map_axis : array_like, optional
        Table load breakpoints in kPa, by default `DEFAULT_MAP_AXIS`.
    params : VESurfaceParams, optional
        Surface parameters, by default the ground-truth surface.

    Returns
    -------
    numpy.ndarray
        VE fractions with shape ``(len(rpm_axis), len(map_axis))``, indexed
        as ``table[speed_index, load_index]``.
    """
    rpm_grid, map_grid = np.meshgrid(
        np.asarray(rpm_axis, dtype=float),
        np.asarray(map_axis, dtype=float),
        indexing="ij",
    )
    return volumetric_efficiency(rpm_grid, map_grid, params)