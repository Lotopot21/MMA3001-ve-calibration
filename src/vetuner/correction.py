"""VE table correction from logged air-fuel ratio data.

The forward model gives an exact relation between fuelling error and table
error::

    AFR_measured / AFR_target = VE_true / VE_table

so the ratio of measured to target air-fuel ratio is precisely the factor by
which the table under-reads. Multiplying the table by that factor recovers the
true surface, exactly, given perfect measurements and full coverage.

Real logs satisfy neither condition, which is what the surrounding filtering,
gating and surface fitting exist to address.
"""

from dataclasses import dataclass, replace

import numpy as np
from numpy.typing import ArrayLike, NDArray

from vetuner.engine_model import target_afr
from vetuner.gating import GatingConfig, GatingResult, gate_samples
from vetuner.sensors import SensorLog
from vetuner.surface_fit import per_cell_average
from vetuner.ve_surface import DEFAULT_MAP_AXIS, DEFAULT_RPM_AXIS


@dataclass(frozen=True)
class CorrectionResult:
    """Outcome of one correction pass.

    Attributes
    ----------
    table : numpy.ndarray
        The corrected VE table, as fractions.
    factors : numpy.ndarray
        Correction factor applied to each cell. One means unchanged.
    counts : numpy.ndarray
        Number of log samples contributing to each cell.
    n_samples_used : int
        Total samples that reached the surface fit.
    """

    table: NDArray[np.float64]
    factors: NDArray[np.float64]
    counts: NDArray[np.int64]
    n_samples_used: int


def sample_correction_factors(log: SensorLog) -> NDArray[np.float64]:
    """Compute the per-sample correction factor from a log.

    Parameters
    ----------
    log : SensorLog
        Measured engine data.

    Returns
    -------
    numpy.ndarray
        Ratio of measured to target air-fuel ratio, one per sample. Values
        above one indicate the engine ran lean, meaning the table under-read
        the air actually ingested.

    Notes
    -----
    The target air-fuel ratio is a calibration setting rather than a
    measurement, so it is legitimately available to the calibration pipeline.
    No knowledge of the true VE surface is used.
    """
    return log.afr / target_afr(log.map_kpa)


def apply_correction(
    table: ArrayLike,
    factors: ArrayLike,
    gain: float = 1.0,
    ve_min: float = 0.20,
    ve_max: float = 1.15,
) -> NDArray[np.float64]:
    """Apply damped correction factors to a VE table.

    Implements ``VE_new = VE_old * (1 + gain * (factor - 1))``, a damped
    fixed-point update. A gain of one applies the full indicated correction;
    smaller gains trade convergence speed for resistance to measurement noise.

    Parameters
    ----------
    table : array_like
        Current VE table, as fractions.
    factors : array_like
        Correction factor per cell, same shape as `table`.
    gain : float, optional
        Damping factor, by default 1.0. Must lie in (0, 1].
    ve_min, ve_max : float, optional
        Physical clamping bounds applied to the result.

    Returns
    -------
    numpy.ndarray
        The corrected table.

    Raises
    ------
    ValueError
        If shapes disagree or `gain` lies outside (0, 1].
    """
    tbl = np.asarray(table, dtype=float)
    fac = np.asarray(factors, dtype=float)

    if tbl.shape != fac.shape:
        raise ValueError("Table and correction factors must have the same shape")
    if not 0.0 < gain <= 1.0:
        raise ValueError("Gain must lie in (0, 1]")

    return np.clip(tbl * (1.0 + gain * (fac - 1.0)), ve_min, ve_max)


def baseline_correction(
    log: SensorLog,
    table: ArrayLike,
    gain: float = 1.0,
    rpm_axis: ArrayLike = DEFAULT_RPM_AXIS,
    map_axis: ArrayLike = DEFAULT_MAP_AXIS,
) -> CorrectionResult:
    """Run a single naive correction pass over a log.

    Deliberately minimal: every sample is used, with no noise filtering and no
    rejection of transient or fuel-cut conditions, and correction factors are
    assigned to cells by nearest-node averaging. Serves as the reference
    against which later refinements are measured.

    Parameters
    ----------
    log : SensorLog
        Measured engine data.
    table : array_like
        Starting VE table, as fractions.
    gain : float, optional
        Damping factor, by default 1.0 for a full single-pass correction.
    rpm_axis, map_axis : array_like, optional
        Table breakpoints.

    Returns
    -------
    CorrectionResult
        The corrected table and supporting diagnostics.

    Notes
    -----
    Two failure modes are expected and intended. Overrun samples carry a
    saturated lean reading that is not a fuelling error, so cells receiving
    them are corrected sharply upwards. Transient samples are attributed to
    whichever cell the engine occupied when the reading arrived rather than
    when the charge was burnt, smearing corrections across neighbours.
    """
    factors_per_sample = sample_correction_factors(log)
    factors, counts = per_cell_average(
        factors_per_sample, log.rpm, log.map_kpa, rpm_axis, map_axis, default=1.0
    )
    corrected = apply_correction(table, factors, gain=gain)

    return CorrectionResult(
        table=corrected,
        factors=factors,
        counts=counts,
        n_samples_used=len(factors_per_sample),
    )
    
    
def select_samples(log: SensorLog, mask: ArrayLike) -> SensorLog:
    """Return a log containing only the samples where `mask` is True.

    Parameters
    ----------
    log : SensorLog
        Measured engine data.
    mask : array_like
        Boolean mask of samples to keep.

    Returns
    -------
    SensorLog
        A log holding the selected samples.

    Raises
    ------
    ValueError
        If the mask length does not match the log.
    """
    keep = np.asarray(mask, dtype=bool)
    if keep.shape != log.time_s.shape:
        raise ValueError("Mask must have the same length as the log")

    return replace(
        log,
        time_s=log.time_s[keep],
        rpm=log.rpm[keep],
        map_kpa=log.map_kpa[keep],
        iat_c=log.iat_c[keep],
        tps=log.tps[keep],
        afr=log.afr[keep],
        batt_v=log.batt_v[keep],
    )


def gated_correction(
    log: SensorLog,
    table: ArrayLike,
    gain: float = 1.0,
    gating: GatingConfig = GatingConfig(),
    rpm_axis: ArrayLike = DEFAULT_RPM_AXIS,
    map_axis: ArrayLike = DEFAULT_MAP_AXIS,
) -> tuple[CorrectionResult, GatingResult]:
    """Run a correction pass using only samples that pass gating.

    Identical to `baseline_correction` except that overrun, transient,
    out-of-domain and implausible samples are discarded first.

    Parameters
    ----------
    log : SensorLog
        Measured engine data.
    table : array_like
        Starting VE table, as fractions.
    gain : float, optional
        Damping factor, by default 1.0.
    gating : GatingConfig, optional
        Gating thresholds.
    rpm_axis, map_axis : array_like, optional
        Table breakpoints.

    Returns
    -------
    tuple
        The correction result and the gating result, so that the samples
        rejected can be reported alongside the table produced.
    """
    gate = gate_samples(log, gating, rpm_axis, map_axis)
    kept = select_samples(log, gate.accepted)
    result = baseline_correction(kept, table, gain=gain, rpm_axis=rpm_axis, map_axis=map_axis)
    return result, gate
