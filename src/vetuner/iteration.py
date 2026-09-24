"""Iterative VE table calibration.

A single correction pass recovers the true table exactly when measurements are
perfect and coverage is complete. Neither holds for real logs, so the
correction is applied repeatedly as a damped fixed-point iteration::

    VE_{n+1} = VE_n * (1 + k * (AFR_measured / AFR_target - 1))

The damping gain ``k`` controls the trade-off that governs the scheme. Large
gains converge quickly but carry the full measurement noise of each pass into
the table; small gains suppress noise but approach the solution slowly and may
stall before reaching it.

Each pass re-runs the forward engine model with the current table, because
correcting the table changes what the engine would do. Re-using the original
log would apply corrections derived from a calibration that is no longer in
force, and the iteration would diverge from the physical process it
represents.
"""

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from vetuner.correction import apply_correction, sample_correction_factors, select_samples
from vetuner.drive_cycle import DriveCycle
from vetuner.engine_model import EngineGeometry, Injector, simulate_engine
from vetuner.gating import GatingConfig, gate_samples
from vetuner.sensors import SensorConfig, apply_sensor_model
from vetuner.surface_fit import fit_surface
from vetuner.ve_surface import (
    DEFAULT_MAP_AXIS,
    DEFAULT_RPM_AXIS,
    TRUE_SURFACE,
    VESurfaceParams,
)


@dataclass(frozen=True)
class IterationHistory:
    """Record of a complete iterative calibration run.

    Attributes
    ----------
    tables : list of numpy.ndarray
        The table after each pass, starting with the initial table. Length is
        one greater than the number of passes performed.
    max_change : list of float
        Largest absolute change to any cell in each pass, in VE fractions.
        Used as the convergence criterion.
    rms_change : list of float
        Root mean square change across all cells in each pass.
    coverage : list of float
        Fraction of cells receiving data in each pass.
    n_accepted : list of int
        Samples surviving gating in each pass.
    converged : bool
        Whether the stopping criterion was met before the pass limit.
    n_passes : int
        Number of passes performed.

    Notes
    -----
    Convergence here means the iteration stopped changing the table, not that
    the table is correct. A scheme can converge to a wrong answer if the data
    is biased or the coverage is incomplete, so the change criterion must be
    read alongside an independent accuracy measure.
    """

    tables: list[NDArray[np.float64]]
    max_change: list[float]
    rms_change: list[float]
    coverage: list[float]
    n_accepted: list[int]
    converged: bool
    n_passes: int

    @property
    def final_table(self) -> NDArray[np.float64]:
        """The table produced by the last pass."""
        return self.tables[-1]


def iterate_calibration(
    cycle: DriveCycle,
    initial_table: ArrayLike,
    gain: float = 0.7,
    max_passes: int = 10,
    tolerance: float = 1e-3,
    method: str = "gaussian_process",
    true_params: VESurfaceParams = TRUE_SURFACE,
    sensor_config: SensorConfig = SensorConfig(),
    gating_config: GatingConfig = GatingConfig(),
    engine: EngineGeometry = EngineGeometry(),
    injector: Injector = Injector(),
    rpm_axis: ArrayLike = DEFAULT_RPM_AXIS,
    map_axis: ArrayLike = DEFAULT_MAP_AXIS,
    seed: int = 0,
    **fit_kwargs: object,
) -> IterationHistory:
    """Run the damped fixed-point calibration to convergence.

    Each pass simulates the engine running the current table, logs the result
    through the sensor model, gates the samples, fits a correction surface and
    applies it with the given damping gain.

    A fresh random seed is drawn for each pass, so successive passes see
    independent measurement noise. Re-using one noise realisation would let the
    iteration converge onto that particular realisation rather than onto the
    underlying surface, understating the noise sensitivity the gain is there to
    control.

    Parameters
    ----------
    cycle : DriveCycle
        Operating conditions to calibrate against.
    initial_table : array_like
        Starting VE table, as fractions.
    gain : float, optional
        Damping factor in (0, 1], by default 0.7.
    max_passes : int, optional
        Maximum passes before stopping, by default 10.
    tolerance : float, optional
        Convergence threshold on the largest cell change in one pass, in VE
        fractions, by default 1e-3, or 0.1 VE percentage points.
    method : str, optional
        Surface fitting method, by default ``"gaussian_process"``.
    true_params : VESurfaceParams, optional
        Hidden true surface driving the simulated engine.
    sensor_config : SensorConfig, optional
        Sensor imperfection settings.
    gating_config : GatingConfig, optional
        Sample acceptance thresholds.
    engine : EngineGeometry, optional
        Engine geometry.
    injector : Injector, optional
        Injector characteristics.
    rpm_axis, map_axis : array_like, optional
        Table breakpoints.
    seed : int, optional
        Base random seed, by default 0. Pass ``n`` uses ``seed + n``.
    **fit_kwargs
        Forwarded to the surface fitting method.

    Returns
    -------
    IterationHistory
        The table after each pass and the convergence diagnostics.

    Raises
    ------
    ValueError
        If `gain` lies outside (0, 1], `max_passes` is not positive, or
        `tolerance` is not positive.
    """
    if not 0.0 < gain <= 1.0:
        raise ValueError("Gain must lie in (0, 1]")
    if max_passes < 1:
        raise ValueError("Must allow at least one pass")
    if tolerance <= 0.0:
        raise ValueError("Tolerance must be positive")

    table = np.asarray(initial_table, dtype=float).copy()
    tables = [table.copy()]
    max_change: list[float] = []
    rms_change: list[float] = []
    coverage: list[float] = []
    n_accepted: list[int] = []
    converged = False

    for pass_index in range(max_passes):
        result = simulate_engine(
            cycle, table, engine=engine, injector=injector,
            true_params=true_params, rpm_axis=rpm_axis, map_axis=map_axis,
        )
        log = apply_sensor_model(result, config=sensor_config, seed=seed + pass_index)
        kept = select_samples(log, gate_samples(log, gating_config, rpm_axis, map_axis).accepted)

        if len(kept) == 0:
            break

        factors = sample_correction_factors(kept)
        surface = fit_surface(
            method, factors, kept.rpm, kept.map_kpa, rpm_axis, map_axis, **fit_kwargs
        )
        updated = apply_correction(table, surface, gain=gain)

        change = updated - table
        max_change.append(float(np.abs(change).max()))
        rms_change.append(float(np.sqrt(np.mean(change**2))))
        n_accepted.append(len(kept))

        from vetuner.surface_fit import cell_sample_counts

        counts = cell_sample_counts(kept.rpm, kept.map_kpa, rpm_axis, map_axis)
        coverage.append(float((counts > 0).mean()))

        table = updated
        tables.append(table.copy())

        if max_change[-1] < tolerance:
            converged = True
            break

    return IterationHistory(
        tables=tables,
        max_change=max_change,
        rms_change=rms_change,
        coverage=coverage,
        n_accepted=n_accepted,
        converged=converged,
        n_passes=len(tables) - 1,
    )


def theoretical_error_decay(gain: float, n_passes: int) -> NDArray[np.float64]:
    """Predict the error remaining after each pass of an undamped ideal scheme.

    For a linear fixed-point iteration with damping gain ``k``, the error after
    ``n`` passes is ``(1 - k)**n`` times the initial error. This holds exactly
    when the correction factor equals the table error, which requires perfect
    measurements and complete coverage.

    Deviation from this curve therefore measures how far the real problem
    departs from the ideal: measurement noise, incomplete coverage and the
    smoothing introduced by surface fitting all slow convergence relative to
    the prediction.

    Parameters
    ----------
    gain : float
        Damping factor.
    n_passes : int
        Number of passes to predict.

    Returns
    -------
    numpy.ndarray
        Relative error remaining after each pass, starting at one before any
        pass has run. Length is ``n_passes + 1``.

    Raises
    ------
    ValueError
        If `n_passes` is negative.
    """
    if n_passes < 0:
        raise ValueError("Number of passes must be non-negative")
    return (1.0 - gain) ** np.arange(n_passes + 1, dtype=float)

def iterate_until_no_improvement(
    train_cycle: DriveCycle,
    test_cycle: DriveCycle,
    initial_table: ArrayLike,
    gain: float = 0.6,
    max_passes: int = 10,
    patience: int = 1,
    **kwargs: object,
) -> tuple[NDArray[np.float64], list[float], int]:
    """Iterate until held-out fuelling accuracy stops improving.

    The scheme's fixed point is not the true table: because the ECU reads the
    table by interpolation, the iteration converges to whichever table makes
    the *interpolated* read produce the target mixture, which differs from the
    truth by the interpolation error. Accuracy therefore reaches a minimum
    after a few passes and degrades slowly thereafter.

    A criterion based on the size of the table update cannot detect this, since
    the update continues to shrink while accuracy worsens. Scoring each pass on
    a cycle not used to fit it stops the iteration at its best point and uses
    only information available without knowing the true surface, so the same
    criterion applies to real logs.

    Parameters
    ----------
    train_cycle : DriveCycle
        Cycle used to generate calibration data.
    test_cycle : DriveCycle
        Independent cycle used to score each pass.
    initial_table : array_like
        Starting VE table, as fractions.
    gain : float, optional
        Damping factor, by default 0.6.
    max_passes : int, optional
        Maximum passes, by default 10.
    patience : int, optional
        Passes allowed to worsen before stopping, by default 1.
    **kwargs
        Forwarded to `iterate_calibration`.

    Returns
    -------
    tuple
        The best table found, the held-out error after each pass, and the
        index of the best pass.
    """
    from vetuner.metrics import evaluate_afr

    history = iterate_calibration(
        train_cycle, initial_table, gain=gain, max_passes=max_passes,
        tolerance=1e-12, **kwargs,
    )

    scores = [evaluate_afr(table, test_cycle).mean_abs_pct for table in history.tables]
    best = int(np.argmin(scores))

    for index in range(1, len(scores)):
        if index - int(np.argmin(scores[: index + 1])) > patience:
            best = int(np.argmin(scores[: index + 1]))
            break

    return history.tables[best], scores, best