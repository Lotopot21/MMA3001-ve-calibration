"""Scoring for corrected VE tables.

Two independent views are provided. Table metrics compare an estimated surface
against the known truth, which is possible only with synthetic data. Air-fuel
ratio metrics score a table by how well an engine running it holds its target,
which is measurable on a real engine and so transfers to real logs.
"""

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from vetuner.drive_cycle import DriveCycle
from vetuner.engine_model import simulate_engine


@dataclass(frozen=True)
class TableMetrics:
    """Accuracy of an estimated VE table against the known truth.

    All errors are in volumetric efficiency percentage points, so a value of
    2.0 means the estimate is wrong by two points of VE, such as 88 instead
    of 90.

    Attributes
    ----------
    rmse : float
        Root mean square error across all cells.
    mean_abs_error : float
        Mean absolute error across all cells.
    max_abs_error : float
        Largest absolute error in any cell.
    rmse_visited : float
        Root mean square error over cells the log reached.
    rmse_unvisited : float
        Root mean square error over cells the log never reached. NaN when the
        log covered every cell.
    coverage : float
        Fraction of cells containing at least one sample.
    n_visited, n_cells : int
        Cell counts.
    """

    rmse: float
    mean_abs_error: float
    max_abs_error: float
    rmse_visited: float
    rmse_unvisited: float
    coverage: float
    n_visited: int
    n_cells: int


@dataclass(frozen=True)
class AfrMetrics:
    """Air-fuel ratio accuracy achieved by a table on a given cycle.

    All errors are percentages of the target air-fuel ratio.

    Attributes
    ----------
    rmse_pct : float
        Root mean square relative error.
    mean_abs_pct : float
        Mean absolute relative error.
    p95_abs_pct : float
        Ninety-fifth percentile absolute relative error.
    max_abs_pct : float
        Largest absolute relative error.
    fraction_within_2pct : float
        Proportion of fuelled samples held within two percent of target.
    """

    rmse_pct: float
    mean_abs_pct: float
    p95_abs_pct: float
    max_abs_pct: float
    fraction_within_2pct: float


def evaluate_table(
    estimated: ArrayLike,
    truth: ArrayLike,
    counts: ArrayLike | None = None,
) -> TableMetrics:
    """Score an estimated VE table against the known true surface.

    Parameters
    ----------
    estimated : array_like
        Estimated VE table, as fractions.
    truth : array_like
        True VE table, as fractions, sampled on the same grid.
    counts : array_like, optional
        Samples per cell. When supplied, errors are also reported separately
        for visited and unvisited cells.

    Returns
    -------
    TableMetrics
        Error statistics in VE percentage points.

    Raises
    ------
    ValueError
        If the arrays do not share a shape.
    """
    est = np.asarray(estimated, dtype=float) * 100.0
    ref = np.asarray(truth, dtype=float) * 100.0

    if est.shape != ref.shape:
        raise ValueError("Estimated and true tables must have the same shape")

    error = est - ref

    if counts is None:
        visited = np.ones(est.shape, dtype=bool)
    else:
        cnt = np.asarray(counts)
        if cnt.shape != est.shape:
            raise ValueError("Counts must match the table shape")
        visited = cnt > 0

    def _rmse(values: NDArray[np.float64]) -> float:
        return float(np.sqrt(np.mean(values**2))) if values.size else float("nan")

    return TableMetrics(
        rmse=_rmse(error),
        mean_abs_error=float(np.mean(np.abs(error))),
        max_abs_error=float(np.max(np.abs(error))),
        rmse_visited=_rmse(error[visited]),
        rmse_unvisited=_rmse(error[~visited]),
        coverage=float(visited.mean()),
        n_visited=int(visited.sum()),
        n_cells=int(est.size),
    )


def evaluate_afr(
    table: ArrayLike,
    cycle: DriveCycle,
    **simulation_kwargs: object,
) -> AfrMetrics:
    """Score a VE table by the fuelling accuracy it achieves on a cycle.

    Runs the forward engine model with the supplied table and compares the
    resulting air-fuel ratio against target. Passing a cycle not used to fit
    the table gives a held-out estimate of generalisation.

    Parameters
    ----------
    table : array_like
        VE table to evaluate, as fractions.
    cycle : DriveCycle
        Operating conditions to evaluate over.
    **simulation_kwargs
        Forwarded to `vetuner.engine_model.simulate_engine`.

    Returns
    -------
    AfrMetrics
        Relative error statistics over fuelled samples. Overrun samples are
        excluded, since air-fuel ratio is undefined when no fuel is injected.
    """
    result = simulate_engine(cycle, table, **simulation_kwargs)
    fuelled = ~result.fuel_cut
    relative = np.abs(
        result.afr_actual[fuelled] / result.afr_target[fuelled] - 1.0
    ) * 100.0

    return AfrMetrics(
        rmse_pct=float(np.sqrt(np.mean(relative**2))),
        mean_abs_pct=float(np.mean(relative)),
        p95_abs_pct=float(np.percentile(relative, 95)),
        max_abs_pct=float(np.max(relative)),
        fraction_within_2pct=float(np.mean(relative < 2.0)),
    )