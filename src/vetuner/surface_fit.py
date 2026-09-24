"""Surface fitting methods for scattered VE correction data.

A log visits operating points wherever the engine happened to run, so the
correction factors it yields are scattered irregularly across the speed-load
plane. These functions estimate a value at each table node from that scattered
data.

This module currently provides per-cell averaging. Further methods are added
for comparison in later work.
"""

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.interpolate import RBFInterpolator
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel

from vetuner.ve_surface import DEFAULT_MAP_AXIS, DEFAULT_RPM_AXIS


def nearest_cell_indices(
    rpm: ArrayLike,
    map_kpa: ArrayLike,
    rpm_axis: ArrayLike = DEFAULT_RPM_AXIS,
    map_axis: ArrayLike = DEFAULT_MAP_AXIS,
) -> tuple[NDArray[np.intp], NDArray[np.intp]]:
    """Assign each sample to its nearest table node.

    Samples outside the axis range are assigned to the nearest edge node,
    matching the clamping an ECU applies when reading a table.

    Parameters
    ----------
    rpm : array_like
        Engine speed of each sample, in rpm.
    map_kpa : array_like
        Manifold pressure of each sample, in kPa.
    rpm_axis, map_axis : array_like, optional
        Table breakpoints.

    Returns
    -------
    tuple of numpy.ndarray
        Speed and load indices, one pair per sample.

    Raises
    ------
    ValueError
        If `rpm` and `map_kpa` have different lengths.

    Notes
    -----
    Assignment is by nearest breakpoint, so a sample lying between two nodes
    contributes its full weight to one of them. This discards sub-cell
    position information, which is the principal weakness of the method.
    """
    rpm_arr = np.asarray(rpm, dtype=float)
    map_arr = np.asarray(map_kpa, dtype=float)

    if rpm_arr.shape != map_arr.shape:
        raise ValueError("Speed and load sample arrays must have the same shape")

    rpm_ax = np.asarray(rpm_axis, dtype=float)
    map_ax = np.asarray(map_axis, dtype=float)

    rpm_idx = np.abs(rpm_arr[:, None] - rpm_ax[None, :]).argmin(axis=1)
    map_idx = np.abs(map_arr[:, None] - map_ax[None, :]).argmin(axis=1)
    return rpm_idx.astype(np.intp), map_idx.astype(np.intp)


def cell_sample_counts(
    rpm: ArrayLike,
    map_kpa: ArrayLike,
    rpm_axis: ArrayLike = DEFAULT_RPM_AXIS,
    map_axis: ArrayLike = DEFAULT_MAP_AXIS,
) -> NDArray[np.int64]:
    """Count how many samples fall in each table cell.

    Parameters
    ----------
    rpm : array_like
        Engine speed of each sample, in rpm.
    map_kpa : array_like
        Manifold pressure of each sample, in kPa.
    rpm_axis, map_axis : array_like, optional
        Table breakpoints.

    Returns
    -------
    numpy.ndarray
        Integer counts with shape ``(len(rpm_axis), len(map_axis))``.
    """
    rpm_idx, map_idx = nearest_cell_indices(rpm, map_kpa, rpm_axis, map_axis)
    counts = np.zeros(
        (len(np.asarray(rpm_axis)), len(np.asarray(map_axis))), dtype=np.int64
    )
    np.add.at(counts, (rpm_idx, map_idx), 1)
    return counts


def per_cell_average(
    values: ArrayLike,
    rpm: ArrayLike,
    map_kpa: ArrayLike,
    rpm_axis: ArrayLike = DEFAULT_RPM_AXIS,
    map_axis: ArrayLike = DEFAULT_MAP_AXIS,
    default: float = 1.0,
) -> tuple[NDArray[np.float64], NDArray[np.int64]]:
    """Average scattered values onto table nodes by nearest-node binning.

    Parameters
    ----------
    values : array_like
        Value associated with each sample, such as a correction factor.
    rpm : array_like
        Engine speed of each sample, in rpm.
    map_kpa : array_like
        Manifold pressure of each sample, in kPa.
    rpm_axis, map_axis : array_like, optional
        Table breakpoints.
    default : float, optional
        Value assigned to cells containing no samples, by default 1.0.
        A correction factor of one leaves the existing table untouched.

    Returns
    -------
    tuple of numpy.ndarray
        The averaged surface and the sample count per cell, both shaped
        ``(len(rpm_axis), len(map_axis))``.

    Raises
    ------
    ValueError
        If `values` does not match the sample arrays in length.

    Notes
    -----
    Cost is linear in sample count. Cells holding few samples inherit the full
    noise of those samples, since averaging reduces noise only as the square
    root of the count.
    """
    vals = np.asarray(values, dtype=float)
    rpm_arr = np.asarray(rpm, dtype=float)

    if vals.shape != rpm_arr.shape:
        raise ValueError("Values must have the same length as the sample arrays")

    rpm_idx, map_idx = nearest_cell_indices(rpm, map_kpa, rpm_axis, map_axis)
    shape = (len(np.asarray(rpm_axis)), len(np.asarray(map_axis)))

    totals = np.zeros(shape, dtype=float)
    counts = np.zeros(shape, dtype=np.int64)
    np.add.at(totals, (rpm_idx, map_idx), vals)
    np.add.at(counts, (rpm_idx, map_idx), 1)

    surface = np.full(shape, float(default))
    visited = counts > 0
    surface[visited] = totals[visited] / counts[visited]
    return surface, counts

@dataclass(frozen=True)
class CellAggregate:
    """Scattered samples reduced to one point per visited table cell.

    Attributes
    ----------
    coords : numpy.ndarray
        Cell coordinates of each visited cell, shape ``(n_visited, 2)``, as
        fractional table indices.
    mean : numpy.ndarray
        Mean value in each visited cell.
    count : numpy.ndarray
        Number of samples contributing to each visited cell.
    variance : numpy.ndarray
        Sample variance within each visited cell. Zero where a cell holds a
        single sample.
    indices : tuple of numpy.ndarray
        Speed and load indices of each visited cell.
    """

    coords: NDArray[np.float64]
    mean: NDArray[np.float64]
    count: NDArray[np.int64]
    variance: NDArray[np.float64]
    indices: tuple[NDArray[np.intp], NDArray[np.intp]]


def to_cell_coords(
    rpm: ArrayLike,
    map_kpa: ArrayLike,
    rpm_axis: ArrayLike = DEFAULT_RPM_AXIS,
    map_axis: ArrayLike = DEFAULT_MAP_AXIS,
) -> NDArray[np.float64]:
    """Convert operating points into fractional table index coordinates.

    A point lying exactly on breakpoint ``i`` maps to ``i``; a point halfway
    between breakpoints ``i`` and ``i + 1`` maps to ``i + 0.5``.

    Working in these coordinates gives both axes comparable scale, so that
    distance-based methods weight engine speed and load equivalently rather
    than being dominated by whichever axis spans the larger numerical range.

    Parameters
    ----------
    rpm : array_like
        Engine speeds in rpm.
    map_kpa : array_like
        Manifold pressures in kPa.
    rpm_axis, map_axis : array_like, optional
        Table breakpoints, strictly increasing.

    Returns
    -------
    numpy.ndarray
        Coordinates with shape ``(n_samples, 2)``. Points beyond the axes are
        clamped to the edge indices.
    """
    rpm_ax = np.asarray(rpm_axis, dtype=float)
    map_ax = np.asarray(map_axis, dtype=float)

    x = np.interp(np.asarray(rpm, dtype=float), rpm_ax, np.arange(len(rpm_ax)))
    y = np.interp(np.asarray(map_kpa, dtype=float), map_ax, np.arange(len(map_ax)))
    return np.column_stack([x, y])


def table_node_coords(
    rpm_axis: ArrayLike = DEFAULT_RPM_AXIS,
    map_axis: ArrayLike = DEFAULT_MAP_AXIS,
) -> NDArray[np.float64]:
    """Return the cell coordinates of every table node.

    Parameters
    ----------
    rpm_axis, map_axis : array_like, optional
        Table breakpoints.

    Returns
    -------
    numpy.ndarray
        Coordinates with shape ``(n_rpm * n_map, 2)``, ordered to match a
        C-order flattening of the table.
    """
    n_rpm = len(np.asarray(rpm_axis))
    n_map = len(np.asarray(map_axis))
    xi, yi = np.meshgrid(np.arange(n_rpm), np.arange(n_map), indexing="ij")
    return np.column_stack([xi.ravel().astype(float), yi.ravel().astype(float)])


def aggregate_to_cells(
    values: ArrayLike,
    rpm: ArrayLike,
    map_kpa: ArrayLike,
    rpm_axis: ArrayLike = DEFAULT_RPM_AXIS,
    map_axis: ArrayLike = DEFAULT_MAP_AXIS,
) -> CellAggregate:
    """Reduce scattered samples to one summary point per visited cell.

    The cell mean is a sufficient statistic for the underlying correction
    factor under independent measurement noise, so fitting the means rather
    than the raw samples loses no information while reducing the number of
    points by orders of magnitude. The per-cell count and variance are
    retained so that later fitting can weight each cell by how well it is
    determined.

    Parameters
    ----------
    values : array_like
        Value associated with each sample.
    rpm : array_like
        Engine speed of each sample, in rpm.
    map_kpa : array_like
        Manifold pressure of each sample, in kPa.
    rpm_axis, map_axis : array_like, optional
        Table breakpoints.

    Returns
    -------
    CellAggregate
        Summary of the visited cells only.

    Raises
    ------
    ValueError
        If no cell receives any sample.
    """
    vals = np.asarray(values, dtype=float)
    rpm_idx, map_idx = nearest_cell_indices(rpm, map_kpa, rpm_axis, map_axis)

    shape = (len(np.asarray(rpm_axis)), len(np.asarray(map_axis)))
    totals = np.zeros(shape)
    squares = np.zeros(shape)
    counts = np.zeros(shape, dtype=np.int64)

    np.add.at(totals, (rpm_idx, map_idx), vals)
    np.add.at(squares, (rpm_idx, map_idx), vals**2)
    np.add.at(counts, (rpm_idx, map_idx), 1)

    visited_i, visited_j = np.nonzero(counts)
    if visited_i.size == 0:
        raise ValueError("No samples fall within the table")

    n = counts[visited_i, visited_j]
    mean = totals[visited_i, visited_j] / n
    variance = np.maximum(squares[visited_i, visited_j] / n - mean**2, 0.0)

    return CellAggregate(
        coords=np.column_stack([visited_i.astype(float), visited_j.astype(float)]),
        mean=mean,
        count=n,
        variance=variance,
        indices=(visited_i, visited_j),
    )


def bilinear_scatter(
    values: ArrayLike,
    rpm: ArrayLike,
    map_kpa: ArrayLike,
    rpm_axis: ArrayLike = DEFAULT_RPM_AXIS,
    map_axis: ArrayLike = DEFAULT_MAP_AXIS,
    default: float = 1.0,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Distribute scattered values onto table nodes by bilinear weighting.

    Each sample contributes to the four surrounding nodes in proportion to its
    proximity to each. This is the transpose of the bilinear interpolation an
    ECU performs when reading the table, so the operation inverts the one that
    produced the measurement. Unlike nearest-node binning it preserves
    sub-cell position, removing the quantisation error that arises when a
    sample lying between nodes is assigned wholly to one.

    Parameters
    ----------
    values : array_like
        Value associated with each sample.
    rpm : array_like
        Engine speed of each sample, in rpm.
    map_kpa : array_like
        Manifold pressure of each sample, in kPa.
    rpm_axis, map_axis : array_like, optional
        Table breakpoints.
    default : float, optional
        Value assigned to nodes receiving no weight, by default 1.0.

    Returns
    -------
    tuple of numpy.ndarray
        The fitted surface and the total weight accumulated at each node,
        both shaped to the table.

    Raises
    ------
    ValueError
        If `values` does not match the sample arrays in length.
    """
    vals = np.asarray(values, dtype=float)
    coords = to_cell_coords(rpm, map_kpa, rpm_axis, map_axis)

    if vals.shape[0] != coords.shape[0]:
        raise ValueError("Values must have the same length as the sample arrays")

    shape = (len(np.asarray(rpm_axis)), len(np.asarray(map_axis)))

    i0 = np.clip(np.floor(coords[:, 0]).astype(np.intp), 0, shape[0] - 2)
    j0 = np.clip(np.floor(coords[:, 1]).astype(np.intp), 0, shape[1] - 2)
    u = np.clip(coords[:, 0] - i0, 0.0, 1.0)
    v = np.clip(coords[:, 1] - j0, 0.0, 1.0)

    weighted = np.zeros(shape)
    weights = np.zeros(shape)

    for di, dj, w in (
        (0, 0, (1.0 - u) * (1.0 - v)),
        (1, 0, u * (1.0 - v)),
        (0, 1, (1.0 - u) * v),
        (1, 1, u * v),
    ):
        np.add.at(weighted, (i0 + di, j0 + dj), w * vals)
        np.add.at(weights, (i0 + di, j0 + dj), w)

    surface = np.full(shape, float(default))
    supported = weights > 1e-12
    surface[supported] = weighted[supported] / weights[supported]
    return surface, weights


def polynomial_surface(
    values: ArrayLike,
    rpm: ArrayLike,
    map_kpa: ArrayLike,
    rpm_axis: ArrayLike = DEFAULT_RPM_AXIS,
    map_axis: ArrayLike = DEFAULT_MAP_AXIS,
    degree: int = 3,
) -> NDArray[np.float64]:
    """Fit a global bivariate polynomial by least squares.

    Produces a smooth surface defined everywhere, including cells the log never
    visited. The cost of that coverage is that a low-order polynomial cannot
    represent local features: a resonance peak or dip narrower than the fitted
    surface's curvature is flattened out.

    Parameters
    ----------
    values : array_like
        Value associated with each sample.
    rpm : array_like
        Engine speed of each sample, in rpm.
    map_kpa : array_like
        Manifold pressure of each sample, in kPa.
    rpm_axis, map_axis : array_like, optional
        Table breakpoints.
    degree : int, optional
        Total polynomial degree, by default 3.

    Returns
    -------
    numpy.ndarray
        The fitted surface, shaped to the table.

    Raises
    ------
    ValueError
        If `degree` is negative, or there are fewer samples than coefficients.
    """
    if degree < 0:
        raise ValueError("Polynomial degree must be non-negative")

    vals = np.asarray(values, dtype=float)
    coords = to_cell_coords(rpm, map_kpa, rpm_axis, map_axis)
    powers = [(a, b) for a in range(degree + 1) for b in range(degree + 1 - a)]

    if len(vals) < len(powers):
        raise ValueError(
            f"Need at least {len(powers)} samples for a degree {degree} fit"
        )

    def design(points: NDArray[np.float64]) -> NDArray[np.float64]:
        return np.column_stack([points[:, 0] ** a * points[:, 1] ** b for a, b in powers])

    coefficients, *_ = np.linalg.lstsq(design(coords), vals, rcond=None)
    nodes = table_node_coords(rpm_axis, map_axis)
    shape = (len(np.asarray(rpm_axis)), len(np.asarray(map_axis)))
    return (design(nodes) @ coefficients).reshape(shape)


def thin_plate_spline(
    values: ArrayLike,
    rpm: ArrayLike,
    map_kpa: ArrayLike,
    rpm_axis: ArrayLike = DEFAULT_RPM_AXIS,
    map_axis: ArrayLike = DEFAULT_MAP_AXIS,
    smoothing: float = 1.0,
) -> NDArray[np.float64]:
    """Fit a thin plate spline to cell-aggregated values.

    A thin plate spline minimises bending energy subject to fitting the data,
    giving the smoothest surface consistent with the observations. It handles
    irregular sampling natively and reproduces local features that a global
    polynomial cannot.

    Samples are aggregated to cell means first, both to reduce the cubic
    solve cost and to average out measurement noise before fitting.

    Parameters
    ----------
    values : array_like
        Value associated with each sample.
    rpm : array_like
        Engine speed of each sample, in rpm.
    map_kpa : array_like
        Manifold pressure of each sample, in kPa.
    rpm_axis, map_axis : array_like, optional
        Table breakpoints.
    smoothing : float, optional
        Regularisation weight, by default 1.0. Zero interpolates the cell
        means exactly; larger values trade fidelity for smoothness.

    Returns
    -------
    numpy.ndarray
        The fitted surface, shaped to the table.

    Notes
    -----
    Extrapolation beyond the region the log covered is unconstrained and may
    move sharply away from plausible values. Cells far from any data should be
    treated as unreliable regardless of the value returned.
    """
    cells = aggregate_to_cells(values, rpm, map_kpa, rpm_axis, map_axis)
    interpolator = RBFInterpolator(
        cells.coords, cells.mean, kernel="thin_plate_spline", smoothing=smoothing
    )
    shape = (len(np.asarray(rpm_axis)), len(np.asarray(map_axis)))
    return interpolator(table_node_coords(rpm_axis, map_axis)).reshape(shape)


def gaussian_process_surface(
    values: ArrayLike,
    rpm: ArrayLike,
    map_kpa: ArrayLike,
    rpm_axis: ArrayLike = DEFAULT_RPM_AXIS,
    map_axis: ArrayLike = DEFAULT_MAP_AXIS,
    length_scale_cells: float = 2.0,
    seed: int = 0,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Fit a Gaussian process to cell-aggregated values.

    Unlike the other methods this returns an uncertainty alongside the
    estimate. The predicted standard deviation grows with distance from the
    observed data, so cells the log never reached are identified as such
    rather than silently receiving an extrapolated value. That makes it
    possible to leave unsupported cells at their existing value instead of
    applying a correction derived from nothing.

    Each cell is weighted by the standard error of its mean, so cells holding
    many consistent samples are fitted closely while thin or noisy cells are
    allowed to deviate.

    Parameters
    ----------
    values : array_like
        Value associated with each sample.
    rpm : array_like
        Engine speed of each sample, in rpm.
    map_kpa : array_like
        Manifold pressure of each sample, in kPa.
    rpm_axis, map_axis : array_like, optional
        Table breakpoints.
    length_scale_cells : float, optional
        Initial correlation length in table cells, by default 2.0. Optimised
        during fitting; the initial value only sets where the search starts.
    seed : int, optional
        Random seed for the optimiser restarts, by default 0.

    Returns
    -------
    tuple of numpy.ndarray
        The fitted surface and its predicted standard deviation, both shaped
        to the table.

    Notes
    -----
    A Matern kernel with smoothness parameter 5/2 is used rather than a
    squared exponential. The squared exponential assumes an infinitely
    differentiable surface, which over-smooths the resonance features a real
    volumetric efficiency surface contains.
    """
    cells = aggregate_to_cells(values, rpm, map_kpa, rpm_axis, map_axis)

    typical_variance = float(np.median(cells.variance[cells.variance > 0.0])) \
        if np.any(cells.variance > 0.0) else 1e-6
    per_cell_variance = np.where(cells.count > 1, cells.variance, typical_variance)
    alpha = np.maximum(per_cell_variance / cells.count, 1e-8)

    kernel = ConstantKernel(1.0, (1e-5, 1e5)) * Matern(
        length_scale=[length_scale_cells, length_scale_cells],
        length_scale_bounds=(0.1, 100.0),
        nu=2.5,
    ) + WhiteKernel(1e-4, (1e-10, 1e-1))

    offset = float(np.mean(cells.mean))
    model = GaussianProcessRegressor(
        kernel=kernel, alpha=alpha, normalize_y=False, n_restarts_optimizer=2,
        random_state=seed,
    )
    model.fit(cells.coords, cells.mean - offset)

    mean, std = model.predict(table_node_coords(rpm_axis, map_axis), return_std=True)
    shape = (len(np.asarray(rpm_axis)), len(np.asarray(map_axis)))
    return (mean + offset).reshape(shape), std.reshape(shape)


def blend_by_confidence(
    surface: ArrayLike,
    std: ArrayLike,
    default: float = 1.0,
    std_threshold: float = 0.02,
) -> NDArray[np.float64]:
    """Fade an estimated surface towards a default where it is uncertain.

    Converts the Gaussian process uncertainty into an engineering decision:
    where the fit is well supported the estimate is used in full, and where it
    is not the surface returns to a neutral value that leaves the existing
    table unchanged. The transition is gradual rather than a hard cutoff, so
    no discontinuity is introduced at the edge of the covered region.

    Parameters
    ----------
    surface : array_like
        Estimated surface.
    std : array_like
        Predicted standard deviation, same shape as `surface`.
    default : float, optional
        Value to fade towards, by default 1.0 for a correction factor.
    std_threshold : float, optional
        Standard deviation at which the estimate is given half weight, by
        default 0.02.

    Returns
    -------
    numpy.ndarray
        The blended surface.

    Raises
    ------
    ValueError
        If the shapes disagree or `std_threshold` is not positive.
    """
    est = np.asarray(surface, dtype=float)
    sigma = np.asarray(std, dtype=float)

    if est.shape != sigma.shape:
        raise ValueError("Surface and standard deviation must have the same shape")
    if std_threshold <= 0.0:
        raise ValueError("Standard deviation threshold must be positive")

    confidence = 1.0 / (1.0 + (sigma / std_threshold) ** 2)
    return confidence * est + (1.0 - confidence) * default


def fit_surface(
    method: str,
    values: ArrayLike,
    rpm: ArrayLike,
    map_kpa: ArrayLike,
    rpm_axis: ArrayLike = DEFAULT_RPM_AXIS,
    map_axis: ArrayLike = DEFAULT_MAP_AXIS,
    **kwargs: object,
) -> NDArray[np.float64]:
    """Fit a surface using the named method.

    Provides a uniform interface for comparing methods on identical data.

    Parameters
    ----------
    method : str
        One of ``"per_cell"``, ``"bilinear"``, ``"polynomial"``,
        ``"thin_plate"``, ``"gaussian_process"`` or ``"gp_blended"``.
    values : array_like
        Value associated with each sample.
    rpm : array_like
        Engine speed of each sample, in rpm.
    map_kpa : array_like
        Manifold pressure of each sample, in kPa.
    rpm_axis, map_axis : array_like, optional
        Table breakpoints.
    **kwargs
        Forwarded to the underlying method.

    Returns
    -------
    numpy.ndarray
        The fitted surface, shaped to the table.

    Raises
    ------
    ValueError
        If `method` is not recognised.
    """
    args = (values, rpm, map_kpa, rpm_axis, map_axis)

    if method == "per_cell":
        return per_cell_average(*args, **kwargs)[0]
    if method == "bilinear":
        return bilinear_scatter(*args, **kwargs)[0]
    if method == "polynomial":
        return polynomial_surface(*args, **kwargs)
    if method == "thin_plate":
        return thin_plate_spline(*args, **kwargs)
    if method == "gaussian_process":
        return gaussian_process_surface(*args, **kwargs)[0]
    if method == "gp_blended":
        surface, std = gaussian_process_surface(*args, **kwargs)
        return blend_by_confidence(surface, std)

    raise ValueError(f"Unknown surface fitting method: {method!r}")


SURFACE_METHODS = (
    "per_cell",
    "bilinear",
    "polynomial",
    "thin_plate",
    "gaussian_process",
    "gp_blended",
)
"""Names accepted by `fit_surface`, in increasing order of sophistication."""