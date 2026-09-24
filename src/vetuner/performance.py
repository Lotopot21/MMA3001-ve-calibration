"""Performance measurement and cost analysis.

Provides timing utilities, arithmetic cost estimates and an optimised cell
assignment routine, so that the computational cost of each calibration method
can be measured and explained rather than only observed.

Timing takes the minimum of several repeats rather than the mean. Measurement
noise on a shared machine is one-sided: an interrupted run is slower than an
uninterrupted one, never faster, so the minimum is the best available estimate
of the true cost while the mean reflects whatever else the machine was doing.
"""

import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

from vetuner.ve_surface import DEFAULT_MAP_AXIS, DEFAULT_RPM_AXIS


@dataclass(frozen=True)
class Timing:
    """Result of timing a callable.

    Attributes
    ----------
    best_ms : float
        Fastest observed run, in milliseconds. The best estimate of the true
        cost.
    median_ms : float
        Median run, in milliseconds. Its excess over `best_ms` indicates how
        much the measurement was disturbed.
    worst_ms : float
        Slowest observed run, in milliseconds.
    repeats : int
        Number of timed runs.
    """

    best_ms: float
    median_ms: float
    worst_ms: float
    repeats: int

    @property
    def noise_ratio(self) -> float:
        """Median divided by best, as a measure of timing stability."""
        return self.median_ms / self.best_ms if self.best_ms > 0 else float("nan")


def time_call(
    func: Callable[..., Any],
    *args: Any,
    repeats: int = 7,
    warmup: int = 1,
    **kwargs: Any,
) -> Timing:
    """Time a callable over several repeats.

    Warmup runs are discarded so that first-call costs, such as lazy imports
    and cache population, do not contaminate the measurement.

    Parameters
    ----------
    func : callable
        The function to time.
    *args
        Positional arguments passed to `func`.
    repeats : int, optional
        Number of timed runs, by default 7. Must be positive.
    warmup : int, optional
        Untimed runs performed first, by default 1.
    **kwargs
        Keyword arguments passed to `func`.

    Returns
    -------
    Timing
        Best, median and worst observed durations.

    Raises
    ------
    ValueError
        If `repeats` is not positive or `warmup` is negative.
    """
    if repeats < 1:
        raise ValueError("Must perform at least one timed run")
    if warmup < 0:
        raise ValueError("Warmup count cannot be negative")

    for _ in range(warmup):
        func(*args, **kwargs)

    durations = []
    for _ in range(repeats):
        start = time.perf_counter()
        func(*args, **kwargs)
        durations.append((time.perf_counter() - start) * 1e3)

    return Timing(
        best_ms=float(np.min(durations)),
        median_ms=float(np.median(durations)),
        worst_ms=float(np.max(durations)),
        repeats=repeats,
    )


def nearest_cell_indices_fast(
    rpm: ArrayLike,
    map_kpa: ArrayLike,
    rpm_axis: ArrayLike = DEFAULT_RPM_AXIS,
    map_axis: ArrayLike = DEFAULT_MAP_AXIS,
) -> tuple[NDArray[np.intp], NDArray[np.intp]]:
    """Assign samples to their nearest table node by binary search.

    Equivalent in result to the broadcasting implementation in
    `vetuner.surface_fit.nearest_cell_indices`, but avoids forming the full
    sample-by-breakpoint distance matrix.

    The broadcasting version evaluates every sample against every breakpoint,
    costing O(n*m) time and, more importantly, O(n*m) memory for the
    intermediate array. Because the breakpoints are sorted, a binary search
    finds each sample's insertion point in O(log m) and only the two adjacent
    breakpoints need comparing, reducing memory to O(n).

    Parameters
    ----------
    rpm : array_like
        Engine speed of each sample, in rpm.
    map_kpa : array_like
        Manifold pressure of each sample, in kPa.
    rpm_axis, map_axis : array_like, optional
        Table breakpoints, strictly increasing.

    Returns
    -------
    tuple of numpy.ndarray
        Speed and load indices, one pair per sample.

    Raises
    ------
    ValueError
        If the sample arrays differ in shape, or an axis has fewer than two
        breakpoints.
    """
    rpm_arr = np.asarray(rpm, dtype=float)
    map_arr = np.asarray(map_kpa, dtype=float)

    if rpm_arr.shape != map_arr.shape:
        raise ValueError("Speed and load sample arrays must have the same shape")

    def nearest(values: NDArray[np.float64], axis: NDArray[np.float64]) -> NDArray[np.intp]:
        if len(axis) < 2:
            raise ValueError("Each axis needs at least two breakpoints")
        upper = np.clip(np.searchsorted(axis, values), 1, len(axis) - 1)
        left = axis[upper - 1]
        right = axis[upper]
        return np.where(values - left <= right - values, upper - 1, upper).astype(np.intp)

    return (
        nearest(rpm_arr, np.asarray(rpm_axis, dtype=float)),
        nearest(map_arr, np.asarray(map_axis, dtype=float)),
    )


def estimate_flops(method: str, n_samples: int, n_cells: int, n_visited: int) -> float:
    """Estimate the floating point operations a surface fitting method performs.

    Estimates are order-of-magnitude, based on the dominant term of each
    algorithm rather than an exact instruction count. They exist to explain
    measured runtimes and to predict how each method scales, not to replace
    measurement.

    Parameters
    ----------
    method : str
        One of the names in `vetuner.surface_fit.SURFACE_METHODS`.
    n_samples : int
        Number of log samples entering the fit.
    n_cells : int
        Total number of table nodes.
    n_visited : int
        Number of cells receiving at least one sample.

    Returns
    -------
    float
        Estimated floating point operations.

    Raises
    ------
    ValueError
        If `method` is not recognised.

    Notes
    -----
    Assumptions by method:

    - ``per_cell``: a distance is formed against every breakpoint on both
      axes, so cost is linear in samples and in breakpoint count.
    - ``bilinear``: binary search for the containing cell, then four weights
      and four scattered additions per sample.
    - ``polynomial``: forming the design matrix costs ``n * p``, and the least
      squares solve ``n * p**2``, for ``p`` coefficients.
    - ``thin_plate`` and the Gaussian process: samples are first aggregated to
      cell means, so the cubic solve is over visited cells rather than
      samples. The Gaussian process repeats that solve once per likelihood
      evaluation during hyperparameter optimisation, taken here as 60.
    """
    n_axis = 2.0 * np.sqrt(n_cells)
    p = 10.0
    gp_likelihood_evaluations = 60.0

    if method == "per_cell":
        return 3.0 * n_samples * n_axis + 2.0 * n_samples
    if method == "bilinear":
        return n_samples * (2.0 * np.log2(max(n_axis, 2.0)) + 24.0)
    if method == "polynomial":
        return n_samples * p + n_samples * p**2 + p**3
    if method == "thin_plate":
        return 3.0 * n_samples * n_axis + n_visited**3 + n_cells * n_visited
    if method in ("gaussian_process", "gp_blended"):
        return (
            3.0 * n_samples * n_axis
            + gp_likelihood_evaluations * n_visited**3
            + n_cells * n_visited
        )

    raise ValueError(f"Unknown surface fitting method: {method!r}")


def arithmetic_intensity(flops: float, bytes_moved: float) -> float:
    """Compute arithmetic intensity in floating point operations per byte.

    Intensity determines whether a computation is limited by the processor's
    arithmetic throughput or by memory bandwidth. Modern processors sustain
    tens of operations per byte, so a routine below that threshold spends most
    of its time waiting for data rather than calculating.

    Parameters
    ----------
    flops : float
        Floating point operations performed.
    bytes_moved : float
        Bytes read from and written to memory.

    Returns
    -------
    float
        Operations per byte.

    Raises
    ------
    ValueError
        If `bytes_moved` is not positive.
    """
    if bytes_moved <= 0.0:
        raise ValueError("Bytes moved must be positive")
    return flops / bytes_moved


def peak_memory_bytes(func: Callable[..., Any], *args: Any, **kwargs: Any) -> tuple[Any, int]:
    """Measure the peak memory a call allocates.

    Parameters
    ----------
    func : callable
        The function to measure.
    *args, **kwargs
        Arguments passed to `func`.

    Returns
    -------
    tuple
        The function's return value and the peak allocation in bytes.

    Notes
    -----
    Uses `tracemalloc`, which counts Python-level allocations including NumPy
    array buffers. It does not capture allocations made inside compiled
    extension code that bypass the Python allocator, so results for
    library-heavy routines are a lower bound.
    """
    import tracemalloc

    tracemalloc.start()
    try:
        result = func(*args, **kwargs)
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()

    return result, int(peak)