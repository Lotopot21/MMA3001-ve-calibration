"""Noise filtering for logged sensor channels.

Three approaches are provided, differing in whether they may use future
samples and in what they preserve.

A moving average and a Savitzky-Golay filter are applied over a window centred
on each sample, so they use samples from both before and after. This makes
them non-causal and therefore unusable in real time, but it also makes them
free of phase lag, which matters here because a lagged air-fuel ratio reading
is attributed to the wrong operating point. An offline calibration tool can use
them; an engine control unit cannot.

The Kalman filter is causal, using only past samples, and so represents what
could run on an engine control unit. It is implemented directly rather than
through a library, since the scalar case is short and its steady-state
behaviour can be checked against a closed-form result.
"""

from dataclasses import dataclass, replace

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.signal import savgol_filter

from vetuner.sensors import SensorLog


@dataclass(frozen=True)
class KalmanState:
    """Steady-state properties of a scalar Kalman filter.

    Attributes
    ----------
    gain : float
        Steady-state Kalman gain, in [0, 1]. The weight given to each new
        measurement once the filter has settled.
    variance : float
        Steady-state posterior variance of the estimate.
    equivalent_tau_samples : float
        Time constant of the first-order lag that behaves identically, in
        samples. A steady-state Kalman filter with gain ``K`` is an
        exponential moving average, so its response is that of a first-order
        lag with ``tau = -1 / ln(1 - K)`` samples.
    """

    gain: float
    variance: float
    equivalent_tau_samples: float


def moving_average(signal: ArrayLike, window: int) -> NDArray[np.float64]:
    """Smooth a signal with a centred moving average.

    Averaging ``N`` independent samples reduces the standard deviation of white
    noise by a factor of ``sqrt(N)``, at the cost of attenuating genuine
    features narrower than the window.

    Centring the window makes the filter non-causal: the output at each sample
    depends on samples that follow it. In exchange the filter introduces no
    phase lag, which a trailing average of the same length would.

    Parameters
    ----------
    signal : array_like
        Input samples.
    window : int
        Window length in samples. Must be positive. Even lengths are increased
        by one so that the window can be centred.

    Returns
    -------
    numpy.ndarray
        Smoothed signal, same shape as `signal`. Samples within half a window
        of each end are averaged over the samples available.

    Raises
    ------
    ValueError
        If `window` is not positive or exceeds the signal length.
    """
    values = np.asarray(signal, dtype=float)

    if window < 1:
        raise ValueError("Window length must be positive")
    if window > len(values):
        raise ValueError("Window length cannot exceed the signal length")

    width = window if window % 2 == 1 else window + 1
    half = width // 2

    padded = np.pad(values, half, mode="edge")
    kernel = np.ones(width) / width
    return np.convolve(padded, kernel, mode="valid")


def savitzky_golay(
    signal: ArrayLike, window: int, polyorder: int = 2
) -> NDArray[np.float64]:
    """Smooth a signal by local polynomial least squares.

    Fits a low-order polynomial to the samples in a window centred on each
    point and takes the fitted value at the centre. Unlike a moving average,
    which assumes the signal is locally constant, this assumes it is locally
    polynomial, so curvature and peaks survive rather than being flattened.

    A polynomial of degree at most `polyorder` passes through the filter
    unchanged, which provides an exact check on the implementation.

    Parameters
    ----------
    signal : array_like
        Input samples.
    window : int
        Window length in samples. Must be odd, positive, and greater than
        `polyorder`.
    polyorder : int, optional
        Degree of the fitted polynomial, by default 2.

    Returns
    -------
    numpy.ndarray
        Smoothed signal, same shape as `signal`.

    Raises
    ------
    ValueError
        If `window` is not odd, is not greater than `polyorder`, or exceeds
        the signal length.
    """
    values = np.asarray(signal, dtype=float)

    if window % 2 == 0:
        raise ValueError("Savitzky-Golay window length must be odd")
    if window <= polyorder:
        raise ValueError("Window length must exceed the polynomial order")
    if window > len(values):
        raise ValueError("Window length cannot exceed the signal length")

    return savgol_filter(values, window, polyorder, mode="nearest")


def kalman_filter(
    signal: ArrayLike,
    process_variance: float,
    measurement_variance: float,
) -> NDArray[np.float64]:
    """Filter a signal with a scalar Kalman filter.

    Models the underlying quantity as a random walk observed through additive
    noise::

        x[k] = x[k-1] + w,    w ~ N(0, process_variance)
        z[k] = x[k]   + v,    v ~ N(0, measurement_variance)

    At each step the estimate is predicted forward, its uncertainty grown by
    the process variance, and then corrected towards the measurement by an
    amount set by the ratio of the two variances. A large process variance
    means the quantity is believed to change quickly, so measurements are
    trusted and little smoothing occurs; a small one means the opposite.

    The filter is causal, using only samples up to the current one, so it
    introduces phase lag. This makes it usable in real time, unlike the
    centred filters in this module.

    Parameters
    ----------
    signal : array_like
        Input samples.
    process_variance : float
        Variance of the change in the underlying quantity per sample. Must be
        non-negative.
    measurement_variance : float
        Variance of the measurement noise. Must be positive.

    Returns
    -------
    numpy.ndarray
        Filtered signal, same shape as `signal`.

    Raises
    ------
    ValueError
        If `process_variance` is negative or `measurement_variance` is not
        positive.
    """
    values = np.asarray(signal, dtype=float)

    if process_variance < 0.0:
        raise ValueError("Process variance must be non-negative")
    if measurement_variance <= 0.0:
        raise ValueError("Measurement variance must be positive")

    estimate = values[0]
    variance = measurement_variance
    output = np.empty_like(values)
    output[0] = estimate

    for index in range(1, len(values)):
        variance += process_variance
        gain = variance / (variance + measurement_variance)
        estimate += gain * (values[index] - estimate)
        variance *= 1.0 - gain
        output[index] = estimate

    return output


def kalman_steady_state(
    process_variance: float, measurement_variance: float
) -> KalmanState:
    """Compute the steady-state behaviour of a scalar Kalman filter.

    The prior variance settles at the positive root of::

        p**2 - Q*p - Q*R = 0

    giving ``p = (Q + sqrt(Q**2 + 4*Q*R)) / 2`` and a steady-state gain of
    ``p / (p + R)``. Having a closed form allows the iterative implementation
    to be verified against it, and lets the filter's effective smoothing be
    stated as an equivalent time constant.

    Parameters
    ----------
    process_variance : float
        Variance of the change in the underlying quantity per sample.
    measurement_variance : float
        Variance of the measurement noise. Must be positive.

    Returns
    -------
    KalmanState
        Steady-state gain, variance and equivalent first-order time constant.

    Raises
    ------
    ValueError
        If `measurement_variance` is not positive or `process_variance` is
        negative.
    """
    if measurement_variance <= 0.0:
        raise ValueError("Measurement variance must be positive")
    if process_variance < 0.0:
        raise ValueError("Process variance must be non-negative")

    q, r = float(process_variance), float(measurement_variance)
    if q == 0.0:
        return KalmanState(gain=0.0, variance=0.0, equivalent_tau_samples=float("inf"))

    prior = 0.5 * (q + np.sqrt(q**2 + 4.0 * q * r))
    gain = prior / (prior + r)
    tau = float("inf") if gain >= 1.0 else -1.0 / np.log(1.0 - gain)

    return KalmanState(
        gain=float(gain),
        variance=float(prior * (1.0 - gain)),
        equivalent_tau_samples=float(tau),
    )


def measure_phase_lag(
    original: ArrayLike, filtered: ArrayLike, max_lag: int = 50
) -> int:
    """Estimate the lag a filter introduces, in samples.

    Cross-correlates the two signals about zero lag and returns the shift at
    which they agree best. A centred filter should return zero; a causal
    filter returns a positive shift.

    Parameters
    ----------
    original : array_like
        Signal before filtering.
    filtered : array_like
        Signal after filtering.
    max_lag : int, optional
        Largest shift considered, in samples, by default 50.

    Returns
    -------
    int
        Lag in samples, positive when `filtered` trails `original`.

    Raises
    ------
    ValueError
        If the signals differ in length or `max_lag` is not positive.
    """
    a = np.asarray(original, dtype=float)
    b = np.asarray(filtered, dtype=float)

    if a.shape != b.shape:
        raise ValueError("Signals must have the same length")
    if max_lag < 1:
        raise ValueError("Maximum lag must be positive")

    a = a - a.mean()
    b = b - b.mean()

    best_lag, best_score = 0, -np.inf
    for lag in range(0, min(max_lag, len(a) - 1) + 1):
        score = float(np.dot(a[: len(a) - lag], b[lag:]))
        if score > best_score:
            best_score, best_lag = score, lag

    return best_lag


def filter_log(log: SensorLog, method: str, **kwargs: object) -> SensorLog:
    """Return a log with its air-fuel ratio channel filtered.

    Only the air-fuel ratio is filtered. The operating point channels carry
    far less noise relative to their range, and smoothing them would blur the
    boundaries that gating relies on to identify transients.

    Parameters
    ----------
    log : SensorLog
        Measured engine data.
    method : str
        One of ``"none"``, ``"moving_average"``, ``"savitzky_golay"`` or
        ``"kalman"``.
    **kwargs
        Forwarded to the chosen filter.

    Returns
    -------
    SensorLog
        A log with the filtered air-fuel ratio channel.

    Raises
    ------
    ValueError
        If `method` is not recognised.
    """
    if method == "none":
        return log
    if method == "moving_average":
        return replace(log, afr=moving_average(log.afr, **kwargs))
    if method == "savitzky_golay":
        return replace(log, afr=savitzky_golay(log.afr, **kwargs))
    if method == "kalman":
        return replace(log, afr=kalman_filter(log.afr, **kwargs))

    raise ValueError(f"Unknown filter method: {method!r}")


FILTER_METHODS = ("none", "moving_average", "savitzky_golay", "kalman")
"""Names accepted by `filter_log`."""