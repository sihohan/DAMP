from typing import Tuple

import numpy as np


def contains_constant_regions(time_series: np.ndarray, subsequence_length: int) -> bool:
    bool_vec = False
    constant_indices = np.nonzero(np.concatenate(([1], np.diff(time_series), [1])))[0]
    constant_length = np.amax(np.diff(constant_indices))
    if (constant_length >= subsequence_length) or (np.var(time_series) < 0.2):
        bool_vec = True

    return bool_vec


def MASS_V2(x=None, y=None):
    # x is the data, y is the query
    m = len(y)
    n = len(x)

    # Compute y stats -- O(n)
    meany = np.mean(y)
    sigmay = np.std(y)

    # Compute x stats
    x_less_than_m = x[: m - 1]
    divider = np.arange(1, m, dtype=float)
    cumsum_ = x_less_than_m.cumsum()
    square_sum_less_than_m = (x_less_than_m**2).cumsum()
    mean_less_than_m = cumsum_ / divider
    std_less_than_m = np.sqrt(
        (square_sum_less_than_m - (cumsum_**2) / divider) / divider
    )

    windows = np.lib.stride_tricks.sliding_window_view(x, m)
    mean_greater_than_m = windows.mean(axis=1)
    std_greater_than_m = windows.std(axis=1)

    meanx = np.concatenate([mean_less_than_m, mean_greater_than_m])
    sigmax = np.concatenate([std_less_than_m, std_greater_than_m])

    y = y[::-1]
    y = np.concatenate((y, [0] * (n - m)))

    # The main trick of getting dot products in O(n log n) time
    X = np.fft.fft(x)
    Y = np.fft.fft(y)
    Z = np.multiply(X, Y)
    z = np.fft.ifft(Z).real

    dist = 2 * (
        m - (z[m - 1 : n] - m * meanx[m - 1 : n] * meany) / (sigmax[m - 1 : n] * sigmay)
    )
    dist = np.sqrt(dist)
    return dist


def nextpow2(x: int) -> float:
    """Computes the exponent of next higher power of 2.
    MATLAB reference: https://www.mathworks.com/help/matlab/ref/nextpow2.html

    Args:
        x (int): Integer

    Returns:
        float: Exponent of next higher power of 2
    """
    return np.ceil(np.log2(np.abs(x)))


def xcorr(
    x: np.ndarray, y: np.ndarray, n_lag: int = 3000, scale: str = "coeff"
) -> Tuple[np.ndarray, np.ndarray]:
    """Computes autocorrelation with lag.
    Modified from https://stackoverflow.com/questions/43652911/python-normalizing-1d-cross-correlation.

    Args:
        x (np.ndarray): Univariate time series
        y (np.ndarray): Univariate time series
        n_lag (int): Lag
        scale (str, optional): Scaling method. Defaults to "coeff".

    Returns:
        Tuple[np.ndarray, np.ndarray]: Autocorrelation and lags
    """
    # Pad shorter array
    if x.size > y.size:
        pad_amount = x.size - y.size
        y = np.append(y, np.repeat(0, pad_amount))
    elif y.size > x.size:
        pad_amount = y.size - x.size
        x = np.append(x, np.repeat(0, pad_amount))

    lags = np.arange(-n_lag, n_lag + 1)
    corr = np.correlate(x, y, mode="same")
    i_center = len(corr) // 2
    corr = corr[i_center - n_lag : i_center + n_lag + 1]

    if scale == "biased":
        corr = corr / x.size
    elif scale == "unbiased":
        corr /= x.size - abs(lags)
    elif scale == "coeff":
        corr /= np.sqrt(np.dot(x, x) * np.dot(y, y))

    return corr, lags


def findpeaks():
    raise NotImplementedError
