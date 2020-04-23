import numpy as np


def strided_array(arr, win_len, step):  # Window len = L, Stride len/stepsize = S
    nrows = ((arr.size - win_len) // step) + 1
    n = arr.strides[0]
    return np.lib.stride_tricks.as_strided(arr, shape=(nrows, win_len), strides=(step * n, n))


def gini(x):
    mad = np.abs(np.subtract.outer(x, x)).mean()
    rmad = mad / np.mean(x)
    G = 0.5 * rmad
    return G


def moving_average(x, kernel, border):
    return np.convolve(x, np.ones(kernel), mode=border) / kernel
