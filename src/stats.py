import numpy as np

def freedman_diaconis_bins(x):
    """
    Return the optimal number of histogram bins using
    the Freedman–Diaconis rule.

    Parameters
    ----------
    x : array-like
        1D array of samples.

    Returns
    -------
    int
        Number of bins.
    """
    x = np.asarray(x).ravel()
    n = x.size
    if n < 2:
        return 1

    # Interquartile range
    q75, q25 = np.percentile(x, [75, 25])
    iqr = q75 - q25
    if iqr == 0:
        return 1

    # Bin width
    bw = 2 * iqr / np.cbrt(n)
    if bw == 0:
        return 1

    data_range = x.max() - x.min()
    return int(np.ceil(data_range / bw))


def histogram_cdf(n, b):
    """
    Compute the cumulative distribution function from a normalized histogram.

    Parameters
    ----------
    n : array-like
        Histogram bin heights (density=True).
    b : array-like
        Bin edges, shape (len(n)+1,).

    Returns
    -------
    cdf : ndarray
        CDF evaluated at each bin edge, shape (len(n)+1,).
        cdf[0] = 0, cdf[-1] = 1.
    """
    n = np.asarray(n)
    b = np.asarray(b)

    widths = b[1:] - b[:-1]
    cdf = np.zeros_like(b, dtype=float)
    cdf[1:] = np.cumsum(n * widths)
    return cdf