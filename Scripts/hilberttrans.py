import numpy as np
from scipy.signal import hilbert

def hilberttrans(data, N=None):
    """
    Hilbert transform.
    Reconstruct imaginary data via hilbert transform.
    Parameters
    ----------
    data : numpy data array column vector
    N : int or None
        Number of Fourier components.
    Returns
    -------
    References
    ----------
    .. [1] Wikipedia, "Analytic signal".
           https://en.wikipedia.org/wiki/Analytic_signal
    .. [2] Leon Cohen, "Time-Frequency Analysis", 1995. Chapter 2.
    .. [3] Alan V. Oppenheim, Ronald W. Schafer. Discrete-Time Signal
           Processing, Third Edition, 2009. Chapter 12.
           ISBN 13: 978-1292-02572-8

    """
    shape = data.shape
    if (data.shape[0] == np.ravel(data).shape[0]): # Check if the data is a column vector
        data = np.ravel(data)
    if N == None:
        N = data.shape[-1]
    # create an empty output array
    fac = N / data.shape[-1]
    z = np.empty(data.shape, dtype=(data.flat[0] + data.flat[1] * 1.j).dtype)
    if data.ndim == 1:
        z[:] = hilbert(data.real, N)[:data.shape[-1]] * fac
    else:
        for i, vec in enumerate(data):
            z[i] = hilbert(vec.real, N)[:data.shape[-1]] * fac

    # correct the real data as sometimes it changes
    z.real = data.real
    z = z.reshape(shape)
    return z