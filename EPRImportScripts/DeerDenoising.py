# -*- coding: utf-8 -*-
"""
Created on Tue Oct 31 12:33:32 2023

Denoising scripts for distance distribution (4P-DEER ...) analysis.

This set of packages was initially written under Matlab by Maddhur Srivastava
(ms2736@cornell.edu). It was then adapted by Chris Myer, staff at
CAC at Cornell (http://www.cac.cornell.edu/).
So the original software working with a Bokh graphical user interface is found
at https://github.com/CornellCAC/denoising.
And you may want the access to the orginal more advanced software hosted at
this address : https://denoising.cornell.edu/

For more information about the techniques, check the different paper from
the litterature:
    - Madhur Srivastava, Elka R. Georgieva, and Jack H. Freed
    (dx.doi.org/10.1021/acs.jpca.7b00183)
    J. Phys. Chem. A 2017, 121, 12, 2452–2465

@author: Timothee Chauvire (tsc84@cornell.edu)
(https://github.com/TChauvire/EPR_ESR_Suite)

(https://ataspinar.com/2018/12/21/a-guide-for-using-the-wavelet-transform-in-
 machine-learning/)
"""

import numpy as np
import pywt
from copy import deepcopy
# thresholds = 3


def ThresholdGuess(noiselevel):
    if noiselevel <= 0.002:
        threshold = 2
    else:
        threshold = 5
    return threshold
    # Check if the value of maxlevel is appropriate
    # if maxlevel < 1 or maxlevel > N+1:
    #     raise ValueError('maxlevel has to be at least and inferior to the'
    #                  'maximum level of wavelet decomposition, here:'
    #                  '{N+1}'.format())


def Deer_Denoising(x_x, x_y, wavename='db6', mode='periodic'):
    raw_signal = x_y.T
    npts = len(x_x)
    signal = np.flipud(raw_signal)
    N = int(np.floor(np.log2(len(signal))))
    coeffs = wavedec(signal, wavename=wavename, mode=mode,
                     level=N, trim_approx=True)
    # thresholds = np.zeros(N+1)
    thresholds = np.zeros(N+1)
    coeffdict = {}
    denoised_signal_dict = {}
    for i in range(N):
        size = len(coeffs[-i])
        thresholds[-i] = size
        new_coeffs = apply_thresholding(coeffs, thresholds)
        coeffdict.update({str(i): new_coeffs})
        dsignal, dcoeffs = generate_denoised_signal(new_coeffs, thresholds,
                                                    wavename, npts)
        denoised_signal_dict.update({str(i): dsignal})
    return denoised_signal_dict, coeffdict, thresholds


def generate_denoised_signal(coeffs, thresholds, wavename, npts):
    denoised_coeffs = apply_thresholding(coeffs, thresholds)
    denoised_signal = pywt.waverec(denoised_coeffs, wavename)[0:npts]
    # for weird reason, pywt.waverec output has not the same size than the
    # original data... so we have to truncate the data.
    return denoised_signal, denoised_coeffs


def apply_thresholding(coeffs, thresholds):
    coeffs2 = deepcopy(coeffs)
    for level in range(len(thresholds)):
        if int(thresholds[level]) > 0.:
            for j in range(int(thresholds[level])):
                coeffs2[level][j] = 0
                coeffs2[level][j] = 0
    return coeffs2
# def apply_thresholding(coeffs, thresholds):
#     tcoeffs = [c.copy() for c in coeffs]
#     for i in range(1, int(len(tcoeffs))):
#         tcoeffs = apply_detail_thresholding(tcoeffs, i, thresholds[-i])
#     return tcoeffs


# def apply_detail_thresholding(coeffs2, level, threshold):
#     if int(threshold) > 0.:
#         upper_bound = int(min(len(coeffs2[-level]), int(threshold)+1))
#         #indices_to_zero = range(0, upper_bound)
#         for i in range(upper_bound):
#             coeffs2[-level][0][i] = 0.
#             coeffs2[-level][1][i] = 0.
#     return coeffs2


def wavedec(data, wavename, mode='symmetric', level=None, axis=-1,
            trim_approx=True):
    """
    Multilevel 1D Discrete Wavelet Transform of data.

    Parameters
    ----------
    data: array_like
        Input data
    wavelet : Wavelet object or name string
        Wavelet to use
    mode : str, optional
        Signal extension mode, see :ref:`Modes <ref-modes>`.
    level : int, optional
        Decomposition level (must be >= 0). If level is None (default) then it
        will be calculated using the ``dwt_max_level`` function.
    axis: int, optional
        Axis over which to compute the DWT. If not given, the
        last axis is used.

    Returns
    -------
    [cA_n, cD_n, cD_n-1, ..., cD2, cD1] : list
        Ordered list of coefficients arrays
        where ``n`` denotes the level of decomposition. The first element
        (``cA_n``) of the result is approximation coefficients array and the
        following elements (``cD_n`` - ``cD_1``) are details coefficients
        arrays.

    Examples
    --------
    >>> from pywt import wavedec
    >>> coeffs = wavedec([1,2,3,4,5,6,7,8], 'db1', level=2)
    >>> cA2, cD2, cD1 = coeffs
    >>> cD1
    array([-0.70710678, -0.70710678, -0.70710678, -0.70710678])
    >>> cD2
    array([-2., -2.])
    >>> cA2
    array([  5.,  13.])

    """
    data = np.asarray(data)

    wavelet = pywt._utils._as_wavelet(wavename)
    try:
        axes_shape = data.shape[axis]
    except IndexError:
        raise ValueError("Axis greater than data dimensions")
    level = pywt._multilevel._check_level(axes_shape, wavelet.dec_len, level)

    coeffs_list = []

    a = data
    for i in range(level):
        a, d = pywt.dwt(a, wavelet, mode, axis)
        if trim_approx:
            coeffs_list.append(d)
        else:
            coeffs_list.append([a.copy(), d])

    if trim_approx:
        coeffs_list.append(a)
    coeffs_list.reverse()

    return coeffs_list
