import numpy as np
from numpy.polynomial.polynomial import polyfit, polyval
from math import prod


def basecorr2D(Spectrum=None, dimension=None, polyorder=None, *args, **kwargs):
    '''
    This function computes and applies polynomial baseline corrections 
    to the input data array Spectrum. 
    It returns the baseline corrected data (First argument of the output tuple) 
    and the polynomial baseline (Second argument of the output tuple). 
    The baseline is computed by least-squares fitting polynomials of 
    required order to the data along specified dimensions.

    Examples:
    To fit a single third-order surface to ND data, use
        cdata = basecorr2D(data,[],[3, 3, Nx]);

    To apply corrections separately along each dimension, use
        cdata = basecorr2D(data,[1, 2],[3, 3]);

    If you want to apply a linear baseline correction along the second dimension only, use
        cdata = basecorr2D(data,[2],[1]);

    To subtract the mean from the data array, use
        cdata = basecorrND(data,[],[0, 0]);

    Remark: 
        nD polynomial least-square fits are computed by constructing 
        the unsquared Vandermonde matrix associated with the problem and using 
        np.linalg.lstsq() to solve the resulting system of linear equations.

    This script is freely inspired by the easyspin suite from the Stefan Stoll lab
    (https://github.com/StollLab/EasySpin/)
    (https://easyspin.org/easyspin/)

    Script written by Timothée Chauviré (https://github.com/TChauvire/EPR_ESR_Suite/), 09/09/2020

    Parameters
    ----------
    Spectrum : Input data array, nD numpy data array
        DESCRIPTION. The default is None.

    dimension : a vector giving all dimensions along which one-dimensional 
    baseline corrections should be applied. E.g. if dimension=[2,1], then a 
    correction is applied along dimension 2, and after that another one 
    along dimension 1. If Dim is set to [], a single all-dimensional 
    baseline fit is applied instead.
    TYPE, List of integer describing the polynomial polyorder
    DESCRIPTION. The default is None.

    polyorder : orders for the polynomial fits listed in dimension, 
    so it must have the same number of elements as Dim. 
    If dimension=[], polyorder must have one entry for each dimension of Spectrum.
    TYPE, List of integer describing the polynomial order
    DESCRIPTION. The default is None.

    Returns
    -------
    CorrSpectrum : Background corrected spectrum in numpy data array format 
    of the same size than Spectrum.

    BaseLine : Baseline used for the background correction in numpy data array 
    format of the same size than Spectrum.
    '''
    # Convert input in numpy data array format
    polyorder = np.array(polyorder)
    dimension = np.array(dimension)
    Spectrum = np.array(Spectrum)
    # Check for the absence of input data
    if Spectrum.all() == None or polyorder.all() == None:
        raise ValueError(
            'basecorr() needs 3 inputs: basecorr(Spectrum,dimension,polyorder)')
    if polyorder.any() < 0 or polyorder.dtype.char not in 'bBhHiIlLqQpP':
        raise ValueError('polyorder must contain positive integers!')
    # Check for the size of the input data, data must be 2D only.
    ndims = Spectrum.ndim
    if Spectrum.shape[0] == np.ravel(Spectrum).shape[0]:
        raise ValueError('Multidimensional fitting not possible for 1D data!')
    if ndims > 2:
        raise ValueError(
            'Multidimensional fitting for {0}-D arrays not implemented!'.format(str(ndims)))
    # if polyorder.size != ndims:
    #     raise ValueError('polyorder must have {0} elements!'.format(str(ndims)))
    if np.max(polyorder) >= np.max(Spectrum.shape):
        raise ValueError(
            'The polyorder value is too large for the given Spectrum!')
    CorrSpectrum = np.zeros(Spectrum.shape)
    m, n = Spectrum.shape
    # If dimension is None, baseline correction is achieved uniformly via surface fitting
    if dimension.all() == None or dimension.size == 0:
        BaseLine = np.zeros((m*n,))
        x = np.tile(np.linspace(1, m, m), (n, 1)).T
        x = np.ravel(x)
        y = np.tile(np.linspace(1, n, n), (m, 1))
        y = np.ravel(y)
        q = 0
        if polyorder.size == 1:
            polyorder = np.append(polyorder, polyorder[0])
        Ny = len(range(polyorder[1]+1))*len(range(polyorder[0]+1))
        D = np.full((m*n, Ny), np.nan)
        for j in range(polyorder[1]+1):
            for i in range(polyorder[0]+1):
                # Construct in each column the Vandermonde Matrix
                # with the x**i * y**j monomial vector
                D[:, q] = np.multiply(x ** i, y ** j)
                q = q+1
        # Solve least squares problem
        C, _, _, _ = np.linalg.lstsq(D, np.ravel(Spectrum), rcond=None)
        print(C.shape)
        for q in range(Ny):
            BaseLine = BaseLine + C[q]*D[:, q]
        BaseLine = BaseLine.reshape((m, n))
    else:
        # If dimension is given, multiple baseline correction is achieved
        # uniformly slice by slice, dimension by dimension.
        # One-dimensional fits along columns (dim=1) or rows (dim=2)
        if polyorder.size != dimension.size:
            raise ValueError(
                'polyorder and dimension must have the same number of elements!')
        for i in dimension:
            if i > ndims or i < 0:  # >=
                raise ValueError(
                    'dimension index out of range! The elements in the list must be 0 or 1 for 2D datatype.')

        for q in range(len(dimension)):
            d = dimension[q]
            if polyorder[q] >= Spectrum.shape[d-1]:
                raise ValueError('dimension {0} has {1} points. Polynomial'
                                 'polyorder {2} not possible. Check polyorder and'
                                 'dimension!'.format(str(d), str(Spectrum.shape[d-1]), str(polyorder[q])))
            # Automatizing the creation of the tuple to manage array transposition
            # Transpose if needed to move dimension of interest along columns
            if d == 1:
                thisSpectrum = np.transpose(Spectrum)
            else:
                thisSpectrum = Spectrum
            m2, n2 = thisSpectrum.shape
            x = np.linspace(1, n2, n2)
            Ny = len(range(polyorder[0]+1))
            D = np.full((n2, Ny), np.nan)
            for i in range(polyorder[0]+1):
                # Construct in each column the Vandermonde Matrix
                # with the x**i * y**j monomial vector
                D[:, i] = x ** i
            thisBaseLine = np.zeros(thisSpectrum.shape)
            C, _, _, _ = np.linalg.lstsq(D, thisSpectrum.T, rcond=None)
            thisBaseLine = np.dot(C.T, D.T)
            BaseLine = np.zeros((m, n))
            if d == 1:
                BaseLine = BaseLine+np.transpose(thisBaseLine)
            else:
                BaseLine = thisBaseLine
    CorrSpectrum = Spectrum - BaseLine
    return CorrSpectrum, BaseLine
