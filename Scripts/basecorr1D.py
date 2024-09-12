import numpy as np
from numpy.polynomial.polynomial import polyfit, polyvander, polyval


def basecorr1D(x=None, y=None, polyorder=0, window=20, *args, **kwargs):
    '''
    Function that achieve a baseline correction by fitting a function a the 
    edge of the data via a window parameter.
    The baseline correction is carried out via the polyfit function:
        (https://numpy.org/doc/stable/reference/generated/numpy.polynomial.polynomial.polyfit.html)
    that Compute least-squares solution to equation V(x) * c = w * y, with V the
    Vandermonde matrix, c the polynomial coefficient, w some weights parameters
    and y the data column vector.

    p(x) = c_0 + c_1 * x + ... + c_n * x^n,

    This script is freely inspired by the easyspin suite from the Stefan Stoll lab
    (https://github.com/StollLab/EasySpin/)
    (https://easyspin.org/easyspin/)

    Script written by Timothée Chauviré (https://github.com/TChauvire/EPR_ESR_Suite/), 09/09/2020

    Parameters
    ----------
    x : abscissa of the data, TYPE : numpy data array, column vector
        DESCRIPTION. The default is None.
    y : data which baseline has to be corrected, 
        TYPE : numpy data array, column vector
        It has to have the same size than x
        DESCRIPTION. The default is None.
    polyorder : order of the polynomial fitted for background subtraction
        TYPE : Integer, optional
        DESCRIPTION. The default is 0.
    window : window used at the start and the end of the data. TYPE: integer, optional
        DESCRIPTION. The default is 20.
        By instance, if window = 20, the fit of the background will be achieved on 
        np.array([y[:20],y[-20:]]).ravel()

    Returns
    -------
    ynew : baseline corrected data. 
        TYPE: numpy data array, same shape as input data y
    c : coefficient used for the polynomial fit as returned by the 
        numpy.polynomial.polynomial.polyfit function
        TYPE : tuple of real values, coefficient of the polynomial from low order to high order rank.
    error_parameters : error associated with the polynomial coefficients, 
        TYPE : tuple of real values, 
        DESCRIPTION: This error is evaluated via the function error_vandermonde() 
        by calculating the covariance matrix and diagonalize it.
    cov : Covariance matrix associated to the Vandermonde matrix of the fitted poynomial. 
        TYPE : Square Matrix 

    '''
    shape = y.shape
    if x.shape[0] != np.ravel(x).shape[0]:
        raise ValueError(
            'x must be a column vector. basecorr1D function does not work on 2D arrays.')
    else:
        x = np.ravel(x)

    if y.shape[0] != np.ravel(y).shape[0]:
        raise ValueError(
            'y must be a column vector. basecorr1D function does not work on 2D arrays.')
    else:
        y = np.ravel(y)

    if y.shape[0] != x.shape[0]:
        raise ValueError('x and y must be column vector of the same size.')

    ynew = np.full(y.shape, np.nan)
    xfit = np.array([x[:window], x[-window:]]).ravel()
    yfit = np.array([y[:window], y[-window:]]).ravel()
    c, stats = polyfit(xfit, yfit, polyorder, full=True)
    ynew = y-polyval(x, c)
    error_parameters, cov = error_vandermonde(
        x, residuals=stats[0], rank=polyorder)
    ynew = ynew.reshape(shape)
    return ynew, c, error_parameters, cov


def error_vandermonde(x, residuals=None, rank=None, *args, **kwargs):
    '''
    Function to generate 1) error estimation on parameters determined by the function
    coef, [residuals, rank, singular_values, rcond] = numpy.polynomial.polynomial.polyfit()
    2) covariance matrix associated.

    The Vandermonde matrix generated by vand = polyvander(x,rank)
    The Covariance matrix cov is obtained via the vandermonde matrix V 
    via this numerical steps:
        1) compute np.dot(V.T,V).inv 
        2) and multiply it by the residual/(nb of data points - nb of coefficients)
    The error parameters are then computed via : error_parameters = np.sqrt(np.diag(cov))

    Script written by Timothée Chauviré (https://github.com/TChauvire/EPR_ESR_Suite/), 09/09/2020

    Parameters
    ----------
    residuals : first value generated by polyfit in the list of the second output
        DESCRIPTION. float number. The default is None.
    vandermonde : Vandermonde matrix generated by vand = polyvander(x,rank)
        DESCRIPTION. The default is None.

    rank : necessary for multidimensional array
        DESCRIPTION. The default is None.

    Raises
    ------
    ValueError if rank is higher than the number of points
        "the number of data points must exceed order "
                                 "to scale the covariance matrix".

    Returns
    -------
    error_parameters : error uncertainties estimated on the parameters ordered 
    from low to high polynomial order.
    By example for a linear model f(x)=a0+a1*x : 
        error_parameters[0] = constant parameters a0, 
        error_parameters[1] = slope a1.

    cov : covariance estimated for the Least-squares fit of a polynomial to data
        return a Matrix(Rank by rank)
    '''
    if len(x) <= rank:
        raise ValueError("the number of data points must exceed order "
                         "to scale the covariance matrix")
    else:
        v = polyvander(x, rank)  # generate the vandermonde matrix
        cov = residuals/(len(x) - rank)*np.linalg.inv(np.dot(v.T, v))
        error_parameters = np.sqrt(np.diag(cov))

    return error_parameters, cov
