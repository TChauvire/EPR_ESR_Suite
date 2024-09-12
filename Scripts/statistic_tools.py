import numpy as np
# from scipy.sparse import csc_matrix
# from scipy.linalg import qr
# from scipy.sparse.linalg import spsolve
# from numpy.dual import inv


def residual(func, *parameters, x, data):
    '''
    Evaluate the residual value of a fit compare to experimental value x and data.

    Script written by Timothée Chauviré (https://github.com/TChauvire/EPR_ESR_Suite/), 09/09/2020

    Parameters
    ----------
    func : user defined function/model to do the simulation
        TYPE : Python mathematical function definition.
    parameters : Best fit parameters of the user fitted function
        TYPE: Tuple of values
    x : experimental abscissa axis
        TYPE : Numpy data array
    data : experimental data
        TYPE : Numpy data array

    Returns
    -------
    residual : residual spectrum of a fitted spectrum compared to an experimental spectrume
        TYPE : Numpy data array

    '''
    y_model = func(x, *parameters)
    return data-y_model


def ComputeSER(func, *parameters, x, data):
    '''
    Compute the normalized residual sum of square of the residual of a function

    Script written by Timothée Chauviré (https://github.com/TChauvire/EPR_ESR_Suite/), 09/09/2020

    Parameters
    ----------
    func : user defined function/model to do the simulation
        TYPE : Python mathematical function definition.
    parameters : Best fit parameters of the user fitted function
        TYPE: Tuple of values
    x : experimental abscissa axis
        TYPE : Numpy data array
    data : experimental data
        TYPE : Numpy data array

    Returns
    -------
    SER : normalized residual sum of square
        TYPE : real float value

    '''
    y = func(x, *parameters)
    NumMeas = data.shape[0]
    NumParams = len(parameters)
    resnorm = np.sum((y-data)**2)
    SER = np.sqrt(resnorm/(NumMeas - NumParams))
    return SER


def ComputeJacobian(func, *params, x, eps=1e-12):
    '''
    Compute the Jacobian of a function.

    Script written by Timothée Chauviré (https://github.com/TChauvire/EPR_ESR_Suite/), 09/09/2020

    Parameters
    ----------
    func : user defined function/model to do the simulation
        TYPE : python function definition with parameters *params
        DESCRIPTION.
    *params : Best fit parameters of the user fitted function
        TYPE: Tuple of values
    x : abscissa axis associated to the experimental data to be fitted.
        TYPE : Numpy data array
    eps : precision
        TYPE= real float, optional
        DESCRIPTION. The default is 1e-12.

    Returns
    -------
    J : jacobian matrix
        TYPE : Numpy data array

    '''

    # Not sure of this implementation, result are weird but with good dimension...
    nparam = len(params)
    npts = len(x)
    J = np.full((npts, nparam), np.nan)
    params2 = np.full((nparam,), np.nan)
    paraminit = np.full((nparam,), np.nan)
    for (i, value) in enumerate(params):
        paraminit[i] = value
    for i in range(nparam):
        params2 = paraminit
        params2[i] = paraminit[i]+eps
        f1 = func(x, *params2)
        params2[i] = paraminit[i]-eps
        f2 = func(x, *params2)
        J[:, i] = (f1 - f2) / (2 * eps)
    return J


# def ComputeJacobian(func,*parameters, x, dx=1e-8):
#     npts = len(x)
#     nparam = len(parameters)
#     jac = np.full((nparam, nparam),np.nan)
#     for i in range(nparam):  # through columns to allow for vector addition
#         for j in range(npts):
#             Dxj = (abs(x[j])*dx if x[j] != 0 else dx)
#             x_plus = np.array([(xi if k != j else xi + Dxj) for k, xi in enumerate(x)])
#             jac[:, j] = (func(x_plus,*parameters) - func(x,*parameters))/Dxj
#     return jac

def error_jacobian(SER=None, jac=None, *args, **kwargs):
    '''
    Compute the error on parameters associated to non linear fit based on Jacobian matrix.

    Script written by Timothée Chauviré (https://github.com/TChauvire/EPR_ESR_Suite/), 09/09/2020

    Parameters
    ----------
    SER : normalized residual sum of square
        TYPE : Numpy data array    
        DESCRIPTION. The default is None.
    jac : jacobian matrix, 
        TYPE : Numpy data array
        DESCRIPTION. The default is None.

    Returns
    -------
    error_parameters : error evaluation of the fitted parameters of a fit
        TYPE : Numpy data array 

    covariance : covariance matrix
        TYPE : Numpy data array 

    '''

    # Problem how to deal with singular matrix...
    covariance = np.dot(SER**2, np.linalg.inv(np.dot(jac.T, jac)))
    error_parameters = np.sqrt(np.diag(covariance))

    # # Other solution :
    # Q,R,P=qr(jac,mode='full',pivoting=True)
    # # R is upper triangular but not in size ==> reshaping is necessary :)
    # Rinv=inv(R[0:R.shape[-1],0:R.shape[-1]])
    # JTJInverse=np.matmul(Rinv,Rinv.T)
    # covariance=np.dot(SER ** 2,JTJInverse)
    return error_parameters, covariance


def cov_to_correlationmatrix(cov=None, *args, **kwargs):
    '''
    Convert covariance matrix to correlation matrix

    Script written by Timothée Chauviré (https://github.com/TChauvire/EPR_ESR_Suite/), 09/09/2020

    Parameters
    ----------
    cov = covariance matrix
        TYPE : Numpy data array
    Returns
    -------
    corr : correlation matrix
        TYPE : Numpy data array
    '''
    v = np.sqrt(np.diag(cov))
    outer_v = np.outer(v, v)
    corr = cov / outer_v
    corr[cov == 0] = 0
    return corr


def stddev_covnmatrix(cov=None, *args, **kwargs):
    '''
    Evaluate standard deviation of fitted parameters from covariance matrix from .

    Script written by Timothée Chauviré (https://github.com/TChauvire/EPR_ESR_Suite/), 09/09/2020

    Parameters
    ----------
    cov : covariance matrix, 
        TYPE : Numpy data array
        DESCRIPTION. The default is None.

    Returns
    -------
    std : standard deviation associated to parameters obtained via a fit of experimental value.
        TYPE : Numpy data array
    '''
    std = np.sqrt(np.diag(cov))
    return std


def corr_to_covariancematrix(corr=None, std=None, *args, **kwargs):
    '''Convert correlation matrix to covariance matrix

    Script written by Timothée Chauviré (https://github.com/TChauvire/EPR_ESR_Suite/), 09/09/2020

    Parameters
    ----------
    corr = Correlation matrix
        TYPE : Numpy data array
    std = standard deviation of each variable
        TYPE : Numpy data array

    Returns
    -------
    cov : covariance matrix, 
        TYPE : Numpy data array
    '''
    cov = corr*std*std.reshape(-1, 1)  # Calculate the covariance matrix
    return cov


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

    from numpy.polynomial.polynomial import polyvander

    if len(x) <= rank:
        raise ValueError("the number of data points must exceed order "
                         "to scale the covariance matrix")
        # note, this used to be: fac = resids / (len(x) - order - 2.0)
        # it was deciced that the "- 2" (originally justified by "Bayesian
        # uncertainty analysis") is not was the user expects
        # (see gh-11196 and gh-11197)
    else:
        v = polyvander(x, rank)  # generate the vandermonde matrix
        cov = residuals/(len(x) - rank)*np.linalg.inv(np.dot(v.T, v))
        error_parameters = np.sqrt(np.diag(cov))

    return error_parameters, cov


# Ideas To Do or test:
#     r,p = scipy.stats.pearsonr(x, y)  # and other optional statistical test
#     reference https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.pearsonr.html#scipy.stats.pearsonr
#     https://en.wikipedia.org/wiki/Pearson_correlation_coefficient

#     cov = np.cov(x,y)

# Code from scipy.optimize.curve_fit: (https://github.com/scipy/scipy/blob/v1.5.1/scipy/optimize/minpack.py#L532-L834)
# Do Moore-Penrose inverse discarding zero singular values.
#         _, s, VT = svd(res.jac, full_matrices=False)
#         threshold = np.finfo(float).eps * max(res.jac.shape) * s[0]
#         s = s[s > threshold]
#         VT = VT[:s.size]
#         pcov = np.dot(VT.T / s**2, VT)

# Check lmfit examples :
#    https://github.com/lmfit/lmfit-py/blob/master/examples/example_fit_with_derivfunc.py

#     bayesian analysis in python (chech= Deer Analysis module DeerLab.py)
