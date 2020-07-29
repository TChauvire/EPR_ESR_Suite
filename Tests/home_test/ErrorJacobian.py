import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import csc_matrix
from scipy.linalg import qr
from scipy.sparse.linalg import spsolve
from numpy.dual import inv

def residual(func,*parameters,x,data):
    y_model = func(x,*parameters)
    return data-y_model

def ComputeSER(func,*parameters,x,data):
    '''
    Compute the normalized residual sum of square of the residual of a function
    
    Parameters
    ----------
    func : defined function/model achieve to do the simulation
        DESCRIPTION.
    parameters : Best fit parameters of the fit
        DESCRIPTION.
    x : TYPE
        DESCRIPTION.
    data : TYPE
        DESCRIPTION.

    Returns
    -------
    SER : TYPE
        DESCRIPTION.

    '''
    y = func(x,*parameters)
    NumMeas=data.shape[0]
    NumParams=len(parameters)
    resnorm = np.sum((y-data)**2)
    SER = np.sqrt(resnorm/(NumMeas - NumParams))
    return SER

# def f(t, x, **params):

#     a = params['a']
#     c = params['c']

#     f1 = a * (x[0] - x[0] * x[1])
#     f2 = -c * (x[1] - x[0] * x[1])   

#     return np.array([f1, f2], dtype = np.float)

def ComputeJacobian(func, *params, x, eps = 1e-12):
    # Not sure of this implementation, result are weird but with good dimension...
    nparam = len(params)
    npts = len(x)
    J = np.full((npts, nparam),np.nan)
    params2 = np.full((nparam,),np.nan)
    paraminit = np.full((nparam,),np.nan)
    for (i, value) in enumerate(params):
        paraminit[i] = value
    for i in range(nparam):
        params2 =  paraminit
        params2[i] = paraminit[i]+eps
        f1 = func(x,*params2)
        params2[i] = paraminit[i]-eps      
        f2 = func(x,*params2)
        J[ : , i] = (f1 - f2) / (2 * eps)
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

def error_jacobian(SER=None,jac=None,*args,**kwargs):
    # Problem how to deal with singular matrix...
    covariance= np.dot(SER**2,np.linalg.inv(np.dot(jac.T,jac)))
    error_parameters=np.sqrt(np.diag(covariance))

    # # Other solution :
    # Q,R,P=qr(jac,mode='full',pivoting=True)
    # # R is upper triangular but not in size ==> reshaping is necessary :)
    # Rinv=inv(R[0:R.shape[-1],0:R.shape[-1]])
    # JTJInverse=np.matmul(Rinv,Rinv.T)
    # covariance=np.dot(SER ** 2,JTJInverse)
    return error_parameters,covariance

def cov_to_correlationmatrix(cov=None,*args,**kwargs):
    '''Convert covariance matrix to correlation matrix''' 
    v = np.sqrt(np.diag(cov))
    outer_v = np.outer(v, v)
    corr = cov / outer_v
    corr[cov == 0] = 0
    return corr

def stddev_covnmatrix(cov=None,*args,**kwargs):    
    std = np.sqrt(np.diag(cov))
    return std

def corr_to_covariancematrix(corr=None,std=None,*args,**kwargs):
    '''Convert correlation matrix to covariance matrix
    corr = Correlation matrix 
    std = standard deviation of each variable
    ''' 
    cov = corr*std*std.reshape(-1,1) # Calculate the covariance matrix
    return cov

def error_vandermonde(x,residuals=None,rank=None,*args,**kwargs):
    '''
    Function to generate error estimation on parameters determined by the function
    coef, [residuals, rank, singular_values, rcond] = numpy.polynomial.polynomial.polyfit() 
    Parameters
    ----------
    residuals : first value generated by polyfit in second output
        DESCRIPTION. float number. The default is None.
    vandermonde : Vandermonde matrix generated by vand = polyvander(x,rank)
        DESCRIPTION. The default is None.
    ???
    ydim : necessary for multidimensional array
        DESCRIPTION. The default is None.
    ???
    
    Raises
    ------
    ValueError if rank is higher than the number of points
        "the number of data points must exceed order "
                                 "to scale the covariance matrix".

    Returns
    -------
    error_parameters : error uncertainties estimated on the parameters ordered 
    from low to high.
    By example for a linear model f(x)=a0+a1*x : 
        error_parameters[0] = constant parameters a0, 
        error_parameters[1] = slope a1.
    error_parameters = np.sqrt(np.diag(cov))
    
    covariance : covariance estimated for the Least-squares fit of a polynomial to data
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
        v = polyvander(x,rank) # generate the vandermonde matrix
        cov = residuals/(len(x) - rank)*np.linalg.inv(np.dot(v.T, v))
        error_parameters=np.sqrt(np.diag(cov))
    
    return error_parameters,cov


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

#     bayesian analysis in python