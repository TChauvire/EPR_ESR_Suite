# -*- coding: utf-8 -*-
"""
Created on Tue Jul 21 15:45:23 2020

@author: TC229401
"""
from scipy import sparse
import numpy as np

def baseline_correction_als(y, lam, p, niter=10):
    '''
    "Asymmetric Least Squares Smoothing" by P. Eilers and H. Boelens in 2005

    Parameters
    ----------
    y : TYPE
        DESCRIPTION.
    lam : TYPE
        DESCRIPTION.
    p : TYPE
        DESCRIPTION.
    niter : TYPE, optional
        DESCRIPTION. The default is 10.

    Returns
    -------
    z : TYPE
        DESCRIPTION.

    '''
  # L = len(y)
  # D = sparse.csc_matrix(np.diff(np.eye(L), 2))
  # w = np.ones(L)
  # for i in xrange(niter):
  #   W = sparse.spdiags(w, 0, L, L)
  #   Z = W + lam * D.dot(D.transpose())
  #   z = spsolve(Z, w*y)
  #   w = p * (y > z) + (1-p) * (y < z)
  
    L = len(y)
    D = sparse.diags([1,-2,1],[0,-1,-2], shape=(L,L-2))
    D = lam * D.dot(D.transpose()) # Precompute this term since it does not depend on `w`
    w = np.ones(L)
    W = sparse.spdiags(w, 0, L, L)
    for i in range(niter):
        W.setdiag(w) # Do not create a new matrix, just update diagonal values
        Z = W + D
        z = sparse.linalg.spsolve(Z, w*y)
        w = p * (y > z) + (1-p) * (y < z)
    return z