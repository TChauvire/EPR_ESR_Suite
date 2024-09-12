# -*- coding: utf-8 -*-
"""
Computes the derivate of the input vector.
   If x is not given, x is assumed to be 1:length(y).
   If y is a matrix, differentiation occurs along columns.

This script is freely inspired by the easyspin suite from the Stefan Stoll lab
(https://github.com/StollLab/EasySpin/)
(https://easyspin.org/easyspin/)

Script written by Timothée Chauviré 
(https://github.com/TChauvire/EPR_ESR_Suite/) Oct, 23rd 2023

Probably useless as numpy.gradient do the job for you !
@author: tim_t
"""

import numpy as np


def deriv(y=None, x=None, *arg, **kwarg):
    if (y.shape[0] != np.ravel(y).shape[0]):
        raise ValueError('The file data must be a numpy data array column'
                         ' vector!')
    elif y == None:
        raise ValueError('No arguments has been specified !')
    elif x == None:
        x = []
    elif x != None:

        return


case 1
y = varargin{1}
x = []
case 2
x = varargin{1}
y = varargin{2}
otherwise
error('Wrong number of input arguments!')
end

RowVector = (numel(y) == size(y, 2))
if (RowVector), y = y(:); end

if isempty(x)
x = 1: size(y, 1)
end

dydx = np.diff(y)/np.tile(np.diff(x[:,]), (1, np.size(y, axis=1)))
# dydx = (dydx([1 1:end], : )+dydx([1:end end], : ))/2 # To translate

if (RowVector), dydx = dydx.'
end

return dydx
