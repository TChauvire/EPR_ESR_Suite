# -*- coding: utf-8 -*-
"""
Created on Tue Dec 12 13:00:51 2023

@author: tim_t
"""

from scipy.signal import argrelextrema
import numpy as np


def findnextextremum(array=np.ndarray, startidx=int, mode='max+'):
    '''
    Function used to find the next extrema (maxima or minima) of a single
    column numpy data array.

    Parameters
    ----------
    array : TYPE, numpy.ndarray
        DESCRIPTION. one column vector numpy ndarray, with a shape like (npts,)
    startidx : TYPE, integer
        DESCRIPTION. starting index from which we will look for an extrema in
        the array.
    mode : TYPE, optional, string
        DESCRIPTION. this keyword define:
            1) the direction in which the function will look for the next
                extrema: '+' mean looking forward in the array;
                         '-' mean looking backward in the array;
            2) which kind of extremum we are looking for:
                         'max' mean looking for a maximum value in the array;
                         'min' mean looking for a minimum value in the array;
        The default is 'max+'.

    Returns
    -------
    TYPE, integer
        DESCRIPTION. return the index of the next extrema

    '''
    if (array.shape[0] != np.ravel(array).shape[0]):
        raise ValueError('The array is\'t a column vector')
    array = np.ravel(array)
    if mode not in ['max+', 'min+', 'max-', 'min-']:
        return ValueError("The parameter mode should be a string type and '"
                          "adopt one of the following string sequence: 'min-',"
                          "min+','max-' or 'max+'.")
    index = None
    direction = mode[-1]
    if direction == '+':
        if mode[:-1] == 'max':
            index = argrelextrema(array[startidx:], np.greater)[0][0]+startidx
        elif mode[:-1] == 'min':
            index = argrelextrema(array[startidx:], np.less)[0][0]+startidx
    elif direction == '-':
        if mode[:-1] == 'max':
            index = argrelextrema(array[:startidx], np.greater)[0][-1]
        elif mode[:-1] == 'min':
            index = argrelextrema(array[:startidx], np.less)[0][-1]
    return index


def findclosestvalue(array=np.ndarray, InitialValue=float):
    '''
    Function used to determine corresponding value / index of a column vector 
    array to an approximate value (determine by instance in a different array
    with a different size, so the index doesn't correspond...')

    Parameters
    ----------
    array : TYPE, numpy.nd array
        DESCRIPTION. one column vector numpy ndarray, with a shape like (npts,)
    InitialValue : TYPE, float
        DESCRIPTION. Initial value in the array that we want to determine 
        the related value and index. 

    Raises
    ------
    ValueError
        DESCRIPTION. 'The array is\'t a column vector'

    Returns
    -------
    value : TYPE. float
        DESCRIPTION. value found in array the closest possible to the initial 
        value
    index : TYPE. integer
        DESCRIPTION. Index position of value in the array.

    '''
    if (array.shape[0] != np.ravel(array).shape[0]):
        raise ValueError('The array is\'t a column vector')
    array = np.ravel(array)
    j = 0
    while len(array[np.abs(array[:,]-float(InitialValue)) <= 1e-3*j]) == 0:
        j += 1
        if j > 1e6:
            raise ValueError('Indefinite while loop. Check that the '
                             'InitialValue is in the range of the array.')
            break
    value = np.squeeze(array[np.abs(array[:,]-float(InitialValue)) <= 1e-3*j])
    index = np.argwhere(array == value)[0][0]
    return value, index
