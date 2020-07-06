# -*- coding: utf-8 -*-
"""
Created on Thu Jun  4 12:04:37 2020

@author: Tim
"""

import numpy as np
import scipy.optimize

def automatic_phase(vector=None,pivot1=1,*args,**kwargs):
    '''
    Function that adjustautomatically the phase of a complex vector by minimizing the 
    imaginary part.
    
    Parameters
    ----------
    vector : complex column vector, numpy data array
        DESCRIPTION. The default is None.
    pivot : integer, has to be lower than the number of point of the column vector
        DESCRIPTION. The default is 1.

    Returns as a first output the phase corrected complex column vector
    as a second output a dictionnary of the single angle correction employed 
    for zero-phase correction and the two angles for the first order phase correction
    -------
    See for reference : 
        - Binczyk et al. BioMedical Engineering OnLine 2015, 14(Suppl 2):S5
        doi:10.1186/1475-925X-14-S2-S5
        - Understanding NMR Spectroscopy James Keeler 2002&2004, 
        chapter 4: Fourier transformation and data processing
        - Chen et al. Journal of Magnetic Resonance 158 (2002) 164–168
        doi: 10.1016/S1090-7807(02)00069-1
        - Worley and Powers, Chemometrics and Intelligent Laboratory Systems 131 (2014) 1–6
        doi: 10.1016/j.chemolab.2013.11.005
        
    TO DO:
        1) Test different algorithm for the phase optimization
        2) Adjust other order >1 for phase correction
    '''
    dtype = vector.dtype.char
    
    if (dtype not in 'GFD' and (vector.shape[0] != np.ravel(vector).shape[0])):
         raise ValueError("The input vector doesn't have the right datatype: "
                           "Complex single column array")
    
    # PreAllocation of the output variables
    npts = int(vector.shape[0])
    zero_phase = 0
    phase_corrected_vector = np.full((npts,),np.nan) 
    phase_parameters = {}
    # Determination of the zero order phase correction analytically
    vector_real = np.real(vector)
    vector_imag = np.imag(vector)
    if npts > 10:
        zero_phase = np.mean(np.arctan2(vector_imag[0:10],vector_real[0:10]))
    else:
        raise ValueError ("The input vector is a too small column vector to adjust is phase")
    phase_corrected_vector = vector*np.exp(1j*zero_phase)
    vector_imag = np.imag(vector)
    # Calculate zero and first order phase correction
    minimum = scipy.optimize.fmin(minfunc, [zero_phase, 0.0], args=(phase_corrected_vector,pivot1))
    phi0, phi1 = minimum
    q =1j*np.pi/180
    First = -1*(np.arange(0,npts)-pivot1)/(npts)
    phase_parameters = {'zero_phase':zero_phase*180/np.pi,'first_phase':(phi0*180/np.pi, phi1*180/np.pi)}
    phase_corrected_vector = np.multiply(phase_corrected_vector,np.exp(q*(phi0+First*phi1)))
    
    return phase_corrected_vector, phase_parameters

def minfunc(phase,complex_vector,pivot1):
    '''
    Return the value minimized of the imaginary part after phase correction

    Parameters
    ----------
    complex_vector : numpy complex array
    phi0=phase[0] : zero order phase in degrees
    phi1=phase[1] : first phase in degrees
    pivot1 : pivot point around which you adjust the phase of the complex vector
    at the first-order 
    Returns zero order phase in degrees phi0 and first order phase in degrees phi1
    -------
    x : float value to minimize
    '''
    phi0 = phase[0]
    phi1 = phase[1]
    npts = int(len(complex_vector))
    q = 1j*np.pi/180
    First = -1*(np.arange(0,npts)-pivot1)/(npts)
    complex_vector_corr = np.full((npts,),np.nan) 
    complex_vector_corr = np.multiply(complex_vector,np.exp(q*(phi0+First*phi1)))
    # calculation of the integral of the imaginary part of the phase corrected signal:
    x = (np.imag(complex_vector_corr)**2).sum(axis=0)
    return x