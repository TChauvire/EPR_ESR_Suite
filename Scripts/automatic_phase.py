import numpy as np
import scipy.optimize
from warnings import warn

def automatic_phase(vector=None,pivot1=1,funcmodel='minfunc', *args,**kwargs):
    '''
    Function that adjust automatically the phase of a complex vector by minimizing the 
    imaginary part.
    
    Script written by Timothée Chauviré (https://github.com/TChauvire/EPR_ESR_Suite/), 09/09/2020
    
    Parameters
    ----------
    vector : complex column vector, numpy data array
        DESCRIPTION. The default is None.
    pivot : integer, has to be lower than the number of point of the column vector
        DESCRIPTION. The default is 1.
    funcmodel : Model function employed for the minimization procedure. 
        Two options are available: 
        'minfunc' : Automated phase Correction based on Minimization of the Imaginary Part
        'acme' : Automated phase Correction based on Minimization of Entropy 
        (ref: Chen Li et al. Journal of Magnetic Resonance 158 (2002) 164-168)
        DESCRIPTION. The default is 'minfunc'.
        
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
        1) Test other algorithms for the phase optimization (ACME and min implemented for the moment)
        2) Adjust other order >1 for phase correction
    '''
    dtype = vector.dtype.char
    shape = vector.shape
    if (dtype not in 'GFD' and (vector.shape[0] != np.ravel(vector).shape[0])):
         raise ValueError("The input vector doesn't have the right datatype: "
                           "Complex single column array")
    else:
        vector = np.ravel(vector)
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
    if funcmodel == 'minfunc':
        minimum = scipy.optimize.fmin(minfunc, [zero_phase, 0.0], args=(phase_corrected_vector,pivot1),maxiter=1000,maxfun=1000)
    elif funcmodel == 'acme':
        minimum = scipy.optimize.fmin(acme, [zero_phase, 0.0], args=(phase_corrected_vector,pivot1),maxiter=1000,maxfun=1000)
    else:
        warn("There is only two options for the funcmodel: 'minfunc' and 'acme'." 
             "By default, 'minfunc' was used.")
        minimum = scipy.optimize.fmin(minfunc, [zero_phase, 0.0], args=(phase_corrected_vector,pivot1))
    phi0, phi1 = minimum
    q =1j*np.pi/180
    First = -1*(np.arange(0,npts)-pivot1)/(npts)
    phase_parameters = {'zero_phase':zero_phase*180/np.pi,'first_phase':(phi0*180/np.pi, phi1*180/np.pi)}
    phase_corrected_vector = np.multiply(phase_corrected_vector,np.exp(q*(phi0+First*phi1)))
    phase_corrected_vector = phase_corrected_vector.reshape(shape)
    return phase_corrected_vector, phase_parameters

def minfunc(phase,complex_vector,pivot1):
    '''
    Phase correction fucntion using minimization method of the imaginary part as algorithm.
    
    Script written by Timothée Chauviré (https://github.com/TChauvire/EPR_ESR_Suite/), 09/09/2020

    Parameters
    ----------
    phase : tuple, phi0=phase[0] : zero order phase in degrees
                   phi1=phase[1] : first phase in degrees
    complex_vector : numpy complex array
    pivot1 : pivot point around which you adjust the phase of the complex vector
    at the first-order 
    Returns zero order phase in degrees phi0 and first order phase in degrees phi1
    
    Output
    -------
    x : float value to minimize
    '''
    phi0, phi1 = phase
    npts = int(len(complex_vector))
    q = 1j*np.pi/180
    First = -1*(np.arange(0,npts)-pivot1)/(npts)
    complex_vector_corr = np.full((npts,),np.nan) 
    complex_vector_corr = np.multiply(complex_vector,np.exp(q*(phi0+First*phi1)))
    # calculation of the integral of the imaginary part of the phase corrected signal:
    x = (np.imag(complex_vector_corr)**2).sum(axis=0)
    return x

def acme(phase, complex_vector, pivot1):
    """
    Phase correction using ACME algorithm by Chen Li et al.
    Journal of Magnetic Resonance 158 (2002) 164-168
    
    Parameters
    -------
    phase : tuple, phi0=phase[0] : zero order phase in degrees
                   phi1=phase[1] : first phase in degrees
    complex_vector : numpy complex array
    pivot1 : pivot point around which you adjust the phase of the complex vector
    at the first-order 
    Returns zero order phase in degrees phi0 and first order phase in degrees phi1
    
    Output
    -------
    x : float value to minimize
        Value of the objective function (phase score)
    """
    stepsize = 1
    npts = int(len(complex_vector))
    phi0, phi1 = phase
    q = 1j*np.pi/180
    First = -1*(np.arange(0,npts)-pivot1)/(npts)
    s0 = np.multiply(complex_vector, np.exp(q*(phi0+First*phi1)))
    data = np.real(s0)

    # Calculation of first derivatives
    ds1 = np.abs((data[1:]-data[:-1]) / (stepsize*2))
    p1 = ds1 / np.sum(ds1)

    # Calculation of entropy
    p1[p1 == 0] = 1

    h1 = -p1 * np.log(p1)
    h1s = np.sum(h1)

    # Calculation of penalty
    pfun = 0.0
    as_ = data - np.abs(data)
    sumas = np.sum(as_)

    if sumas < 0:
        pfun = pfun + np.sum((as_/2) ** 2)

    p = 1000 * pfun

    return (h1s + p) / data.shape[-1] / np.max(data)