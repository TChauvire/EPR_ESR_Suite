import numpy as np
from scipy.special import jv
 
def fieldmodulation(x=None,y=None,ModAmpl=None,Harmonic=1,*args,**kwargs):
    '''
    fieldmod: pseudo-modulated (amplitude in G in EPR/ESR magnetic resonance technique) field modulation
    
    yModInPhase = fieldmodulation(x=None,y=None,ModAmpl=None,Harmonic=1)
        
    Computes the effect of field modulation on an EPR/ESR spectrum.
    Mathematically, it employs the convolution of the fft of the signal 
    with the Bessel function of the first kind of real order and complex argument. (check scipy.special.jv)
    
    This script is freely inspired by the easyspin suite from the Stefan Stoll lab
    (https://github.com/StollLab/EasySpin/)
    (https://easyspin.org/easyspin/)
    
    Script written by Timothée Chauviré (https://github.com/TChauvire/EPR_ESR_Suite/), 09/09/2020
    
    Parameters
    ----------
    x : TYPE numpy array column vector
        DESCRIPTION : - magnetic field axis vector [mT]. 
        The default is None.
    y : TYPE numpy array column vector, x and y needs to have the same length
        DESCRIPTION : EPR absorption spectrum 
        The default is None.
    ModAmpl : Amplitude modulation (in mT), TYPE = positive float
        DESCRIPTION : peak-to-peak modulation amplitude [mT]
        The default is None.
    Harmonic : postive integer, optional
        DESCRIPTION : Harmonic of the EPR spectrum (0, 1, 2, ...)
        The default is 1 (so first derivative).

    Returns
    -------
    yModInPhase : TYPE = numpy array column vector
        DESCRIPTION : pseudo-modulated EPR spectrum.
    
    ToDo Try to adjust the phase contain in the 
    (yModInPhase, yModOutOfPhase) tuple of data to have a better signal

    References
    -------
    Berger, Gunthart, Z.Angew.Math.Phys. 13, 310 (1962)
    Wilson, J.Appl.Phys. 34, 3276 (1963)
    Haworth, Richards, Prog.Nmr.Spectrosc. 1, 1 (1966)
    Hyde et al., Appl.Magn.Reson. 1, 483-496 (1990)
    Hyde et al., J.Magn.Reson. 96, 1-13 (1992)
    Kaelin, Schweiger, J.Magn.Reson. 160, 166-180 (2003)
    Nielsen, Robinson, Conc. Magn. Reson. A 23, 38-48 (2004)
    '''
    if Harmonic < 0 or type(Harmonic) != int:
         raise ValueError('Harmonic must be a positive integer (1, 2, 3, etc)!')
    
    # Check ModAmpl
    if ModAmpl <= 0:
        raise ValueError('Modulation amplitude (3rd argument) must be positive.')
    
    # Get length of vectors.
    npts=len(x)
    if len(y) != npts:
        raise ValueError('x and y must have the same length!')
    
    
    if y.shape[0] != np.ravel(y).shape[0]:
        raise ValueError('y must be a row or column vector.')
    else: 
        y=np.ravel(y)
    
    # Compute relative base-to-peak amplitude.
    dx=x[1] - x[0]
    Ampl=float(ModAmpl / 2 / dx)
    # FFT-based convolution
    #------------------------------------------------------------
    # Compute FFT of input signal, zero negative part.
    NN=2*npts + 1
    
    ffty=np.fft.fft(y,NN)
    ffty[int(np.ceil(NN/2)+1):]=0
    # Convolution with IFT of Bessel function.
    S=(np.arange(0,NN))/ NN
    yMod=np.fft.ifft(np.multiply(ffty,jv(Harmonic,np.dot(2*np.pi*Ampl,S))))
    yMod2=yMod[0:npts]
    
    # Adjust phase.
    yMod3=np.multiply((1j)**Harmonic,yMod2)
    
    yModInPhase=np.real(yMod3)
    #yModOutOfPhase = np.imag(yMod)
    # ToDo Try to adjust the phase to have a better signal???
    return yModInPhase#,yModOutOfPhase