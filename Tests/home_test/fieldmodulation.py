import numpy as np
from scipy.special import jv
 
def fieldmodulation(x=None,y=None,ModAmpl=None,Harmonic=1,*args,**kwargs):       
    '''
        # fieldmod  field modulation
    
    #   yMod = fieldmod(x,y,ModAmpl);
    #   yMod = fieldmod(x,y,ModAmpl,Harmonic);
    #   fieldmod(...)
    
    #   Computes the effect of field modulation
    #   on an EPR absorption spectrum.
    
    #   Input:
        #   - x: magnetic field axis vector [mT]
        #   - y: absorption spectrum
        #   - ModAmpl: peak-to-peak modulation amplitude [mT]
        #   - Harmonic: harmonic (0, 1, 2, ...); default is 1
    
    #   Output:
        #   - yMod: pseudo-modulated spectrum
    
    #   If no output variable is given, fieldmod plots the
    #   original and the modulated spectrum.

    #   Example:
    
    #     x = linspace(300,400,1001);
    #     y = lorentzian(x,342,4);
    #     fieldmod(x,y,20);
    
    # References
    # --------------------------------------------------
    # Berger, Günthart, Z.Angew.Math.Phys. 13, 310 (1962)
    # Wilson, J.Appl.Phys. 34, 3276 (1963)
    # Haworth, Richards, Prog.Nmr.Spectrosc. 1, 1 (1966)
    # Hyde et al., Appl.Magn.Reson. 1, 483-496 (1990)
    # Hyde et al., J.Magn.Reson. 96, 1-13 (1992)
    # Kaelin, Schweiger, J.Magn.Reson. 160, 166-180 (2003)
    # Nielsen, Robinson, Conc. Magn. Reson. A 23, 38-48 (2004)

    Parameters
    ----------
    x : TYPE, optional
        DESCRIPTION. The default is None.
    y : TYPE, optional
        DESCRIPTION. The default is None.
    ModAmpl : TYPE, optional
        DESCRIPTION. The default is None.
    Harmonic : TYPE, optional
        DESCRIPTION. The default is 1.
    *args : TYPE
        DESCRIPTION.
    **kwargs : TYPE
        DESCRIPTION.

    Raises
    ------
    ValueError
        DESCRIPTION.

    Returns
    -------
    yModInPhase : TYPE
        DESCRIPTION.
    yModOutOfPhase : TYPE
        DESCRIPTION.

    '''
    if Harmonic < 0 or type(Harmonic) == int:
         raise ValueError('Harmonic must be a positive integer (1, 2, 3, etc)!')
    
    # Check ModAmpl
    if ModAmpl <= 0:
        raise ValueError('Modulation amplitude (3rd argument) must be positive.')
    
    # Get length of vectors.
    npts=len(x)
    if len(y) != npts:
        raise ValueError('x and y must have the same length!')
    
    y=np.ravel(y)
    if len(y) != npts:
        raise ValueError('y must be a row or column vector.')
    
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
    
    return yModInPhase#,yModOutOfPhase