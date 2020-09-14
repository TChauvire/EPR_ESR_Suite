import numpy as np

def fdaxis(TimeAxis=None,*args,**kwargs):
    '''
    fdaxis, Frequency domain axis 
    -----------------------------
    Returns a vector FreqAxis containing the frequency-
    domain axis of the FFT of a N-point time-
    domain vector TimeAxis sampled with period dT.
    
    This script is freely inspired by the easyspin suite from the Stefan Stoll lab
    (https://github.com/StollLab/EasySpin/)
    (https://easyspin.org/easyspin/)
    
    Script written by Timothée Chauviré (https://github.com/TChauvire/EPR_ESR_Suite/), 09/09/2020

    Parameters
    ----------
    TimeAxis : time-domain vector TimeAxis, TYPE numpy 1D data array, optional
        DESCRIPTION. The default is None.
    
    Returns
    -------
    FreqAxis : Frequency axis associated to the input time axis after Fourier Transformation.
        DESCRIPTION.

    '''
    if (TimeAxis.shape[0] != np.ravel(TimeAxis).shape[0]):
        raise ValueError('The file data must be a column vector')
    else:
        TimeAxis = np.ravel(TimeAxis)
    dT=TimeAxis[1] - TimeAxis[0]
    N=len(TimeAxis)      
    NyquistFrequency=1/(2*dT)
    UnitAxis=np.multiply(2/N,np.arange(0,N)-np.fix(N/2))
    FreqAxis=NyquistFrequency*UnitAxis
    return FreqAxis