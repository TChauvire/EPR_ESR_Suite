import numpy as np
from scipy.signal import lfilter

def rcfilt(y=None,SampTime=None,TimeConstant=None,UpDown='up',*args,**kwargs):
    '''
    Filters a spectrum using a RC low-pass filter as built into 
    cw spectrometers to remove high-frequency noise.

    Parameters
    ----------
    y : Type. numpy data_array
        DESCRIPTION. unfiltered spectrum,The default is None. 
    SampTime : float number, optional
        DESCRIPTION. time constant of the filter, The default is None.
    TimeConstant : float number, optional
        DESCRIPTION. time constant of the filter
        The default is None.
    UpDown : string argument, optional
        DESCRIPTION. 'up' or 'down' defines the direction of the
        field sweep. If omitted, 'up' is assumed.
        The default is 'up'.
    
    SampleTime and TimeConstant must have the same units (s, ms, us, ...)
    
    Returns
    -------
    varargout : TYPE
        DESCRIPTION.
    '''
    if np.array(TimeConstant).dtype.char not in 'bBhHiIlLqQpPdefg':
        raise ValueError('Time constant must be real!')
    if np.array(SampTime).dtype.char not in 'bBhHiIlLqQpPdefg':
        raise ValueError('SampTime must be real!')
    
    if np.array(TimeConstant).size != 1:
        raise ValueError('Time constant must be a scalar!')
    if np.array(SampTime).size != 1:
        raise ValueError('SampTime must be a scalar!')
    
    if (TimeConstant <= 0):
        raise ValueError('Time constant must be positive or zero!')
    if (SampTime <= 0):
        raise ValueError('SampTime must be positive!')
   
    m,n=y.shape
    yFiltered = np.full((m,n),np.nan)
    if y.shape[0] == np.ravel(y).shape[0]:
        y=y.reshape((m,1))
    
    if UpDown in ['down','dn']:
        y=y[::-1]
    
    if TimeConstant == 0:
        e=0
    else:
        e=np.exp(-SampTime/TimeConstant)
    
    for ii in range(n):
        yFiltered[:,ii]=lfilter(np.array([1-e]),np.array([1,- e]),y[:,ii],axis=0)
    
    if UpDown in ['down','dn']:
        y=y[::-1]
        yFiltered=yFiltered[::-1]
    
    if y.shape[0] == np.ravel(y).shape[0]:
        y=y.reshape((m,n))
        yFiltered=yFiltered.reshape((m,n))
       
    return yFiltered