import numpy as np
from scipy.signal import lfilter, savgol_filter
from scipy.linalg import pascal

def datasmooth(y=None,window_length=1,method='binom',polyorder=2,deriv=0,*args,**kwargs):
    '''
    Parameters
    ----------
    y : TYPE, optional
        DESCRIPTION. The default is None.
    window_length : TYPE, optional
        DESCRIPTION. The default is 3.
    method : TYPE, optional
        DESCRIPTION. The default is 'binom'.
    polyorder : TYPE, optional
        DESCRIPTION. The default is 2.
    deriv : TYPE, optional
        DESCRIPTION. The default is 0.
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
    TYPE
        DESCRIPTION.

    '''
    
    if window_length == 0:
        return y

    if window_length < 0:
        raise ValueError('window_length (second argument) must be a positive integer!')
    
    
    if polyorder < deriv:
        raise ValueError('Polynomial order must not be smaller than the derivative index!')
    
    if not polyorder > 0:
        raise ValueError('Polynomial order must be a positive integer!')
    
    n=2*int(window_length) + 1
    npts = y.shape[0]
    y_expanded = np.full((npts+n,),np.nan) # extension of the data input to compensate the phase delay of the filter
    y_expanded[window_length+1:-window_length] = y[:,0]
    y_expanded[:window_length+1] = y[0,0] # edge extension as 'nearest' option
    y_expanded[-window_length:] = y[-1:,0] # edge extension as 'nearest' option
    if 'flat' == method:
        weights = np.divide(np.ones(n),n)
        a = lfilter(weights, 1, y_expanded,axis=0)
        y_smooth = a[n:]
    elif 'binom' == method:
        weights=np.diagonal(pascal(n)[::-1])
        weights=weights / 2 ** (n - 1)
        a = lfilter(weights, 1, y_expanded, axis=0)
        y_smooth = a[n:]
    elif 'savgol' == method:
        y_smooth= savgol_filter(y,n,polyorder,deriv, axis=0, mode='nearest')
    else:
        raise ValueError('Unknown value for third argument!')

    return y_smooth