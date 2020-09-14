import numpy as np
from scipy.signal import lfilter, savgol_filter
from scipy.linalg import pascal

def datasmooth(y=None,window_length=1,method='binom',polyorder=2,deriv=0,*args,**kwargs):
    '''
    datasmooth: Moving average smoothing and differentiation 
    3 filters are optional for smoothing the data:
        flat, the moving average is unweighted, 
        binomial, binomial coefficients are used as weighting factors and 
        Savitzky-Golay polynomial filter of order p.
    
    This script is freely inspired by the easyspin suite from the Stefan Stoll lab
    (https://github.com/StollLab/EasySpin/)
    (https://easyspin.org/easyspin/)
    
    Script written by Timothée Chauviré (https://github.com/TChauvire/EPR_ESR_Suite/), 09/09/2020
    
    Parameters
    ----------
    y : numpy array in column vector data format to compute. 
        The data must be 1D only else an error is produced.
        DESCRIPTION. The default is None.
        
    window_length : number of data points (integer format) used for smoothing y, optional
        DESCRIPTION. The default is 1.
    method : type of method to use for smoothing the data. TYPE = string, optional
        DESCRIPTION: 3 types are proposed:
        'flat', the moving average is unweighted, 
        'binomial', binomial coefficients are used as weighting factors and 
        'savgol', Savitzky-Golay polynomial filter of order p.
            If 'savgol' is specified, a least-squares smoothing using 
            the Savitzky-Golay polynomial filter of order p is computed.
            It least-square fits p-order polynomials to 2*m+1 wide frames.
            If deriv>0, a derivative of y is computed at the same time. 
            E.g. if deriv=3, y is denoised and its third derivative is returned.

        The default method is 'binom'.
    
    polyorder : rank of the polynomial order used by the Savitzky-Golay polynomial filter.
        TYPE = integer, optional
        DESCRIPTION. The default is 2.
    deriv : a derivative of y is computed at the same time. TYPE = integer, optional
        DESCRIPTION. The default is 0.

    Returns
    -------
    y_smooth: weighted average of the input y.
    y_smooth[i] is a weighted average of y[i-window_length:i+window_length].
    y_smooth is enlarged at start and end by its start and end values,
    
    TYPE = numpy data array column vector. Same shape as the input data array y.
        DESCRIPTION.

    '''
    input_shape = y.shape
    if (y.shape[0] != np.ravel(y).shape[0]):
        raise ValueError('The file data must be a column vector')
    else:
        y = np.ravel(y)
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
    y_expanded[window_length+1:-window_length] = y[0:npts]
    y_expanded[:window_length+1] = y[0] # edge extension as 'nearest' option
    y_expanded[-window_length:] = y[-1:] # edge extension as 'nearest' option
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
    y_smooth = y_smooth.reshape(input_shape)
    return y_smooth