import numpy as np

def windowing(window_type=None,M=None,alpha=None,*args,**kwargs):
    '''Returns an apodization window. M is the number
    of points. The string window_type specifies the type of the windowing
    and can be

      'bla'    Blackman
      'bar'    Bartlett
      'con'    Connes
      'cos'    Cosine
      'ham'    Hamming
      'han'    Hann (also called Hanning)
      'wel'    Welch
    The following three windows need the parameter
    alpha. Reasonable ranges for alpha are given.

      'exp'    Exponential    2 to 6
      'gau'    Gaussian       0.6 to 1.2
      'kai'    Kaiser         3 to 9

    A '+' ('-') appended to Type indicates that only the
    right (left) half of the window should be constructed.

      'ham'    symmetric (-1 <= x <= 1, n points)
      'ham+'   right side only (0 <= x <= 1, n points)
      'ham-'   left side only (-1 <= x <= 0, n points)
    '''
    if ((window_type == 3 or window_type == 4) and type(window_type) ==str):
        raise ValueError('The Argument "window_type" must be a 3- or 4-character string!')
    
    if len(window_type) == 4:
        if '+' == window_type[3]:
            xmin=0
            xmax=1
        elif '-' == window_type[3]:
            xmin=-1
            xmax=0
        else:
            raise ValueError('Wrong 4th character in window_type. Should be + or -.')
    else:
        xmin=-1
        xmax=1
    
    if not (type(M) == int and M>0):
        raise ValueError('M must be a positive integer!')
    
    x=np.linspace(xmin,xmax,M)

    if window_type[:3] == 'ham':
        w = 0.54+0.46*np.cos(np.pi*x)
    elif window_type[:3] == 'kai':
        w = np.i0(float(alpha)*np.sqrt(1-x**2.0))/np.i0(float(alpha))
    elif window_type[:3] == 'gau':
        w=np.exp((-2*x**2)/(alpha**2))
    elif window_type[:3] == 'exp':
        w=np.exp(-alpha*np.abs(x))
    elif window_type[:3] == 'han':
        w= 0.5 + 0.5*np.cos(np.pi*x)
    elif window_type[:3] == 'bla':
        w= 0.42 + 0.5*np.cos(np.pi*x) + 0.08*np.cos(2*np.pi*x)
    elif window_type[:3] == 'bar':
        w=w = 1 - np.abs(x);
    elif window_type[:3] == 'con':
        w=(1-x**2)**2
    elif window_type[:3] == 'cos':
        w=np.cos(np.pi*x/2)
    elif window_type[:3] == 'wel':
        w=1-x**2
    else:
        raise ValueError('Unknown apodization window specified!')
    
   
    if (xmax - xmin == 2):
        # Symmetrize (remove possible numerical asymmetries)
        w=(w + w[::-1])/2
    
    w=np.divide(w,np.max(w))
    return w