import numpy as np
def hilberttrans(y=None,*args,**kwargs):
    if (y.dtype.char not in 'efdg' or (y.shape[0] != np.ravel(y).shape[0])):
        raise ValueError('Input y must be a real column vector.')
    
    npts=y.shape[0]
    h = np.full(y.shape,np.nan)
    if npts%2 == 0:
        idx=int(npts/2)
        h[0]=1
        h[idx]=1
        h[1:idx-1]=2
        h[idx+1:npts]=0
    else:
        idx=int(np.ceil(npts/2))
        h[0]=1
        h[1:idx-1]=2
        h[idx:npts]=0
    print(h)
    y_h=np.fft.ifft((np.fft.fft(np.ravel(y)))*np.ravel(h))
    #y_h=y_h.reshape(y.shape)
    return y_h