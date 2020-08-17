import numpy as np

def fdaxis(TimeAxis=None,*args,**kwargs):
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