import numpy as np

def fdaxis(TimeAxis=None,*args,**kwargs):
    if TimeAxis==None:  
        raise ValueError('Wrong number of input arguments!')
    TimeAxis=np.ravel(TimeAxis)
    dT=TimeAxis[1] - TimeAxis[0]
    N=len(TimeAxis)      
    NyquistFrequency=1/(2*dT)
    UnitAxis=np.multiply(2/N,np.arange(0,N-1)-np.fix(N/2))
    FreqAxis=NyquistFrequency*UnitAxis
    return FreqAxis