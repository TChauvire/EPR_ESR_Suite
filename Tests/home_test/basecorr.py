import numpy as np
def basecorr2D(Spectrum=None,Dimension=None,Order=None,*args,**kwargs):
    if np.array(Spectrum).all() == None or Order == None:
        raise ValueError('basecorr() needs 3 inputs: basecorr(Spectrum,Dimension,Order)')
    
    if np.array(Order).any() < 0 or np.array(Order).dtype.char not in 'bBhHiIlLqQpP':
        raise ValueError('Order must contain integers between 0 and 4!')
    ndims = len(Spectrum.shape)
    if Dimension == None:
        if Spectrum.size == len(Spectrum):
            raise ValueError('Multidimensional fitting not possible for 1D data!')
        if ndims > 2:
            raise ValueError('Multidimensional fitting for {0}D arrays not implemented!'.format(str(ndims)))
        if Order.size != ndims:
            raise ValueError('Order must have %d elements!',str(ndims))
        if max(Order) >= max(Spectrum.shape):
            raise ValueError('The Order value is too large for the given Spectrum!')
        m,n=Spectrum.shape
        x=np.tile(np.linspace(1,m,m),(n,1)).T
        x=np.ravel(x)
        y=np.tile(np.linspace(1,n,n),(m,1))
        y=np.ravel(y)
        q=0
        for j in range(Order[1]):
            for i in range(Order[0]):
                # Construct in each column  the x**i * y**j monomial vector
                D[:,q]=np.multiply(x ** i,y ** j) 
                q=q+1
        # Solve least squares problem
        C=np.linalg.solve(D,np.ravel(Spectrum))
        BaseLine=np.zeros(x.shape)
        for q in range(D.shape[1]):
            BaseLine=BaseLine + C[q]@D[:,q]
        BaseLine=BaseLine.reshape((m,n))
        CorrSpectrum=Spectrum - BaseLine
        p = np.NaN
    else:
        if Order != len(Dimension):
            raise ValueError('Order and Dimension must have the same number of elements!')
        nDimension = 0
        for i in Dimension:
            nDimension += i
            
        if  nDimension > ndims or min(Dimension) < 1:
            raise ValueError('Dimension out of range!')
        BaseLine=np.zeros(Spectrum.shape)
        CorrSpectrum=np.copy(Spectrum)
        for q in range(len(Dimension)):
            d=Dimension[q]
            if Order[q] >= Spectrum.shape[d-1]:
                raise ValueError('Dimension {0} has {1} points. Polynomial order {2} not possible. Check Order and Dimension!'.format(str(d),str(Spectrum.shape[d-1]),str(Order[q])))
            thisSpectrum=np.transpose(CorrSpectrum,[d]+list(range(d))+list(range(d+1,ndims)))
            x=np.linspace(0,thisSpectrum.shape[0],thisSpectrum.shape[0]).T
            thisBaseLine=np.zeros(thisSpectrum.shape)
            if Order[q] == 0:
                p=np.mean(thisSpectrum)
                for c in range(thisSpectrum.shape[1]):
                    thisBaseLine[:,c]=p[c]
            else:
                p=np.polyfit(x,thisSpectrum,Order[q],full = False,cov=False)
                for c in range(thisSpectrum.shape[1]):
                    thisBaseLine[:,c]=thisBaseLine[:,c] + np.polyval(p,x)
            thisBaseLine=np.transpose(thisBaseLine,list(range(1,d+1))+[0]+list(range(d+1,ndims)))
            BaseLine=BaseLine + thisBaseLine
            CorrSpectrum=CorrSpectrum - thisBaseLine
    return CorrSpectrum,BaseLine,p