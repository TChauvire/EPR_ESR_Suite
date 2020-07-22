import numpy as np

def basecorr1D(y=None, Mode=None, *args,**kwargs):
    if np.array(y).all()==None or Mode==None:
        raise ValueError('rescale function need at least two input argument: the numpy array and the scaling mode!')

    Mode=str(Mode)
    
    if y.shape[0] != np.ravel(y).shape[0]:
        raise ValueError('y must be a column vector. basecorr1D function does not work on 2D arrays.')
    else:
        y = np.ravel(y)

    npts=y.shape[0]
    if Mode not in ['minmax','maxabs',lsq','lsq0','lsq1','lsq2','lsq3','None']:
        raise ValueError('Unknown scaling mode {0}'.format(str(Mode)))
    else:
        ynew = np.full(y.shape,np.nan)
        if Mode == 'minmax':
            mi=0
            ma=1
            ratio = (ma-mi)/(np.max(y) - np.min(y))
            scalefactor = np.array([ratio, (mi - ratio*np.min(y))]).reshape((2,1))
            D=np.column_stack((y,np.ones(y.shape)))
            ynew=D@scalefactor
        elif Mode == 'maxabs':
            scalefactor = np.array(1/np.max(np.abs(y)))
            ynew=y*scalefactor[0]
        elif Mode == 'shift':
            scalefactor=np.mean(y_notnan) - np.mean(yref_notnan)
            ynew=y - scalefactor
        elif Mode == 'lsq':
            scalefactor=np.linalg.solve(y[notnan_both],yref[notnan_both])
            ynew=scalefactor@y
        elif Mode == 'lsq0':
            D=np.concatenate((y,np.ones(y.shape)),axis=0)
            scalefactor=np.linalg.solve(D[notnan_both],yref[notnan_both])
            ynew=D@scalefactor
        elif Mode == 'lsq1':
            x=(np.arange(1,npts).T)/npts
            D=np.concatenate((y,np.ones(y.shape),x),axis=0)
            scalefactor=np.linalg.solve(D[notnan_both],yref[notnan_both])
            ynew=D@scalefactor
        elif Mode == 'lsq2':
            x=(np.arange(1,npts).T)/npts
            D=np.concatenate((y,np.ones(y.shape),x,x ** 2),axis=0)
            scalefactor=np.linalg.solve(D[notnan_both],yref[notnan_both])
            ynew=D@scalefactor
        elif Mode == 'lsq3':
            x=(np.arange(1,npts).T)/npts
            D=np.concatenate((y,np.ones(y.shape),x,x**2,x**3),axis=0)
            scalefactor=np.linalg.solve(D[notnan_both],yref[notnan_both])
            ynew=D@scalefactor
        elif Mode == 'None':
            
            scalefactor=1
        elif Mode == 'None':
            scalefactor = np.array([1])
            ynew=y*scalefactor[0]
    return ynew,scalefactor