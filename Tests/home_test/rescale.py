import numpy as np

def rescale(y=None, Mode=None, yref=None, *args,**kwargs):
    if np.array(y).all()==None or Mode==None:
        raise ValueError('rescale function need at least two input argument: the numpy array and the scaling mode!')

    Mode=str(Mode)
    
    if y.shape[0] != np.ravel(y).shape[0]:
        raise ValueError('y must be a column vector. rescale function does not work on 2D arrays.')
           
    npts=y.shape[0]
    if np.array(yref).all() == None:
        if Mode not in ['minmax','maxabs','None']:
            raise ValueError('Unknown scaling mode {0}'.format(str(Mode)))
        y = np.ravel(y)
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
        elif Mode == 'None':
            scalefactor = np.array([1])
            ynew=y*scalefactor[0]
    else:
        if yref.shape[0] != np.ravel(yref).shape[0]:
            raise ValueError('yref must be a column vector.')
        equalLengthNeeded=[0,0,0,1,1,1,1,1,0]
        scalename = ['maxabs','minmax','shift','lsq','lsq0','lsq1','lsq2','lsq3','None']

        if Mode not in ['maxabs','minmax','shift','lsq','lsq0','lsq1','lsq2','lsq3','None']:
            raise ValueError('Unknown scaling mode {0}'.format(str(Mode)))
        
        if equalLengthNeeded[scalename.index(Mode)] and (y.size != yref.size):
            raise ValueError('For least-squares rescaling, vectors must have same number of elements.')
        y = np.ravel(y)
        yref = np.ravel(yref)
        yref_notnan=yref[~np.isnan(yref)]
        y_notnan=y[~np.isnan(y)]
        if y_notnan.shape[0] == yref_notnan.shape[0]:
            notnan_both=[~np.isnan(y) == ~np.isnan(yref)]
        if Mode == 'maxabs':
            scalefactor=np.max(np.abs(yref_notnan))/np.max(np.abs(y_notnan))
            ynew=scalefactor*y
        elif Mode == 'minmax':
            scalefactor=(np.max(yref_notnan) - np.min(yref_notnan)) / (np.max(y_notnan) - np.min(y_notnan))
            ynew=scalefactor*(y-np.min(y_notnan))+np.min(yref_notnan)
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
    if (np.real(scalefactor[0]) < 0 and Mode in ['lsq','lsq0','lsq1','lsq2','lsq3']):
        scalefactor[0]=np.abs(scalefactor[0])
        ynew=D@scalefactor
      
    return ynew,scalefactor