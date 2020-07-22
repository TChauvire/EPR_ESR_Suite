import numpy as np
import matplotlib.pyplot as plt

def stackplot(x=None,y=None,scale=1,step=1,sliceLabels=None,*args,**kwargs):
    if np.array(scale).dtype.char not in 'bBhHiIlLqQpPdefg' or np.array(scale).size != 1:
        raise ValueError('Third argument (scale) must be a scalar number.')
    
    shape = y.shape
    y_stacked = np.full(shape,np.nan)
    if y.shape[0] == np.ravel(y).shape[0]:
        nSlices=1
        y=y.reshape((shape[0],nSlices))
    else:
        nSlices=shape[1]
    
    if sliceLabels==None:
        sliceLabels=np.arange(0,nSlices)
    
    if sliceLabels.size != nSlices:
        raise ValueError('Number of y tick labels must equal number of slices.')
    shift = np.full((nSlices,),np.nan)
    for k in range(nSlices):
        yy=y[:,k]
        shift[k]=np.multiply(k,step)
        if scale < 0:
            y_stacked[:,k]= yy/np.max(np.abs((yy)))*np.abs(scale) + shift[k]
        elif scale > 0:
            y_stacked[:,k]=(yy - np.min(yy))/(np.max(yy) - np.min(yy))*scale + shift[k]
        else:
            y_stacked[:,k]=yy + shift[k]
    
    if x.shape[0] == np.ravel(x).shape[0]:
        x=np.ravel(x)
        x=np.tile(x,nSlices)
    
    if shift[nSlices] < shift[0]:
        shift=shift[::-1]
    
    h=plt.plot(x,y)
    set(gca,'YTick',shift)
    set(gca,'YTickLabel',sliceLabels)
    axis('tight')
    
    return y_stacked  