import numpy as np
from sklearn.linear_model import LinearRegression as lin

def poly(input_array_for_poly,degree_for_poly):
    '''qr factorization of a matrix. q` is orthonormal and `r` is upper-triangular.
    - QR decomposition is equivalent to Gram Schmidt orthogonalization, which builds a sequence of orthogonal polynomials that approximate your function with minimal least-squares error
    - in the next step, discard the first column from above matrix.
    - for each value in the range of polynomial, starting from index 0 of pollynomial range, (for k in range(p+1))
        create an array in such a way that elements of array are (original_individual_value)^polynomial_index (x**k)
    - concatenate all of these arrays created through loop, as a master array. This is done through (np.vstack)
    - transpose the master array, so that its more like a tabular form(np.transpose)'''
    input_array_for_poly = np.array(input_array_for_poly)
    X = np.transpose(np.vstack((input_array_for_poly**k for k in range(degree_for_poly+1))))
    return np.linalg.qr(X)[0][:,1:]
def ModPoly(input_array,degree_for_poly,repetition=100,gradient=0.001):
    '''Implementation of Modified polyfit method from paper: 
        Automated Method for Subtraction of Fluorescence from Biological Raman 
        Spectra, by Lieber & Mahadevan-Jansen (2003)
    
    repetition: How many iterations to run. Default is 100
    gradient: Gradient for polynomial loss, default is 0.001. It measures incremental gain over each iteration. If gain in any iteration is less than this, further improvement will stop
    '''

    # initial improvement criteria is set as positive infinity, 
    # to be replaced later on with actual value
    criteria=np.inf

    baseline=[]
    corrected=[]

    ywork=input_array
    yold=input_array
    yorig=input_array

    polx=poly(list(range(1,len(yorig)+1)),degree_for_poly)
    nrep=0

    while (criteria>=gradient) and (nrep<=repetition):
        ypred=lin.fit(polx,yold).predict(polx)
        ywork=np.array(np.minimum(yorig,ypred))
        criteria=np.sum(np.abs((ywork-yold)/yold))
        yold=ywork
        nrep+=1
    corrected=yorig-ypred
    corrected=np.array(list(corrected))
    return corrected
    
def IModPoly(input_array,degree_for_poly, repetition=100,gradient=0.001):
    '''IModPoly from paper: Automated Autofluorescence Background Subtraction 
    Algorithm for Biomedical Raman Spectroscopy, 
    by Zhao, Jianhua, Lui, Harvey, McLean, David I., Zeng, Haishan (2007)
    
    repetition: How many iterations to run. Default is 100
    gradient: Gradient for polynomial loss, default is 0.001. 
    It measures incremental gain over each iteration. 
    If gain in any iteration is less than this, further improvement will stop.
    '''

    yold=input_array
    yorig=input_array
    corrected=[]

    nrep=1
    ngradient=1

    polx=poly(list(range(1,len(yorig)+1)),degree_for_poly)
    ypred=lin.fit(polx,yold).predict(polx)
    Previous_Dev=np.std(yorig-ypred)

    #iteration1
    yold=yold[yorig<=(ypred+Previous_Dev)]
    polx_updated=polx[yorig<=(ypred+Previous_Dev)]
    ypred=ypred[yorig<=(ypred+Previous_Dev)]

    for i in range(2,repetition+1):
        ypred=lin.fit(polx_updated,yold).predict(polx_updated)
        DEV=np.std(yold-ypred)
        if i>2:
            Previous_Dev=DEV
        if np.abs((DEV-Previous_Dev)/DEV) < gradient:
            break
        else:
            for i in range(len(yold)):
                if yold[i]>=ypred[i]+DEV:
                    yold[i]=ypred[i]+DEV
    baseline=lin.predict(polx)
    corrected=yorig-baseline
    corrected=np.array(list(corrected))
    return corrected