# -*- coding: utf-8 -*-
"""
Importing Multiple files
Return a list of name and foldername
Created on Mon Jul  6 11:57:42 2020

@author: TC229401
"""
import os
from datasmooth import *
from eprload_BrukerBES3T import *
from eprload_BrukerESP import *

def ImportMultipleFiles(FolderPath = os.getcwd(), Extension = None,*args,**kwargs):
    result = []
    for root, dirs, files in os.walk(FolderPath):
        for file in files:
            if file.endswith(Extension):
                 result.append(str(os.path.join(root, file)))
    return result

def OpenMultipleFiles(ListOfFiles,Scaling,*args,**kwargs):
    for file in ListOfFiles:
        data, abscissa, par = eprload(file,Scaling)
        if (data.shape[0] != np.ravel(data).shape[0]):
            fulldata = np.full((npts,4*len(ListOfFiles)),np.nan)
            for i in range(len(ListOfFiles)):
                fulldata[0:npts,4*i-3] = abscissa[:,0]
                fulldata[0:npts,4*i-2] = data[:,0]
                fulldata[0:npts,4*i-1] = datasmooth(fulldata[0:npts,4*i-2],window_length=4,method='flat')
                fulldata[0:npts,4*i] = fulldata[0:npts,4*i-1]/np.max(fulldata[0:npts,4*i-1])
    return fulldata

def eprload(FileName = None,Scaling=None,*args,**kwargs):
    if FileName[:-4].upper() in ['.DSC','.DTA']:
        data, abscissa, par = eprload_BrukerESP(FileName,Scaling)
    elif FileName[:-4].upper() in ['.spc','.par']:
        data, abscissa, par = eprload_BrukerBES3T(FileName,Scaling)
    else:
        data, abscissa, par = None,None,None
        raise ValueError("Can\'t Open the File {0} because the extension",
                         "isn't a Bruker extension .DSC,.DTA,.spc, or .par!"
                         ).format(str(FileName))
    return data, abscissa, par    