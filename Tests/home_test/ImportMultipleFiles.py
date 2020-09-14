# -*- coding: utf-8 -*-
"""
Importing Multiple files
Return a list of name and foldername
Created on Mon Jul  6 11:57:42 2020

@author: TC229401
"""
from os.path import join, normpath
from os import walk, getcwd
from datasmooth import *
from eprload_BrukerBES3T import *
from eprload_BrukerESP import *
from basecorr1D import *
from automatic_phase import *

def ImportMultipleNameFiles(FolderPath = getcwd(), Extension = None,*args,**kwargs):
    ListOfFiles = []
    for root, dirs, files in walk(FolderPath):
        for file in files:
            if file.endswith(Extension):
                 ListOfFiles.append(normpath(join(root, file)))
    return ListOfFiles

def MaxLengthOfFiles(ListOfFiles,*args,**kwargs):
    maxlen = 0
    for file in ListOfFiles:
        data, abscissa, par = eprload(file,Scaling=None)
        if maxlen<data.shape[0]:
            maxlen = data.shape[0]
    return maxlen

def OpenMultipleFiles(ListOfFiles,Scaling=None,polyorder=0, window=20,*args,**kwargs):
    maxlen = MaxLengthOfFiles(ListOfFiles,*args,**kwargs)
    fulldata = np.full((maxlen,4*len(ListOfFiles)),np.nan)
    for file in ListOfFiles:
        i = ListOfFiles.index(file)
        data, abscissa, par = eprload(FileName = file,Scaling=Scaling)
        if (data.shape[0] != np.ravel(data).shape[0]):
            raise ValueError('The file {0} is\'t a column vector'.format(par['TITL']))
        else: 
            data = np.ravel(data)
            data,_,_,_ = basecorr1D(x=abscissa,y=data, polyorder=polyorder,window=window)
            npts = abscissa.shape[0]
            fulldata[0:npts,4*i] = abscissa[0:npts,0]
            fulldata[0:npts,4*i+1] = data[0:npts]
            newdata = datasmooth(data[0:npts],window_length=4,method='flat')
            fulldata[0:npts,4*i+2],_,_,_ = basecorr1D(x=abscissa,y=newdata, polyorder=polyorder,window=window)
            fulldata[0:npts,4*i+3] = fulldata[0:npts,4*i+2]/np.max(fulldata[0:npts,4*i+2])
            #Header = par['TITL']
    return fulldata

def eprload(FileName = None,Scaling=None,*args,**kwargs):
    if FileName[-4:].upper() in ['.DSC','.DTA']:
        data, abscissa, par = eprload_BrukerBES3T(FileName,Scaling)
    elif FileName[-4:].lower() in ['.spc','.par']:
        data, abscissa, par = eprload_BrukerESP(FileName,Scaling)
    else:
        data, abscissa, par = None,None,None
        raise ValueError("Can\'t Open the File {0} because the extension",
                         "isn't a Bruker extension .DSC,.DTA,.spc, or .par!"
                         ).format(str(FileName))
    return data, abscissa, par  

def OpenMultipleComplexFiles(ListOfFiles,Scaling=None,polyorder=0, window=20,*args,**kwargs):
    maxlen = MaxLengthOfFiles(ListOfFiles,*args,**kwargs)
    ncol = 6
    fulldata = np.full((maxlen,ncol*len(ListOfFiles)),np.nan,dtype="complex_")
    for file in ListOfFiles:
        i = ListOfFiles.index(file)
        data, abscissa, par = eprload(FileName = file,Scaling=None)
        if (data.shape[0] != np.ravel(data).shape[0]):
            raise ValueError('The file {0} is\'t a column vector'.format(par['TITL']))
        else: 
            data = np.ravel(data)
            ### First Phase the data
            new_data, _ = automatic_phase(vector=data,pivot1=int(data.shape[0]/2),funcmodel='minfunc')
            data_real = new_data.real
            data_imag = new_data.imag
            data_real_new,_,_,_ = basecorr1D(x=abscissa,y=data_real, polyorder=polyorder,window=window)
            data_imag_new,_,_,_ = basecorr1D(x=abscissa,y=data_imag, polyorder=polyorder,window=window)
            npts = abscissa.shape[0]
            fulldata[0:npts,ncol*i] = abscissa[0:npts,0]
            fulldata[0:npts,ncol*i+1] = data[0:npts]
            fulldata[0:npts,ncol*i+2] = new_data[0:npts]
            fulldata[0:npts,ncol*i+3] = data_real_new+1j*data_imag_new
            fulldata[0:npts,ncol*i+4] = datasmooth(np.absolute(fulldata[0:npts,ncol*i+3][0:npts]),
                                                 window_length=4,
                                                 method='flat')
            fulldata[0:npts,ncol*i+5] = fulldata[0:npts,ncol*i+4]/np.amax(fulldata[0:npts,ncol*i+4])
            Header = par['TITL']
    return fulldata