# -*- coding: utf-8 -*-
"""
Importing Multiple files
Return a list of name and foldername
Created on Mon Jul  6 11:57:42 2020

@author: TC229401
"""
from os import walk, getcwd, path
from datasmooth import datasmooth
from eprload_BrukerBES3T import eprload_BrukerBES3T
from eprload_BrukerESP import eprload_BrukerESP
from basecorr1D import basecorr1D
from automatic_phase import automatic_phase
import numpy as np
# for 3PEseem import:
import numpy.polynomial.polynomial as pl
from fdaxis import fdaxis
from windowing import windowing
from scipy.optimize import curve_fit
from scipy.integrate import cumulative_trapezoid


def eprload(FileName=None, Scaling=None, *args, **kwargs):
    if FileName[-4:].upper() in ['.DSC', '.DTA']:
        data, abscissa, par = eprload_BrukerBES3T(FileName, Scaling)
    elif FileName[-4:].lower() in ['.spc', '.par']:
        data, abscissa, par = eprload_BrukerESP(FileName, Scaling)
    else:
        data, abscissa, par = None, None, None
        raise ValueError("Can\'t Open the File {0} ".format(str(FileName)) +
                         "because the extension isn\'t a Bruker extension " +
                         ".DSC, .DTA, .spc, or .par!")
    return data, abscissa, par


def ImportMultipleNameFiles(FolderPath=getcwd(), Extension=None,
                            *args, **kwargs):
    ListOfFiles = []
    for root, dirs, files in walk(FolderPath):
        for file in files:
            if file.endswith(Extension):
                ListOfFiles.append(path.normpath(
                    path.join(root, file)))
    return ListOfFiles


def MaxLengthOfFiles(ListOfFiles, *args, **kwargs):
    maxlen = 0
    for file in ListOfFiles:
        data, abscissa, par = eprload(file, Scaling=None)
        if maxlen < data.shape[0]:
            maxlen = data.shape[0]
    return maxlen


def OpenMultipleFiles(ListOfFiles, Scaling=None, polyorder=0,
                      window=20, *args, **kwargs):
    maxlen = MaxLengthOfFiles(ListOfFiles, *args, **kwargs)
    ncol = 4
    fulldata = np.full((maxlen, 4*len(ListOfFiles)), np.nan)
    Header = list(np.zeros((ncol*len(ListOfFiles),)))
    for file in ListOfFiles:
        i = ListOfFiles.index(file)
        data, abscissa, par = eprload(FileName=file, Scaling=Scaling)
        if (data.shape[0] != np.ravel(data).shape[0]):
            raise ValueError(
                'The file {0} is\'t a column vector'.format(par['TITL']))
        else:
            data = np.ravel(data)
            data, _, _, _ = basecorr1D(
                x=abscissa, y=data, polyorder=polyorder, window=window)
            npts = abscissa.shape[0]
            fulldata[0:npts, 4*i] = abscissa[0:npts, 0]
            fulldata[0:npts, 4*i+1] = data[0:npts]
            newdata = datasmooth(data[0:npts], window_length=8, method='binom')
            fulldata[0:npts, 4*i+2], _, _, _ = basecorr1D(
                x=abscissa, y=newdata, polyorder=polyorder, window=window)
            fulldata[0:npts, 4*i+3] = fulldata[0:npts, 4*i+2] / \
                np.max(fulldata[0:npts, 4*i+2])
            Header[ncol*i] = par['XNAM']
            Header[ncol*i+1] = par['TITL']
            Header[ncol*i+2] = par['TITL']+str("_4ptsSmoothed_BkgdCorrected")
            Header[ncol*i+3] = par['TITL'] + \
                str("_4ptsSmoothed_BkgdCorrected_Normalized")
    return fulldata, Header


def OpenDoubleIntegral(ListOfFiles, Scaling=None, polyorder=[0, 0],
                       window=20, *args, **kwargs):
    maxlen = MaxLengthOfFiles(ListOfFiles, *args, **kwargs)
    ncol = 4
    FirstDeriv = np.full((maxlen, 4*len(ListOfFiles)), np.nan)
    ZeroDeriv = np.full((maxlen, 4*len(ListOfFiles)), np.nan)
    Header = list(np.zeros((ncol*len(ListOfFiles),)))
    HeaderInt = list(np.zeros((len(ListOfFiles),)))
    IntgValue = list(np.zeros((len(ListOfFiles),)))
    for file in ListOfFiles:
        i = ListOfFiles.index(file)
        data, abscissa, par = eprload(FileName=file, Scaling=Scaling)
        if (data.shape[0] != np.ravel(data).shape[0]):
            raise ValueError(
                'The file {0} is\'t a column vector'.format(par['TITL']))
        else:
            data = np.ravel(data)
            data, _, _, _ = basecorr1D(
                x=abscissa, y=data, polyorder=polyorder[0], window=window)
            npts = abscissa.shape[0]
            FirstDeriv[0:npts, 4*i] = abscissa[0:npts, 0]
            FirstDeriv[0:npts, 4*i+1] = data[0:npts]
            newdata = datasmooth(data[0:npts], window_length=4, method='binom')
            FirstDeriv[0:npts, 4*i+2] = newdata[0:npts]
            FirstDeriv[0:npts, 4*i+3] = FirstDeriv[0:npts, 4*i+2] / \
                np.max(FirstDeriv[0:npts, 4*i+2])
            Header[ncol*i] = par['XNAM']
            Header[ncol*i+1] = par['TITL']
            Header[ncol*i+2] = par['TITL']+str("_4ptsSmoothed_BkgdCorrected")
            Header[ncol*i+3] = par['TITL'] + \
                str("_4ptsSmoothed_BkgdCorrected_Normalized")
            # achieve first integral of the data
            newdata2 = cumulative_trapezoid(data.ravel(),
                                            abscissa.ravel(), initial=0)
            npts2 = newdata2.shape[0]
            newdata2, _, _, _ = basecorr1D(
                x=abscissa, y=newdata2, polyorder=polyorder[1], window=window)
            ZeroDeriv[0:npts2, 4*i] = abscissa[0:npts2, 0]
            ZeroDeriv[0:npts2, 4*i+1] = newdata2[0:npts2]
            newdata2 = datasmooth(
                newdata2[0:npts2], window_length=4, method='binom')
            ZeroDeriv[0:npts2, 4*i+2] = newdata2[0:npts2]
            ZeroDeriv[0:npts2, 4*i+3] = ZeroDeriv[0:npts2, 4*i+2] / \
                np.max(ZeroDeriv[0:npts2, 4*i+2])
            # integrate the zero derivative
            IntgValue[i] = np.trapz(
                ZeroDeriv[0:npts2, 4*i+1].ravel(), abscissa[0:npts2].ravel())
            HeaderInt[i] = par['TITL']
    return FirstDeriv, ZeroDeriv, Header, IntgValue, HeaderInt


def OpenDoubleIntegral2(ListOfFiles, Scaling=None, polyorder=[0, 0],
                        window=20, *args, **kwargs):
    maxlen = MaxLengthOfFiles(ListOfFiles, *args, **kwargs)
    ncol = 4
    FirstDeriv = np.full((maxlen, 4*len(ListOfFiles)), np.nan)
    ZeroDeriv = np.full((maxlen, 4*len(ListOfFiles)), np.nan)
    Header = list(np.zeros((ncol*len(ListOfFiles),)))
    HeaderInt = list(np.zeros((len(ListOfFiles),)))
    IntgValue = list(np.zeros((len(ListOfFiles),)))
    bckg, _, _ = eprload(FileName=ListOfFiles[0], Scaling=Scaling)
    for file in ListOfFiles:
        i = ListOfFiles.index(file)
        data, abscissa, par = eprload(FileName=file, Scaling=Scaling)
        if (data.shape[0] != np.ravel(data).shape[0]):
            raise ValueError(
                'The file {0} is\'t a column vector'.format(par['TITL']))
        else:
            # data = np.ravel(data) - np.ravel(bckg)
            data, _, _, _ = basecorr1D(
                x=abscissa, y=data, polyorder=polyorder[0], window=window)
            npts = abscissa.shape[0]
            FirstDeriv[0:npts, 4*i] = abscissa[0:npts, 0]
            FirstDeriv[0:npts, 4*i+1] = data[0:npts]
            newdata = datasmooth(data[0:npts], window_length=4, method='binom')
            FirstDeriv[0:npts, 4*i+2] = newdata[0:npts]
            FirstDeriv[0:npts, 4*i+3] = FirstDeriv[0:npts, 4*i+2] / \
                np.max(FirstDeriv[0:npts, 4*i+2])
            Header[ncol*i] = par['XNAM']
            Header[ncol*i+1] = par['TITL']
            Header[ncol*i+2] = par['TITL']+str("_4ptsSmoothed_BkgdCorrected")
            Header[ncol*i+3] = par['TITL'] + \
                str("_4ptsSmoothed_BkgdCorrected_Normalized")
            # achieve first integral of the data
            newdata2 = cumulative_trapezoid(data.ravel(),
                                            abscissa.ravel(), initial=None)
            npts2 = newdata2.shape[0]
            newdata2, _, _, _ = basecorr1D(
                x=abscissa[0:npts2,].ravel(), y=newdata2[0:npts2,].ravel(),
                polyorder=polyorder[1], window=window)
            ZeroDeriv[0:npts2, 4*i] = abscissa[0:npts2, 0]
            ZeroDeriv[0:npts2, 4*i+1] = newdata2[0:npts2]
            newdata2 = datasmooth(
                newdata2[0:npts2], window_length=4, method='binom')
            ZeroDeriv[0:npts2, 4*i+2] = newdata2[0:npts2]
            ZeroDeriv[0:npts2, 4*i+3] = ZeroDeriv[0:npts2, 4*i+2] / \
                np.max(ZeroDeriv[0:npts2, 4*i+2])
            # integrate the zero derivative for half of the spectrum
            init = int(np.floor(npts2/4))
            end = int(np.floor(3*npts2/4))
            IntgValue[i] = np.trapz(ZeroDeriv[init:end, 4*i+1].ravel(),
                                    abscissa[init+1:end+1].ravel())
            HeaderInt[i] = par['TITL']
    return FirstDeriv, ZeroDeriv, Header, IntgValue, HeaderInt


def OpenDoubleIntegralKin(ListOfFiles, Scaling=None, polyorder=[0, 0],
                          window=20, *args, **kwargs):
    maxlen = MaxLengthOfFiles(ListOfFiles, *args, **kwargs)
    ncol = 4
    FirstDeriv = np.full((maxlen, 4*len(ListOfFiles)), np.nan)
    ZeroDeriv = np.full((maxlen, 4*len(ListOfFiles)), np.nan)
    Header = list(np.zeros((ncol*len(ListOfFiles),)))
    HeaderInt = list(np.zeros((len(ListOfFiles),)))
    IntgValue = list(np.zeros((len(ListOfFiles),)))
    bckg, _, _ = eprload(FileName=ListOfFiles[0], Scaling=Scaling)
    for file in ListOfFiles:
        i = ListOfFiles.index(file)
        data, abscissa, par = eprload(FileName=file, Scaling=Scaling)
        data = np.mean(data, axis=1)
        abscissa = abscissa[:, 0]
        if (data.shape[0] != np.ravel(data).shape[0]):
            raise ValueError(
                'The file {0} is\'t a column vector'.format(par['TITL']))
        else:
            # data = np.ravel(data) - np.ravel(bckg)
            data, _, _, _ = basecorr1D(
                x=abscissa, y=data, polyorder=polyorder[0], window=window)
            npts = abscissa.shape[0]
            FirstDeriv[0:npts, 4*i] = abscissa[0:npts,]
            FirstDeriv[0:npts, 4*i+1] = data[0:npts]
            newdata = datasmooth(data[0:npts], window_length=4, method='binom')
            FirstDeriv[0:npts, 4*i+2] = newdata[0:npts]
            FirstDeriv[0:npts, 4*i+3] = FirstDeriv[0:npts, 4*i+2] / \
                np.max(FirstDeriv[0:npts, 4*i+2])
            Header[ncol*i] = par['XNAM']
            Header[ncol*i+1] = par['TITL']
            Header[ncol*i+2] = par['TITL']+str("_4ptsSmoothed_BkgdCorrected")
            Header[ncol*i+3] = par['TITL'] + \
                str("_4ptsSmoothed_BkgdCorrected_Normalized")
            # achieve first integral of the data
            newdata2 = cumulative_trapezoid(data.ravel(),
                                            abscissa.ravel(), initial=None)
            npts2 = newdata2.shape[0]
            newdata2, _, _, _ = basecorr1D(
                x=abscissa[0:npts2,].ravel(), y=newdata2[0:npts2,].ravel(),
                polyorder=polyorder[1], window=window)
            ZeroDeriv[0:npts2, 4*i] = abscissa[0:npts2,]
            ZeroDeriv[0:npts2, 4*i+1] = newdata2[0:npts2]
            newdata2 = datasmooth(
                newdata2[0:npts2], window_length=4, method='binom')
            ZeroDeriv[0:npts2, 4*i+2] = newdata2[0:npts2]
            ZeroDeriv[0:npts2, 4*i+3] = ZeroDeriv[0:npts2, 4*i+2] / \
                np.max(ZeroDeriv[0:npts2, 4*i+2])
            # integrate the zero derivative for half of the spectrum
            init = int(np.floor(npts2/5))
            end = int(np.floor(4*npts2/5))
            IntgValue[i] = np.trapz(ZeroDeriv[init:end, 4*i+1].ravel(),
                                    abscissa[init+1:end+1].ravel())
            HeaderInt[i] = par['TITL']
    return FirstDeriv, ZeroDeriv, Header, IntgValue, HeaderInt


def OpenDoubleIntegralKin2(filename, Scaling=None, col=[], polyorder=[0, 0],
                           window=20, *args, **kwargs):
    ListOfFiles = [filename]
    maxlen = MaxLengthOfFiles(ListOfFiles, *args, **kwargs)
    ncol = 4
    FirstDeriv = np.full((maxlen, 4), np.nan)
    ZeroDeriv = np.full((maxlen, 4), np.nan)
    Header = list(np.zeros(ncol,))
    HeaderInt = list(np.zeros(1,))
    IntgValue = list(np.zeros(1,))
    truncsize = 0
    data, abscissa, par = eprload(filename, Scaling=Scaling)
    abscissa = abscissa[:, 0]
    extractdata = np.full((maxlen-2*truncsize,), 0.0)
    for i in col:
        extractdata += data[:, i]
        # extractdata += data[truncsize:-truncsize, i]
    extractdata = extractdata/len(col)
    # abscissa = abscissa[truncsize:-truncsize, 0]
    extractdata, _, _, _ = basecorr1D(x=abscissa, y=extractdata,
                                      polyorder=polyorder[0], window=window)
    npts = abscissa.shape[0]
    FirstDeriv[0:npts, 0] = abscissa[0:npts,]
    FirstDeriv[0:npts, 1] = extractdata[0:npts]
    newdata = datasmooth(extractdata[0:npts], window_length=0, method='binom')
    FirstDeriv[0:npts, 2] = newdata[0:npts]
    FirstDeriv[0:npts, 3] = FirstDeriv[0:npts, 2] / \
        np.max(FirstDeriv[0:npts, 2])
    Header[0] = par['XNAM']
    Header[1] = par['TITL']
    Header[2] = par['TITL']+str("_4ptsSmoothed_BkgdCorrected")
    Header[3] = par['TITL']+str("_4ptsSmoothed_BkgdCorrected_Norm")
    # achieve first integral of the data
    newdata2 = cumulative_trapezoid(extractdata.ravel(),
                                    abscissa.ravel(), initial=None)
    npts2 = newdata2.shape[0]
    newdata2, _, _, _ = basecorr1D(
        x=abscissa[0:npts2,].ravel(), y=newdata2[0:npts2,].ravel(),
        polyorder=polyorder[1], window=window)
    ZeroDeriv[0:npts2, 0] = abscissa[0:npts2,]
    ZeroDeriv[0:npts2, 1] = newdata2[0:npts2]
    newdata2 = datasmooth(
        newdata2[0:npts2], window_length=0, method='binom')
    ZeroDeriv[0:npts2, 2] = newdata2[0:npts2]
    ZeroDeriv[0:npts2, 3] = ZeroDeriv[0:npts2, 2] / \
        np.max(ZeroDeriv[0:npts2, 2])
    # integrate the zero derivative for half of the spectrum
    init = int(np.floor(npts2/5))
    end = int(np.floor(4*npts2/5))
    IntgValue = np.trapz(ZeroDeriv[init:end, 1].ravel(),
                         abscissa[init+1:end+1].ravel())
    HeaderInt = par['TITL']
    return FirstDeriv, ZeroDeriv, Header, IntgValue, HeaderInt


def OpenDoubleIntegralKin3(filename, Scaling=None, polyorder=[0, 0],
                           window=20, PP_Index=[], *args, **kwargs):
    ListOfFiles = [filename]
    maxlen = MaxLengthOfFiles(ListOfFiles, *args, **kwargs)
    # ncol = 4
    data, abscissa, par = eprload(FileName=filename, Scaling=Scaling)
    nslice = par['YPTS']
    FirstDeriv = np.full((maxlen, nslice+1), np.nan)
    ZeroDeriv = np.full((maxlen, nslice+1), np.nan)
    Header = list(np.zeros((nslice+1),))
    HeaderInt = list(np.zeros(nslice+1,))
    IntgValue = np.full((nslice, 4), np.nan)
    x = abscissa[:, 0]
    for i in range(par['YPTS']):
        data[:, i], _, _, _ = basecorr1D(
            x=x, y=data[:, i], polyorder=polyorder[0], window=window)
        npts = x.shape[0]
        FirstDeriv[0:npts, 0] = x[0:npts,]
        newdata = datasmooth(
            data[0:npts, i], window_length=0, method='binom')
        FirstDeriv[0:npts, i+1] = newdata[0:npts,]
        Header[0] = par['XNAM']
        Header[i+1] = par['TITL']+str("_slice{0}".format(i))
        # get the Peak to Peak intensity with index value:
        # index has to be specified by pair
        nave = 0
        Ipp1 = (np.mean(data[PP_Index[0]-nave:PP_Index[0]+nave+1, i], axis=0) -
                np.mean(data[PP_Index[1]-nave:PP_Index[1]+nave+1, i], axis=0))
        # Ipp2 = (np.mean(data[PP_Index[2]-nave:PP_Index[2]+nave+1, i], axis=0) -
        #         np.mean(data[PP_Index[3]-nave:PP_Index[3]+nave+1, i], axis=0))
        # achieve first integral of the data
        newdata2 = cumulative_trapezoid(data[:, i].ravel(),
                                        x.ravel(), initial=None)
        npts2 = newdata2.shape[0]
        newdata2, _, _, _ = basecorr1D(
            x=x[0:npts2,].ravel(), y=newdata2[0:npts2,].ravel(),
            polyorder=polyorder[1], window=window)
        ZeroDeriv[0:npts2, 0] = x[0:npts2,]
        newdata2 = datasmooth(
            newdata2[0:npts2], window_length=0, method='binom')
        ZeroDeriv[0:npts2, i+1] = newdata2[0:npts2]
        init = int(np.floor(npts2/5))
        end = int(np.floor(4*npts2/5))
        IntgValue[0:nslice, 0] = abscissa[0:nslice, 1]
        IntgValue[i, 1] = np.trapz(newdata2[init:end].ravel(),
                                   x[init+1:end+1].ravel())
        HeaderInt[i] = par['TITL']+str("_slice{0}".format(i))
        IntgValue[i, 2] = Ipp1
        # IntgValue[i, 3] = Ipp2
    return FirstDeriv, ZeroDeriv, Header, IntgValue, HeaderInt


def OpenMultipleComplexFiles(ListOfFiles, Scaling=None, polyorder=0,
                             window=20, *args, **kwargs):
    maxlen = MaxLengthOfFiles(ListOfFiles, *args, **kwargs)
    ncol = 6
    fulldata = np.full((maxlen, ncol*len(ListOfFiles)),
                       np.nan, dtype="complex_")
    for file in ListOfFiles:
        i = ListOfFiles.index(file)
        data, x, par = eprload(FileName=file, Scaling=None)
        if (data.shape[0] != np.ravel(data).shape[0]):
            raise ValueError('The file {0} is\'t'
                             ' a column vector'.format(par['TITL']))
        else:
            data = np.ravel(data)
            # First Phase the data
            new_data, _ = automatic_phase(vector=data, pivot1=int(
                data.shape[0]/2), funcmodel='minfunc')
            data_real = new_data.real
            data_imag = new_data.imag
            data_real_new, _, _, _ = basecorr1D(
                x=x, y=data_real, polyorder=polyorder, window=window)
            data_imag_new, _, _, _ = basecorr1D(
                x=x, y=data_imag, polyorder=polyorder, window=window)
            npts = x.shape[0]
            fulldata[0:npts, ncol*i] = x[0:npts, 0]
            fulldata[0:npts, ncol*i+1] = data[0:npts]
            fulldata[0:npts, ncol*i+2] = new_data[0:npts]
            fulldata[0:npts, ncol*i+3] = data_real_new+1j*data_imag_new
            dataAbs = np.absolute(fulldata[0:npts, ncol*i+3][0:npts])
            dataAbs = datasmooth(dataAbs, window_length=4,
                                 method='flat')
            fulldata[0:npts, ncol*i+4] = dataAbs
            d2 = fulldata[0:npts, ncol*i+4]
            d2_real, _, _, _ = basecorr1D(x=x, y=d2.real,
                                          polyorder=polyorder,
                                          window=window)
            d2_imag, _, _, _ = basecorr1D(x=x, y=d2.imag,
                                          polyorder=polyorder,
                                          window=window)
            d2 = d2_real + 1j*d2_imag
            fulldata[0:npts, ncol*i+5] = (d2) / np.amax(d2)
            Header = par['TITL']
    return fulldata


def OpenMultipleComplexFiles2(ListOfFiles, Scaling=None, polyorder=1,
                              window=20, *args, **kwargs):
    maxlen = MaxLengthOfFiles(ListOfFiles, *args, **kwargs)
    ncol = 7
    fulldata = np.full((maxlen, ncol*len(ListOfFiles)),
                       np.nan, dtype="complex_")
    Header = list(np.zeros((ncol*len(ListOfFiles),)))
    for file in ListOfFiles:
        i = ListOfFiles.index(file)
        data, abscissa, par = eprload(FileName=file, Scaling='n')
        if (data.shape[0] != np.ravel(data).shape[0]):
            raise ValueError(
                'The file {0} is\'t a column vector'.format(par['TITL']))
        else:
            data = np.ravel(data)
            # First Phase the data
            new_data, _ = automatic_phase(vector=data, pivot1=int(
                data.shape[0]/2), funcmodel='minfunc')
            data_real = new_data.real
            data_imag = new_data.imag
            data_real_new, _, _, _ = basecorr1D(
                x=abscissa, y=data_real, polyorder=polyorder, window=window)
            data_imag_new, _, _, _ = basecorr1D(
                x=abscissa, y=data_imag, polyorder=polyorder, window=window)
            npts = abscissa.shape[0]
            fulldata[0:npts, ncol*i] = abscissa[0:npts, 0]
            fulldata[0:npts, ncol*i+1] = data_real_new[0:npts]
            fulldata[0:npts, ncol*i+2] = data_imag_new[0:npts]
            fulldata[0:npts, ncol*i+3] = np.absolute(data_real+1j*data_imag)
            fulldata[0:npts, ncol*i+4] = datasmooth(
                data_real_new[0:npts], window_length=4, method='flat')
            fulldata[0:npts, ncol*i+5] = datasmooth(
                data_imag_new[0:npts], window_length=4, method='flat')
            fulldata[0:npts, ncol*i+6] = datasmooth(np.absolute(
                data_real+1j*data_imag), window_length=4, method='flat')
            Header[ncol*i] = par['XNAM']
            Header[ncol*i+1] = par['TITL']+str("_real")
            Header[ncol*i+2] = par['TITL']+str("_imag")
            Header[ncol*i+3] = par['TITL']+str("_abs")
            Header[ncol*i+4] = par['TITL']+str("_real_4ptsSmoothed")
            Header[ncol*i+5] = par['TITL']+str("_imag_4ptsSmoothed")
            Header[ncol*i+6] = par['TITL']+str("_abs_4ptsSmoothed")
    return np.real(fulldata), Header


def OpenDavies(ListOfFiles, Scaling=None, polyorder=0, window=20,
               *args, **kwargs):
    maxlen = MaxLengthOfFiles(ListOfFiles, *args, **kwargs)
    ncol = 7
    fulldata = np.full((maxlen, ncol*len(ListOfFiles)),
                       np.nan, dtype=float)
    Header = list(np.zeros((ncol*len(ListOfFiles),)))
    for file in ListOfFiles:
        i = ListOfFiles.index(file)
        data, x, par = eprload(FileName=file, Scaling=None)
        if (data.shape[0] != np.ravel(data).shape[0]):
            raise ValueError(
                'The file {0} is\'t a column vector'.format(par['TITL']))
        else:
            data = np.ravel(data)
            # First Phase the data
            new_data, _ = automatic_phase(vector=data, pivot1=int(
                data.shape[0]/2), funcmodel='minfunc')
            data_real = new_data.real
            data_imag = new_data.imag
            data_real_new, _, _, _ = basecorr1D(
                x=x, y=-data_real, polyorder=polyorder, window=window)
            npts = x.shape[0]
            fulldata[0:npts, ncol*i] = x[0:npts, 0]
            fulldata[0:npts, ncol*i+1] = -data_real[0:npts]
            fulldata[0:npts, ncol*i+2] = data_imag[0:npts]
            fulldata[0:npts, ncol*i+3] = np.absolute(-data_real+1j*data_imag)
            fulldata[0:npts, ncol*i+4] = data_real_new[0:npts]
            fulldata[0:npts, ncol*i+5] = datasmooth(data_real_new[0:npts],
                                                    window_length=4,
                                                    method='flat')
            d2 = fulldata[0:npts, ncol*i+5]
            fulldata[0:npts, ncol*i+6] = (d2) / np.amax(d2)
            Header[ncol*i] = par['XNAM']
            Header[ncol*i+1] = par['TITL']+str("_real")
            Header[ncol*i+2] = par['TITL']+str("_imag")
            Header[ncol*i+3] = par['TITL']+str("_abs")
            Header[ncol*i+4] = par['TITL']+str("_real_bckg")
            Header[ncol*i+5] = par['TITL']+str("_real_4ptsSmoothed")
            Header[ncol*i+6] = par['TITL']+str("_real_Normalized")
    return fulldata, Header


def OpenDavies2(ListOfFiles, Scaling=None, polyorder=0, window=20,
                *args, **kwargs):
    maxlen = MaxLengthOfFiles(ListOfFiles, *args, **kwargs)
    ncol = 7
    fulldata = np.full((maxlen, ncol*len(ListOfFiles)),
                       np.nan, dtype=float)
    Header = list(np.zeros((ncol*len(ListOfFiles),)))
    for file in ListOfFiles:
        i = ListOfFiles.index(file)
        data, x, par = eprload(FileName=file, Scaling=None)
        if (data.shape[0] != np.ravel(data).shape[0]):
            raise ValueError(
                'The file {0} is\'t a column vector'.format(par['TITL']))
        else:
            data = np.ravel(data)
            # First Phase the data
            # if data.real[0] > np.min(data.real):
            #     # in the complex raw data after phasing, and we have to correct it.
            #     data2 = -1*data
            new_data, _ = automatic_phase(vector=data, pivot1=int(
                data.shape[0]/2), funcmodel='minfunc')
            data_real = new_data.real
            data_imag = new_data.imag
            # It means that there is a 180° rotation
            # if np.argmax(data_real)<100
            #     # in the complex raw data after phasing, and we have to correct it.
            #     data_real = -1*data_real
            data_real_new, _, _, _ = basecorr1D(
                x=x, y=data_real, polyorder=polyorder, window=window)

            npts = x.shape[0]
            fulldata[0:npts, ncol*i] = x[0:npts, 0]
            fulldata[0:npts, ncol*i+1] = data_real[0:npts]
            fulldata[0:npts, ncol*i+2] = data_imag[0:npts]
            fulldata[0:npts, ncol*i+3] = np.absolute(data_real+1j*data_imag)
            fulldata[0:npts, ncol*i+4] = data_real_new[0:npts]
            fulldata[0:npts, ncol*i+5] = datasmooth(data_real_new[0:npts],
                                                    window_length=4,
                                                    method='flat')
            d2 = fulldata[0:npts, ncol*i+5]
            fulldata[0:npts, ncol*i+6] = (d2) / np.amax(d2)
            Header[ncol*i] = par['XNAM']
            Header[ncol*i+1] = par['TITL']+str("_real")
            Header[ncol*i+2] = par['TITL']+str("_imag")
            Header[ncol*i+3] = par['TITL']+str("_abs")
            Header[ncol*i+4] = par['TITL']+str("_real_bckg")
            Header[ncol*i+5] = par['TITL']+str("_real_4ptsSmoothed")
            Header[ncol*i+6] = par['TITL']+str("_real_Normalized")
    return fulldata, Header


def OpenEDNMRFiles(ListOfFiles, Scaling=None, polyorder=0, window=20,
                   *args, **kwargs):
    maxlen = MaxLengthOfFiles(ListOfFiles, *args, **kwargs)
    ncol = 6
    fulldata = np.full((maxlen, ncol*len(ListOfFiles)),
                       np.nan, dtype=float)
    snr = np.full((4, len(ListOfFiles)), np.nan, dtype=float)
    Header = list(np.zeros((ncol*len(ListOfFiles),)))
    for file in ListOfFiles:
        i = ListOfFiles.index(file)
        data, x, par = eprload(FileName=file, Scaling='n')
        if (data.shape[0] != np.ravel(data).shape[0]):
            raise ValueError(
                'The file {0} is\'t a column vector'.format(par['TITL']))
        else:
            data = -1*np.ravel(data)
            # First Phase the data
            new_data, _ = automatic_phase(vector=data, pivot1=int(
                data.shape[0]/2), funcmodel='minfunc')
            data_real = new_data.real
            data_imag = new_data.imag
            data_real_new, _, _, _ = basecorr1D(
                x=x, y=data_real, polyorder=polyorder, window=window)
            data_imag_new, _, _, _ = basecorr1D(
                x=x, y=data_imag, polyorder=polyorder, window=window)
            data_real_new = -1*data_real_new
            SNR_imag1 = np.std(data_imag_new[-400:,])/np.amax(data_real_new)
            SNR_imag2 = np.std(data_imag_new[:400,])/np.amax(data_real_new)
            SNR_real1 = np.std(data_real_new[-50:,])/np.amax(data_real_new)
            SNR_real2 = np.std(data_real_new[:50,])/np.amax(data_real_new)
            snr[0, i] = SNR_imag1
            snr[1, i] = SNR_imag2
            snr[2, i] = SNR_real1
            snr[3, i] = SNR_real2
            npts = x.shape[0]
            fulldata[0:npts, ncol*i] = x[0:npts, 0]
            fulldata[0:npts, ncol*i+1] = data_real[0:npts].real
            fulldata[0:npts, ncol*i+2] = data_imag[0:npts].real
            fulldata[0:npts, ncol*i+3] = data_real_new[0:npts].real
            fulldata[0:npts, ncol*i+4] = data_imag_new[0:npts].real
            d2 = data_real_new[0:npts].real
            fulldata[0:npts, ncol*i+5] = (d2) / np.amax(d2)
            Header[ncol*i] = 'Frequency [MHz]'
            Header[ncol*i+1] = par['TITL']+str("_real")
            Header[ncol*i+2] = par['TITL']+str("_imag")
            Header[ncol*i+3] = par['TITL']+str("_real_bckg")
            Header[ncol*i+4] = par['TITL']+str("_imag_bckg")
            Header[ncol*i+5] = par['TITL']+str("_real_Normalized")
    return fulldata, Header, snr


def OpenEDNMRFiles2(ListOfFiles, polyorder=0, window=20,
                    *args, **kwargs):
    maxlen = MaxLengthOfFiles(ListOfFiles, *args, **kwargs)
    ncol = 6
    fulldata = np.full((maxlen, ncol*len(ListOfFiles)),
                       np.nan, dtype=float)
    snr = np.full((4, len(ListOfFiles)), np.nan, dtype=float)
    Header = list(np.zeros((ncol*len(ListOfFiles),)))
    # achieve frequecy correction with the field position of the first spectrum
    _, _, par1 = eprload(ListOfFiles[0])
    field1 = par1['CenterField'][0]
    for file in ListOfFiles:
        i = ListOfFiles.index(file)
        data, x, par = eprload(FileName=file, Scaling='n')
        if (data.shape[0] != np.ravel(data).shape[0]):
            raise ValueError(
                'The file {0} is\'t a column vector'.format(par['TITL']))
        else:
            data = np.ravel(data)
            # First Phase the data
            new_data, _ = automatic_phase(vector=data, pivot1=int(
                data.shape[0]/2), funcmodel='minfunc')
            data_real = new_data.real
            data_imag = new_data.imag
            data_real_new, _, _, _ = basecorr1D(
                x=x, y=data_real, polyorder=polyorder, window=window)
            data_imag_new, _, _, _ = basecorr1D(
                x=x, y=data_imag, polyorder=polyorder, window=window)
            data_real_new = -1*data_real_new
            SNR_imag1 = np.std(data_imag_new[-400:,])/np.amax(data_real_new)
            SNR_imag2 = np.std(data_imag_new[:400,])/np.amax(data_real_new)
            SNR_real1 = np.std(data_real_new[-50:,])/np.amax(data_real_new)
            SNR_real2 = np.std(data_real_new[:50,])/np.amax(data_real_new)
            snr[0, i] = SNR_imag1
            snr[1, i] = SNR_imag2
            snr[2, i] = SNR_real1
            snr[3, i] = SNR_real2
            npts = x.shape[0]
            # achieve frequency correction with the difference in field
            # between the first spectrum and the experimental spectrum
            field2 = par['CenterField'][0]
            fulldata[0:npts, ncol*i] = x[0:npts, 0]/field2*field1  # a field
            # ratio is applied here.
            fulldata[0:npts, ncol*i+1] = data_real[0:npts].real
            fulldata[0:npts, ncol*i+2] = data_imag[0:npts].real
            fulldata[0:npts, ncol*i+3] = data_real_new[0:npts].real
            fulldata[0:npts, ncol*i+4] = data_imag_new[0:npts].real
            d2 = data_real_new[0:npts].real
            fulldata[0:npts, ncol*i+5] = (d2) / np.amax(d2)
            Header[ncol*i] = 'Frequency [MHz]'
            Header[ncol*i+1] = par['TITL']+str("_real")
            Header[ncol*i+2] = par['TITL']+str("_imag")
            Header[ncol*i+3] = par['TITL']+str("_real_bckg")
            Header[ncol*i+4] = par['TITL']+str("_imag_bckg")
            Header[ncol*i+5] = par['TITL']+str("_real_Normalized")
    return fulldata, Header, snr


def OpenOneEseemFile(FullFileName, window=20, poly=5, mode='poly', win='exp+',
                     zerofilling=5, *args, **kwargs):
    # epr doesn
    data, x, par = eprload(FileName=FullFileName)
    ncol = data.shape[1]
    npts = data.shape[0]
    npts2 = zerofilling*npts
    fileID = path.split(FullFileName)
    Data_Time = np.full((npts, 5*ncol), np.nan, dtype="float")
    Data_Freq = np.full((npts2, 5*ncol), np.nan, dtype="float")
    Header_Time = list(np.zeros((5*ncol,)))
    Header_Freq = list(np.zeros((5*ncol,)))
    tmax = x[npts-1, 0]
    for i in range(data.shape[1]):
        pivot = int(np.floor(data.shape[0]))
        new_data, _ = automatic_phase(vector=data[:, i], pivot1=pivot,
                                      funcmodel='minfunc')
        Data_Time[0:npts, 5*i] = x[:, 0].ravel()
        # /np.amax(yreal.ravel())
        new_data = data[:, i]
        # new_data.real.ravel()
        Data_Time[0:npts, 5*i+1] = data[:, i].real.ravel()
        # new_data.imag.ravel()
        Data_Time[0:npts, 5*i+2] = data[:, i].imag.ravel()
        if mode == 'polyexp':
            if np.abs(new_data.real).any() == new_data.real.any():
                yexp = np.log(new_data.real.ravel())
            else:
                yexp = np.log(np.abs(new_data).real.ravel())
        else:
            yexp = new_data.real.ravel()
        # Subtract the background:

        def monoexp(x, a, b, c):
            y = a + (b)*(np.exp((-1.0)*(x / c)))
            return y
        if mode == 'polyexp':
            c, stats = pl.polyfit(x[:, 0].ravel(), yexp, deg=poly, full=True)
            ypoly = pl.polyval(x[:, 0].ravel(), c)
            Data_Time[0:npts, 5*i+3] = np.exp(ypoly).ravel()
            yeseem = np.exp(yexp - ypoly)-1
        elif mode == 'poly':
            c, stats = pl.polyfit(x[:, 0].ravel(), yexp, deg=poly, full=True)
            ypoly = pl.polyval(x[:, 0].ravel(), c)
            Data_Time[0:npts, 5*i+3] = ypoly.ravel()
            yeseem = yexp - ypoly
        elif mode == 'monoexp':
            x01 = [10000, 400000, 0.0001]
            # x01 = [1e8, 1e8, 1e8]
            ub1 = [1e11, 1e11, 1e11]
            lb1 = [0, 0, 0]
            b1 = (lb1, ub1)
            # Monoexponential function definition
            popt1, pcov1 = curve_fit(monoexp, x[:, 0].ravel(), new_data.real.ravel(),
                                     p0=x01, sigma=None, absolute_sigma=False,
                                     check_finite=None, bounds=b1)
            perr1 = np.sqrt(np.diag(pcov1))
            yfit1 = monoexp(x[:, 0].ravel(), popt1[0], popt1[1], popt1[2])
            Data_Time[0:npts, 5*i+3] = yfit1.ravel()
            yeseem = yexp-yfit1
        yeseem = yeseem-np.mean(yeseem[-20:,])
        # error_parameters, _ = error_vandermonde(x, residuals=stats[0], rank=3)
        Data_Time[0:npts, 5*i+4] = yeseem.ravel()
        newt = np.linspace(0, tmax*5, 5*npts)/1000  # Time axis in us
        freq = fdaxis(TimeAxis=newt)  # Frequency axis in MHz
        win = windowing(window_type='ham+', N=npts)
        y2 = np.zeros((npts*5,), dtype="complex_")  # zerofilling
        y2[0:npts,] = yeseem[0:npts,]*win[0:npts,]
        data_fft = np.fft.fftshift(np.fft.fft(y2))
        Pivot = int(np.floor(data_fft.shape[0]/2))
        data_fft_Phased, _ = automatic_phase(vector=data_fft, pivot1=Pivot,
                                             funcmodel='minfunc')
        Data_Freq[0:npts2, 5*i] = freq
        Data_Freq[0:npts2, 5*i+1] = data_fft_Phased.real.ravel()
        Data_Freq[0:npts2, 5*i+2] = data_fft_Phased.imag.ravel()
        d2 = np.absolute(data_fft_Phased).ravel()
        d2 = d2-np.mean(d2[-100:])
        Data_Freq[0:npts2, 5*i+3] = d2
        Data_Freq[0:npts2, 5*i+4] = d2/np.amax(d2)
        Header_Time[5*i] = 'Time [ns]'
        Header_Time[5*i+1] = fileID[1]+'_Realpart_tau={0}ns'.format(x[i, 1])
        Header_Time[5*i+2] = fileID[1]+'_Imagpart_tau={0}ns'.format(x[i, 1])
        Header_Time[5*i+3] = 'Background ExpPoly fitting at {0}th '\
            'order'.format(poly)
        Header_Time[5*i+4] = 'Backgrounded subtracted data_tau={0}ns'\
            'order'.format(x[i, 1])
        Header_Freq[5*i] = 'Frequency [MHz]'
        Header_Freq[5*i+1] = fileID[1] + \
            '_FFTReal_Phased_tau={0}ns'.format(x[i, 1])
        Header_Freq[5*i+2] = fileID[1] + \
            '_FFTImag_Phased_tau={0}ns'.format(x[i, 1])
        Header_Freq[5*i+3] = fileID[1] + \
            '_FFTAbs_Phased_tau={0}ns'.format(x[i, 1])
        Header_Freq[5*i+4] = 'FFTAbs_Normalized_tau={0}ns'.format(x[i, 1])
    return Data_Time, Header_Time, Data_Freq, Header_Freq


def OpenOneEseemFile(FullFileName, window=20, poly=5, mode='poly', win='exp+',
                     zerofilling=5, *args, **kwargs):
    # epr doesn
    data, x, par = eprload(FileName=FullFileName)
    ncol = data.shape[1]
    npts = data.shape[0]
#    data =
    npts2 = zerofilling*npts
    fileID = path.split(FullFileName)
    Data_Time = np.full((npts, 5*ncol), np.nan, dtype="float")
    Data_Freq = np.full((npts2, 5*ncol), np.nan, dtype="float")
    Header_Time = list(np.zeros((5*ncol,)))
    Header_Freq = list(np.zeros((5*ncol,)))
    tmax = x[npts-1, 0]
    for i in range(data.shape[1]):
        pivot = int(np.floor(data.shape[0])/2)
        new_data, _ = automatic_phase(vector=data[:, i], pivot1=pivot,
                                      funcmodel='minfunc')
        Data_Time[0:npts, 5*i] = x[:, 0].ravel()
        # /np.amax(yreal.ravel())
        new_data = data[:, i]
        # new_data.real.ravel()
        Data_Time[0:npts, 5*i+1] = data[:, i].real.ravel()
        # new_data.imag.ravel()
        Data_Time[0:npts, 5*i+2] = data[:, i].imag.ravel()
        if mode == 'polyexp':
            if np.abs(new_data.real).any() == new_data.real.any():
                yexp = np.log(new_data.real.ravel())
            else:
                yexp = np.log(np.abs(new_data).real.ravel())
        else:
            yexp = new_data.real.ravel()
        # Subtract the background:

        def monoexp(x, a, b, c):
            y = a + (b)*(np.exp((-1.0)*(x / c)))
            return y
        if mode == 'polyexp':
            c, stats = pl.polyfit(x[:, 0].ravel(), yexp, deg=poly, full=True)
            ypoly = pl.polyval(x[:, 0].ravel(), c)
            Data_Time[0:npts, 5*i+3] = np.exp(ypoly).ravel()
            yeseem = np.exp(yexp - ypoly)-1
        elif mode == 'poly':
            c, stats = pl.polyfit(x[:, 0].ravel(), yexp, deg=poly, full=True)
            ypoly = pl.polyval(x[:, 0].ravel(), c)
            Data_Time[0:npts, 5*i+3] = ypoly.ravel()
            yeseem = yexp - ypoly
        elif mode == 'monoexp':
            # x01 = [10000, 400000, 0.0001]
            x01 = [1e8, 1e8, 1e8]
            ub1 = [1e11, 1e11, 1e11]
            lb1 = [0, 0, 0]
            b1 = (lb1, ub1)
            # Monoexponential function definition
            popt1, pcov1 = curve_fit(monoexp, x[:, 0].ravel(), new_data.real.ravel(),
                                     p0=x01, sigma=None, absolute_sigma=False,
                                     check_finite=None, bounds=b1)
            perr1 = np.sqrt(np.diag(pcov1))
            yfit1 = monoexp(x[:, 0].ravel(), popt1[0], popt1[1], popt1[2])
            Data_Time[0:npts, 5*i+3] = yfit1.ravel()
            yeseem = yexp-yfit1
        yeseem = yeseem-np.mean(yeseem[-20:,])
        # error_parameters, _ = error_vandermonde(x, residuals=stats[0], rank=3)
        Data_Time[0:npts, 5*i+4] = yeseem.ravel()
        newt = np.linspace(0, tmax*5, 5*npts)/1000  # Time axis in us
        freq = fdaxis(TimeAxis=newt)  # Frequency axis in MHz
        win = windowing(window_type='ham+', N=npts)
        y2 = np.zeros((npts*5,), dtype="complex_")  # zerofilling
        y2[0:npts,] = yeseem[0:npts,]*win[0:npts,]
        data_fft = np.fft.fftshift(np.fft.fft(y2))
        Pivot = int(np.floor(data_fft.shape[0]/2))
        data_fft_Phased, _ = automatic_phase(vector=data_fft, pivot1=Pivot,
                                             funcmodel='minfunc')
        Data_Freq[0:npts2, 5*i] = freq
        Data_Freq[0:npts2, 5*i+1] = data_fft_Phased.real.ravel()
        Data_Freq[0:npts2, 5*i+2] = data_fft_Phased.imag.ravel()
        d2 = np.absolute(data_fft_Phased).ravel()
        d2 = d2-np.mean(d2[-100:])
        Data_Freq[0:npts2, 5*i+3] = d2
        Data_Freq[0:npts2, 5*i+4] = d2/np.amax(d2)
        Header_Time[5*i] = 'Time [ns]'
        Header_Time[5*i+1] = fileID[1]+'_Realpart_tau={0}ns'.format(x[i, 1])
        Header_Time[5*i+2] = fileID[1]+'_Imagpart_tau={0}ns'.format(x[i, 1])
        Header_Time[5*i+3] = 'Background ExpPoly fitting at {0}th '\
            'order'.format(poly)
        Header_Time[5*i+4] = 'Backgrounded subtracted data_tau={0}ns'\
            'order'.format(x[i, 1])
        Header_Freq[5*i] = 'Frequency [MHz]'
        Header_Freq[5*i+1] = fileID[1] + \
            '_FFTReal_Phased_tau={0}ns'.format(x[i, 1])
        Header_Freq[5*i+2] = fileID[1] + \
            '_FFTImag_Phased_tau={0}ns'.format(x[i, 1])
        Header_Freq[5*i+3] = fileID[1] + \
            '_FFTAbs_Phased_tau={0}ns'.format(x[i, 1])
        Header_Freq[5*i+4] = 'FFTAbs_Normalized_tau={0}ns'.format(x[i, 1])
    return Data_Time, Header_Time, Data_Freq, Header_Freq


def OpenOneEseemFileField(FullFileName, window=20, poly=5, mode='poly', win='exp+',
                          zerofilling=5, *args, **kwargs):
    # epr doesn
    data, x, par = eprload(FileName=FullFileName)
    ncol = data.shape[1]
    npts = data.shape[0]
    npts2 = zerofilling*npts
    fileID = path.split(FullFileName)
    Data_Time = np.full((npts, 5*ncol), np.nan, dtype="float")
    Data_Freq = np.full((npts2, 5*ncol), np.nan, dtype="float")
    Header_Time = list(np.zeros((5*ncol,)))
    Header_Freq = list(np.zeros((5*ncol,)))
    tmax = x[npts-1, 0]
    for i in range(data.shape[1]):
        pivot = int(np.floor(data.shape[0]/2))
        new_data, _ = automatic_phase(vector=data[:, i], pivot1=pivot,
                                      funcmodel='minfunc')
        Data_Time[0:npts, 5*i] = x[:, 0].ravel()
        # /np.amax(yreal.ravel())
        Data_Time[0:npts, 5*i+1] = new_data.real.ravel()
        Data_Time[0:npts, 5*i+2] = new_data.imag.ravel()
        if np.abs(new_data.real).any() == new_data.real.any():
            yexp = np.log(new_data.real.ravel())
        else:
            yexp = np.log(np.abs(new_data).real.ravel())

        def monoexp(x, a, b, c):
            y = a + (b)*(np.exp((-1.0)*(x / c)))
            return y
        # Subtract the background:
        if mode == 'poly':
            c, stats = pl.polyfit(x[:, 0].ravel(), yexp, deg=poly, full=True)
            ypoly = pl.polyval(x[:, 0].ravel(), c)
            Data_Time[0:npts, 5*i+3] = ypoly.ravel()
            yeseem = np.exp(yexp - ypoly)-1
        elif mode == 'monoexp':
            x01 = [1000, 8e5, 100]
            # x01 = [1e8, 1e8, 1e8]
            ub1 = [1e11, 1e11, 1e11]
            lb1 = [0, 0, 0]
            b1 = (lb1, ub1)
            # Monoexponential function definition
            popt1, pcov1 = curve_fit(monoexp, x[:, 0].ravel(), new_data.real.ravel(),
                                     p0=x01, sigma=None, absolute_sigma=False,
                                     check_finite=None, bounds=b1)
            perr1 = np.sqrt(np.diag(pcov1))
            yfit1 = monoexp(x[:, 0].ravel(), popt1[0], popt1[1], popt1[2])
            yeseem = yexp-yfit1
        yeseem = yeseem-np.mean(yeseem[-20:,])
        # error_parameters, _ = error_vandermonde(x, residuals=stats[0], rank=3)
        Data_Time[0:npts, 5*i+4] = yeseem.ravel()
        newt = np.linspace(0, tmax*5, 5*npts)/1000  # Time axis in us
        freq = fdaxis(TimeAxis=newt)  # Frequency axis in MHz
        win = windowing(window_type='exp+', N=npts, alpha=3)
        y2 = np.zeros((npts*5,), dtype=float)  # zerofilling
        y2[0:npts,] = yeseem[0:npts,]*win[0:npts,]
        data_fft = np.fft.fftshift(np.fft.fft(y2))
        Pivot = int(np.floor(data_fft.shape[0]/2))
        data_fft_Phased, _ = automatic_phase(vector=data_fft, pivot1=Pivot,
                                             funcmodel='minfunc')
        Data_Freq[0:npts2, 5*i] = freq
        Data_Freq[0:npts2, 5*i+1] = data_fft_Phased.real.ravel()
        Data_Freq[0:npts2, 5*i+2] = data_fft_Phased.imag.ravel()
        d2 = np.absolute(data_fft_Phased).ravel()
        d2 = d2-np.mean(d2[-100:])
        Data_Freq[0:npts2, 5*i+3] = d2
        Data_Freq[0:npts2, 5*i+4] = d2/np.amax(d2)
        Header_Time[5*i] = 'Time [ns]'
        Header_Time[5*i+1] = fileID[1]+'_Realpart_Field={0}G'.format(x[i, 1])
        Header_Time[5*i+2] = fileID[1]+'_Imagpart_Field={0}G'.format(x[i, 1])
        Header_Time[5*i+3] = 'Background ExpPoly fitting at {0}th '\
            'order'.format(poly)
        Header_Time[5*i+4] = 'Backgrounded subtracted data_Field={0}G'\
            'order'.format(x[i, 1])
        Header_Freq[5*i] = 'Frequency [MHz]'
        Header_Freq[5*i+1] = fileID[1] + \
            '_FFTReal_Phased_Field={0}G'.format(x[i, 1])
        Header_Freq[5*i+2] = fileID[1] + \
            '_FFTImag_Phased_Field={0}G'.format(x[i, 1])
        Header_Freq[5*i+3] = fileID[1] + \
            '_FFTAbs_Phased_tau={0}ns'.format(x[i, 1])
        Header_Freq[5*i+4] = 'FFTAbs_Normalized_Field={0}G'.format(x[i, 1])
    return Data_Time, Header_Time, Data_Freq, Header_Freq


def OpenOneEseemFile2(FullFileName, window=20, poly=5, mode='poly', win='exp+',
                      zerofilling=5, *args, **kwargs):
    # epr doesn
    data2, x2, par = eprload(FileName=FullFileName)
    ninit = 1
    data = data2[ninit:,]
    x = x2[ninit:,]
    ncol = data.shape[1]
    npts = data.shape[0]
#    data =
    npts2 = zerofilling*npts
    fileID = path.split(FullFileName)
    Data_Time = np.full((npts, 5*ncol), np.nan, dtype="float")
    Data_Freq = np.full((npts2, 5*ncol), np.nan, dtype="float")
    Header_Time = list(np.zeros((5*ncol,)))
    Header_Freq = list(np.zeros((5*ncol,)))
    tmax = x[npts-1, 0]
    for i in range(data.shape[1]):
        pivot = int(np.floor(data.shape[0])/2)
        new_data, _ = automatic_phase(vector=data[:, i], pivot1=pivot,
                                      funcmodel='minfunc')
        Data_Time[0:npts, 5*i] = x[:, 0].ravel()
        # /np.amax(yreal.ravel())
        # new_data.real.ravel()
        Data_Time[0:npts, 5*i+1] = new_data[:, i].real.ravel()
        # new_data.imag.ravel()
        Data_Time[0:npts, 5*i+2] = new_data[:, i].imag.ravel()
        if mode == 'polyexp':
            if np.abs(new_data.real).any() == new_data.real.any():
                yexp = np.log(new_data.real.ravel())
            else:
                yexp = np.log(np.abs(new_data).real.ravel())
        else:
            yexp = new_data.real.ravel()
        # Subtract the background:

        def monoexp(x, a, b, c):
            y = a + (b)*(np.exp((-1.0)*(x / c)))
            return y
        if mode == 'polyexp':
            c, stats = pl.polyfit(x[:, 0].ravel(), yexp, deg=poly, full=True)
            ypoly = pl.polyval(x[:, 0].ravel(), c)
            Data_Time[0:npts, 5*i+3] = np.exp(ypoly).ravel()
            yeseem = np.exp(yexp - ypoly)-1
        elif mode == 'poly':
            c, stats = pl.polyfit(x[:, 0].ravel(), yexp, deg=poly, full=True)
            ypoly = pl.polyval(x[:, 0].ravel(), c)
            Data_Time[0:npts, 5*i+3] = ypoly.ravel()
            yeseem = yexp - ypoly
        elif mode == 'monoexp':
            x01 = [10000, 400000, 0.0001]
            # x01 = [1e8, 1e8, 1e8]
            ub1 = [1e11, 1e11, 1e11]
            lb1 = [0, 0, 0]
            b1 = (lb1, ub1)
            # Monoexponential function definition
            popt1, pcov1 = curve_fit(monoexp, x[:, 0].ravel(), new_data.real.ravel(),
                                     p0=x01, sigma=None, absolute_sigma=False,
                                     check_finite=None, bounds=b1)
            perr1 = np.sqrt(np.diag(pcov1))
            yfit1 = monoexp(x[:, 0].ravel(), popt1[0], popt1[1], popt1[2])
            Data_Time[0:npts, 5*i+3] = yfit1.ravel()
            yeseem = yexp-yfit1
        yeseem = yeseem-np.mean(yeseem[-20:,])
        # error_parameters, _ = error_vandermonde(x, residuals=stats[0], rank=3)
        Data_Time[0:npts, 5*i+4] = yeseem.ravel()
        newt = np.linspace(0, tmax*5, 5*npts)/1000  # Time axis in us
        freq = fdaxis(TimeAxis=newt)  # Frequency axis in MHz
        win = windowing(window_type='ham+', N=npts)
        y2 = np.zeros((npts*5,), dtype="complex_")  # zerofilling
        y2[0:npts,] = yeseem[0:npts,]*win[0:npts,]
        data_fft = np.fft.fftshift(np.fft.fft(y2))
        Pivot = int(np.floor(data_fft.shape[0]/2))
        data_fft_Phased, _ = automatic_phase(vector=data_fft, pivot1=Pivot,
                                             funcmodel='minfunc')
        Data_Freq[0:npts2, 5*i] = freq
        Data_Freq[0:npts2, 5*i+1] = data_fft_Phased.real.ravel()
        Data_Freq[0:npts2, 5*i+2] = data_fft_Phased.imag.ravel()
        d2 = np.absolute(data_fft_Phased).ravel()
        d2 = d2-np.mean(d2[-100:])
        Data_Freq[0:npts2, 5*i+3] = d2
        Data_Freq[0:npts2, 5*i+4] = d2/np.amax(d2)
        Header_Time[5*i] = 'Time [ns]'
        Header_Time[5*i+1] = fileID[1]+'_Realpart_tau={0}ns'.format(x[i, 1])
        Header_Time[5*i+2] = fileID[1]+'_Imagpart_tau={0}ns'.format(x[i, 1])
        Header_Time[5*i+3] = 'Background ExpPoly fitting at {0}th '\
            'order'.format(poly)
        Header_Time[5*i+4] = 'Backgrounded subtracted data_tau={0}ns'\
            'order'.format(x[i, 1])
        Header_Freq[5*i] = 'Frequency [MHz]'
        Header_Freq[5*i+1] = fileID[1] + \
            '_FFTReal_Phased_tau={0}ns'.format(x[i, 1])
        Header_Freq[5*i+2] = fileID[1] + \
            '_FFTImag_Phased_tau={0}ns'.format(x[i, 1])
        Header_Freq[5*i+3] = fileID[1] + \
            '_FFTAbs_Phased_tau={0}ns'.format(x[i, 1])
        Header_Freq[5*i+4] = 'FFTAbs_Normalized_tau={0}ns'.format(x[i, 1])
    return Data_Time, Header_Time, Data_Freq, Header_Freq


def OpenRidmeFile(FullFileName, window=20, poly=5, mode='poly', *args, **kwargs):
    data, x, par = eprload(FileName=FullFileName)
    # ninit = 1
    # data = data2[ninit:,]
    # x = x2[ninit:,]
    ncol = data.shape[1]
    npts = data.shape[0]
    fileID = path.split(FullFileName)
    Data_Time = np.full((npts, 5*ncol), np.nan, dtype="float")
    Header_Time = list(np.zeros((5*ncol,)))
    tmax = x[npts-1, 0]
    pivot = int(np.floor(data.shape[0])/2)
    data0, _ = automatic_phase(vector=data[:, 0], pivot1=pivot,
                               funcmodel='minfunc')
    for i in range(data.shape[1]):

        new_data, _ = automatic_phase(vector=data[:, i], pivot1=pivot,
                                      funcmodel='minfunc')

        Data_Time[0:npts, 5*i] = x[:, 0].ravel()
        Data_Time[0:npts, 5*i+1] = new_data[:, ].real.ravel()
        new_data2 = (new_data[:, ].real.ravel()/data0[:, ].real.ravel())
        Data_Time[0:npts, 5*i+2] = new_data[:,].imag.ravel()
        if mode == 'polyexp':
            if np.abs(new_data.real).any() == new_data.real.any():
                yexp = np.log(new_data.real.ravel())
            else:
                yexp = np.log(np.abs(new_data).real.ravel())
        else:
            yexp = new_data2[:, ].real.ravel(
            )/np.max(new_data2[:, ].real.ravel())
        # Subtract the background on the second half:
        init = int(np.floor(npts/4))

        def monoexp(x, a, b, c):
            y = a + (b)*(np.exp((-1.0)*(x / c)))
            return y
        if mode == 'polyexp':
            c, stats = pl.polyfit(
                x[init:, 0].ravel(), yexp[init:,], deg=poly, full=True)
            ypoly = pl.polyval(x[:, 0].ravel(), c)
            Data_Time[0:npts, 5*i+3] = np.exp(ypoly).ravel()
            yfinal = np.exp(yexp - ypoly)-1
        elif mode == 'poly':
            c, stats = pl.polyfit(
                x[init:, 0].ravel(), yexp[init:,], deg=poly, full=True)
            ypoly = pl.polyval(x[:, 0].ravel(), c)
            Data_Time[0:npts, 5*i+3] = ypoly.ravel()
            yfinal = yexp - ypoly
        elif mode == 'monoexp':
            x01 = [10000, 400000, 0.0001]
            # x01 = [1e8, 1e8, 1e8]
            ub1 = [1e11, 1e11, 1e11]
            lb1 = [0, 0, 0]
            b1 = (lb1, ub1)
            # Monoexponential function definition
            popt1, pcov1 = curve_fit(monoexp, x[init:, 0].ravel(), yexp[init:,],
                                     p0=x01, sigma=None, absolute_sigma=False,
                                     check_finite=None, bounds=b1)
            perr1 = np.sqrt(np.diag(pcov1))
            yfit1 = monoexp(x[:, 0].ravel(), popt1[0], popt1[1], popt1[2])
            Data_Time[0:npts, 5*i+3] = yfit1.ravel()
            yfinal = yexp-yfit1
        yfinal = yfinal-np.mean(yfinal[-20:,])
        Data_Time[0:npts, 5*i+4] = yfinal.ravel()
        Header_Time[5*i] = 'Time [ns]'
        Header_Time[5*i+1] = fileID[1] + \
            '_Realpart_Tm={0}us'.format(10+5*x[i, 1])
        Header_Time[5*i+2] = fileID[1] + \
            '_Imagpart_Tm={0}us'.format(10+5*x[i, 1])
        Header_Time[5*i+3] = 'Background ExpPoly fitting at {0}th '\
            'order'.format(poly)
        Header_Time[5*i+4] = 'Backgrounded subtracted data_tau={0}ns'\
            'order'.format(x[i, 1])
    return Data_Time, Header_Time
