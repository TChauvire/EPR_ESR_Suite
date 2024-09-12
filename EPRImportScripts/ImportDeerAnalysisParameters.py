# -*- coding: utf-8 -*-
"""
Created on Tue Oct 17 12:37:19 2023

@author: tim_t

General toolbox scripts for 1) importing DEER Bruker data, 2) autophase the
complex data 3) generate background subtraction, 4) importing result
from the DEER analysis matlab software, 5) analyze distance domain data,
6) generate table for publications, 7) ...

Scripts to import selected information from DEER_Analysis software
(https://epr.ethz.ch/software.html) to construct the table for the publication:
    [ref to insert]
The script rely on the EPR_ESR Suite:
(https://github.com/TChauvire/EPR_ESR_Suite).

# For the data :
# 1) First Column is time
# 2) Second column is the raw data (complex))
# 3) Third column is the phase corrected data (complex))
# 4) Fourth column is the real part of the phase corrected data (real))
# 5) Fifth column are the background data (real)
# 5) Sixth column would be the background subtracted data (real)
# To call a column, you have to use the title of the data to access
TITL = path.split(Filename)[1]
# the global table : datatable = DataDictionnary.get(TITL)

For the parameters :
1)
2) Noise level estimated from the root-mean-square
amplitude of the imaginary part after phase correction.
Parameters

"""

from os import walk, getcwd, path, makedirs
import re
import csv
from SVD_scripts import get_KUsV, process_data
from automatic_phase import automatic_phase
from basecorr1D import basecorr1D, error_vandermonde
from ImportMultipleFiles import eprload, datasmooth
from fdaxis import fdaxis
from windowing import windowing
import numpy as np
import numpy.polynomial.polynomial as pl
from scipy.signal import find_peaks, peak_widths  # find_peaks_cwt
# used for gaussian fit of the distance domain
from scipy.optimize import curve_fit  # 1) used for gaussian analysis of the
# distance domain
# 2) used for the backgnd subtraction in the time domain
import deerlab as dl
from DeerDenoising import ThresholdGuess, Deer_Denoising
#############################################################################
folder = 'C:\\Users\\tim_t\\Python\\EPRImportScripts\\DataFilesDEER\\TestFiles'
filename1 = 'C:\\Users\\tim_t\\Python\\EPRImportScripts\\DataFilesDEER\\'\
    'TestFiles\\DEER_Anthr_50K_af1+42.DSC'
filename2 = 'C:\\Users\\tim_t\\Python\\EPRImportScripts\\DataFilesDEER\\'\
    'TestFiles\\DEER_Anthr_50K_af1-66.DSC'
filename3 = 'C:\\Users\\tim_t\\Python\\EPRImportScripts\\DataFilesDEER\\'\
    'TestFiles\\M_GdnHCl_DEER_2700ns.DSC'

DataDict, ParamDict = {}, {}


def ImportMultipleDEERAsciiFiles(FolderPath=getcwd(), Extension='txt'):
    ListOfFiles = []
    for root, dirs, files in walk(FolderPath):
        for file in files:
            if file.endswith(Extension):
                ListOfFiles.append(path.normpath(path.join(root, file)))
    return ListOfFiles


#  ListOfFiles = ImportMultipleDEERAsciiFiles(folder, 'DSC')


def ExponentialCorr1D_DEER(x=None, y=None, Dimensionorder=3, Percent_tmax=9/10,
                           mode='strexp', truncsize=3/4, *args, **kwargs):
    '''
    Function that achieve a baseline correction by fitting a function
    parameterized by a streched exponential for a supposed homogeneous
    three-dimensional solution (d=3) or a stretched exponential for other
    dimensions as described by multiple publications.
    See by example : (dx.doi.org/10.1039/c9cp06111h )

    .. math::

    B(t) = \exp\left(-\kappa \vert t\vert^{d}\right)

    k is the decay rate constant of the background and d is
    the so-called fractal dimension of the exponential

    The fitting is done on the last 3/4 points of the data.
    Script written by Timothée Chauviré
    (https://github.com/TChauvire/EPR_ESR_Suite/), 10/18/2023

    Parameters
    ----------
    x : abscissa of the data, TYPE : numpy data array, column vector
        DESCRIPTION. The default is None.
    y : data which baseline has to be corrected,
        TYPE : numpy data array, column vector
        It has to have the same size than x
        DESCRIPTION. The default is None.
    Dimensionorder : order of so-called fractal dimension of the exponential
        TYPE : Integer, optional
        DESCRIPTION. The default is 0.

    Returns
    -------
    ynew : baseline data array
        TYPE: numpy data array, same shape as input data y
    (k,d) : coefficient used for the exponential fit
        TYPE : tuple of real values, coefficient of the streched exponential
    perr : error coefficient obtained from the covariance matrix
        (perr = np.sqrt(np.diag(pcov)))
        TYPE : diagonalized 2-D array
    mode='strexp', stretched exponential ackground subtraction
    mode='poly', polynomial fit of the logarithmic data
    '''
    shape = y.shape
    if x.shape[0] != np.ravel(x).shape[0]:
        raise ValueError('x must be a column vector. ExponentialCorr1D_DEER'
                         'function does not work on 2D arrays.')
    else:
        x = np.ravel(x)

    if y.shape[0] != np.ravel(y).shape[0]:
        raise ValueError('y must be a column vector. ExponentialCorr1D_DEER'
                         'function does not work on 2D arrays.')
    else:
        y = np.ravel(y)

    if y.shape[0] != x.shape[0]:
        raise ValueError('x and y must be column vector of the same size.')
    yfit = np.full(y.shape, np.nan)
    npts = x.shape[0]
    npts_new = int(np.floor(npts*truncsize))
    itmax = int(np.floor(Percent_tmax*npts))
    xfitinit = (np.array(x[(npts-npts_new):itmax])).ravel()
    # xfitinit = np.array(x[-npts_new:tmax]).ravel()
    yfit = (y/np.max(y)).ravel().real
    yfitinit = (np.array(yfit[(npts-npts_new):itmax])).ravel()
    # yfitinit = np.array(yfit[-npts_new:tmax]).ravel()

    def strexp(x, ModDepth, decay, stretch):
        a = (1-ModDepth)*(np.exp((-1)*(np.abs(decay*x))) ** (stretch/3))
        return a

    def strexp2(x, ModDepth, decay):
        a = (1-ModDepth)*(np.exp((-1)*(np.abs(decay*x))) **
                          (Dimensionorder/3))
        return a

    # # Add parameters
    p0_1 = [0.3, 0.25, Dimensionorder]
    b_1 = ([0, 0, 2], [1, 200, 6])  # (lowerbound,upperbound)  bounds=b,
    # Add parameters
    p0_2 = [0.3, 0.25]
    b_2 = ([0, 0], [1, 200])  # (lowerbound,upperbound)  bounds=b,

    if mode == 'strexp':
        poptarray, pcov = curve_fit(strexp, xfitinit, yfitinit, p0=p0_1,
                                    sigma=None, absolute_sigma=False,
                                    check_finite=None, bounds=b_1)
        perr = np.sqrt(np.diag(pcov))
        yfit2 = (strexp(x, poptarray[0],
                 poptarray[1], poptarray[2]))*np.max(y)
        yfit2 = yfit2.reshape(shape)
        return yfit2, poptarray, perr
    if mode == 'strexp_fixed':
        poptarray, pcov = curve_fit(strexp2, xfitinit, yfitinit, p0=p0_2,
                                    sigma=None, absolute_sigma=False,
                                    check_finite=None, bounds=b_2)
        perr = np.sqrt(np.diag(pcov))
        yfit2 = (strexp2(x, poptarray[0], poptarray[1]))*np.max(y)
        poptarray = np.append(poptarray, Dimensionorder)
        yfit2 = yfit2.reshape(shape)
        return yfit2, poptarray, perr
    if mode == 'poly':
        c, stats = pl.polyfit(xfitinit, yfitinit,
                              deg=Dimensionorder, full=True)
        ypoly = pl.polyval(x, c)*np.max(y)
        error_parameters, _ = error_vandermonde(x, residuals=stats[0], rank=3)
        return ypoly, c, error_parameters
    if mode == 'polyexp':
        c, stats = pl.polyfit(xfitinit, np.log(yfitinit),
                              deg=Dimensionorder, full=True)
        ypoly = np.exp(pl.polyval(x, c))*np.max(y)
        error_parameters, _ = error_vandermonde(x, residuals=stats[0], rank=3)
        return ypoly, c, error_parameters


def BckgndSubtractionOfRealPart(Filename, DataDict, ParamDict, Scaling=None,
                                Dimensionorder=3, Percent_tmax=9/10,
                                mode='strexp', truncsize=3/4, *args, **kwargs):
    '''

    ----------
    ListOfFiles : TYPE
        DESCRIPTION.
    Scaling : TYPE, optional
        DESCRIPTION. The default is None.
    Dimensionorder : TYPE, optional
        DESCRIPTION. The default is 1.
    Percent_tmax : TYPE, optional
        DESCRIPTION. The default is 9/10.
    *args : TYPE
        DESCRIPTION.
    **kwargs : TYPE
        DESCRIPTION.

    Raises
    ------
    ValueError
        DESCRIPTION.

    Returns
    -------
    BckgndSubtractedData : TYPE
        DESCRIPTION.
    Modulation_Depth : TYPE
        DESCRIPTION.
    RelBckgndDecay : TYPE
        DESCRIPTION.
    NoiseLevel : TYPE
        DESCRIPTION.

    '''
    fileID = path.split(Filename)
    data, abscissa, par = eprload(Filename, Scaling)
    if (data.shape[0] != np.ravel(data).shape[0]):
        raise ValueError(
            'The file {0} is\'t a column vector'.format(par['TITL']))
    else:
        npts = abscissa.shape[0]
        new_data = np.full(npts, np.nan, dtype="complex_")
        pivot = int(np.floor(data.shape[0]/2))
        new_data, _ = automatic_phase(vector=data, pivot1=pivot,
                                      funcmodel='minfunc')
        data_real = np.ravel(new_data.real)
        data_imag = np.ravel(new_data.imag)
        abscissa = np.ravel(abscissa)
        # Achieve background correction of the real part :
        # newdata_real = datasmooth(
        #     data_real[0:npts], window_length=10, method='binom')
        # itmin = np.argmax(newdata_real)
        # newx = dl.correctzerotime(data_real, abscissa)
        newx = abscissa
        itmin = np.abs(newx).argmin()
        # itmin = newx([newx==0])
        if itmin > 50:  # The determination of zerotime didn't work, do nothing
            itmin = 0
            newx = abscissa
        tmin = newx[itmin]
        data_bckgnd, p0, perr = ExponentialCorr1D_DEER(x=newx, y=data_real,
                                                       Dimensionorder=Dimensionorder,
                                                       Percent_tmax=Percent_tmax,
                                                       mode=mode, truncsize=truncsize)
        w = int(np.floor(npts/2))
        # Achieve automatic base line correction correction of the imaginary
        # part :
        data_imag_new, _, _, _ = basecorr1D(x=newx, y=data_imag,
                                            polyorder=1, window=w)
        if np.floor(Percent_tmax*npts)-1 <= npts:
            itmax = int(np.floor(Percent_tmax*npts)-1)
        else:
            itmax = npts
        RelBckgndDecay = 1 - data_bckgnd[itmax] / data_bckgnd[itmin]
        FinalData = (data_real - data_bckgnd)/np.max(data_real)
        FinalData = FinalData-FinalData[itmax-20:itmax].mean()
        # BckgndSubtractedData = (data_real - data_bckgnd)/(np.max(data_real - data_bckgnd))
        # Two noises-level are computed :
        # 1) With the full imaginary part "sigma_noise"
        # 2) With half of the imaginary part "sigma_noise_half"
        center = int(np.floor(npts/2))
        sigma_noise = np.std(data_imag_new[itmin:itmax,])/np.max(data_real)
        sigma_noise_half = np.std(data_imag_new[center-int(np.floor(npts/4)):
                                                center+int(np.floor(npts/4))],
                                  )/np.max(data_real)
        NoiseLevel = (sigma_noise, sigma_noise_half)
        # Calculate the Root mean square of error
        RMSE = ComputeRMSE(data_real/np.max(data_real),
                           data_bckgnd/np.max(data_real), p0)
        # Let's create a global dictionnary for storing all the data and the
        # parameters:
        # TITL = str(par['TITL'])
        TITL = fileID[1]
        fulldata = np.full((5*npts, 50), np.nan, dtype="complex_")
        fulldata[0:npts, 0] = newx.ravel()
        fulldata[0:npts, 1] = data.ravel()
        fulldata[0:npts, 2] = new_data.ravel()
        fulldata[0:npts, 3] = new_data.real.ravel()
        fulldata[0:npts, 4] = data_bckgnd.ravel()
        fulldata[0:npts, 5] = FinalData.ravel()

        DataDict.update({TITL: fulldata})
        Header = list(np.zeros((50,)))
        Header[0] = str(par['XNAM'])
        Header[1] = str(TITL+"_rawData")
        Header[2] = str(TITL+"_phased")
        Header[3] = str(TITL+"_phased_real")
        Header[4] = str(TITL+"_background_real")
        Header[5] = str(TITL+"_backgroundsubtracted_real")
        HeaderTITL = str(TITL+'_Header')
        DataDict[HeaderTITL] = Header
        Exp_parameter = {'RelBckgndDecay': RelBckgndDecay, 'tmin':  tmin,
                         'NoiseLevel': NoiseLevel, 'tmax': abscissa[itmax],
                         'itmax': itmax, 'itmin': itmin, 'RMSE': RMSE}
        # Assign the modulation depth in the parameters
        if mode == 'strexp':
            Bckg_type = str(mode+'_'+str(Percent_tmax)+'_'+str(truncsize))
            DEER_SNR = (p0[0]/sigma_noise, p0[0]/sigma_noise_half)
            Exp_parameter.update({'ModDepth': p0[0], 'decay': p0[1],
                                  'stretch': p0[2], 'Bckg_type': Bckg_type,
                                  'DEER_SNR': DEER_SNR, 'polyparam': ''})

        if mode == 'strexp_fixed':
            Bckg_type = str(mode+'_'+str(Percent_tmax)+'_'+str(truncsize))
            DEER_SNR = (p0[0]/sigma_noise, p0[0]/sigma_noise_half)
            Exp_parameter.update({'ModDepth': p0[0], 'decay': p0[1],
                                  'stretch': p0[2], 'Bckg_type': Bckg_type,
                                  'DEER_SNR': DEER_SNR, 'polyparam': ''})
        elif mode == 'poly':
            Bckg_type = str(mode+str(len(p0)-1)+'_'+str(Percent_tmax)+'_' +
                            str(truncsize))
            Mod_Depth = FinalData[itmin-1:itmin+1].mean()
            DEER_SNR = (Mod_Depth/sigma_noise, Mod_Depth/sigma_noise_half)
            Exp_parameter.update({'ModDepth': Mod_Depth, 'polyparam': p0,
                                  'Bckg_type': Bckg_type,
                                  'DEER_SNR': DEER_SNR})
        elif mode == 'polyexp':
            Bckg_type = str(mode+str(len(p0)-1)+'_'+str(Percent_tmax)+'_' +
                            str(truncsize))
            Mod_Depth = FinalData[itmin-1:itmin+1].mean()
            DEER_SNR = (Mod_Depth/sigma_noise, Mod_Depth/sigma_noise_half)
            Exp_parameter.update({'ModDepth': Mod_Depth, 'polyparam': p0,
                                  'Bckg_type': Bckg_type,
                                  'DEER_SNR': DEER_SNR})
        ParamDict.update({TITL: Exp_parameter})
    return DataDict, ParamDict

    # def is_float(s):
    #     if s is None:
    #         return False
    #     try:
    #         float(s)
    #         return True
    #     except ValueError:
    #         return False
    # time = fulldata[:, 0].ravel()
    # y = fulldata[:, 5].ravel()
    # cleanedy = y[~np.isnan(y)][itmin:itmax].real
    # cleanedx = (time[~np.isnan(time)][itmin:itmax].real)/1000
    # newfilename = str(fileID[1]+'.txt')
    # fullnewfilename = path.join(fileID[0], newfilename)


def BckgndSubtractionOfRealPart2(Filename, DataDict, ParamDict, Scaling=None,
                                 Dimensionorder=3, Percent_tmax=9/10,
                                 mode='strexp', truncsize=3/4, zerofilling=0,
                                 *args, **kwargs):
    '''

    ----------
    ListOfFiles : TYPE
        DESCRIPTION.
    Scaling : TYPE, optional
        DESCRIPTION. The default is None.
    Dimensionorder : TYPE, optional
        DESCRIPTION. The default is 1.
    Percent_tmax : TYPE, optional
        DESCRIPTION. The default is 9/10.
    *args : TYPE
        DESCRIPTION.
    **kwargs : TYPE
        DESCRIPTION.

    Raises
    ------
    ValueError
        DESCRIPTION.

    Returns
    -------
    BckgndSubtractedData : TYPE
        DESCRIPTION.
    Modulation_Depth : TYPE
        DESCRIPTION.
    RelBckgndDecay : TYPE
        DESCRIPTION.
    NoiseLevel : TYPE
        DESCRIPTION.

    '''
    fileID = path.split(Filename)
    data, abscissa, par = eprload(Filename, Scaling)
    if (data.shape[0] != np.ravel(data).shape[0]):
        raise ValueError(
            'The file {0} is\'t a column vector'.format(par['TITL']))
    else:
        npts = abscissa.shape[0]
        new_data = np.full(npts, np.nan, dtype="complex_")
        pivot = int(np.floor(data.shape[0]/2))
        new_data, _ = automatic_phase(vector=data, pivot1=pivot,
                                      funcmodel='minfunc')
        data_real = np.ravel(new_data.real)
        data_imag = np.ravel(new_data.imag)
        abscissa = np.ravel(abscissa)
        # Achieve background correction of the real part :
        # newdata_real = datasmooth(
        #     data_real[0:npts], window_length=10, method='binom')
        # itmin = np.argmax(newdata_real)
        newx = dl.correctzerotime(data_real, abscissa)
        itmin = np.abs(newx).argmin()
        # itmin = newx([newx==0])
        if itmin > 50:  # The determination of zerotime didn't work, do nothing
            itmin = 0
            newx = abscissa
        tmin = newx[itmin]
        data_bckgnd, p0, perr = ExponentialCorr1D_DEER(x=newx, y=data_real,
                                                       Dimensionorder=Dimensionorder,
                                                       Percent_tmax=Percent_tmax,
                                                       mode=mode, truncsize=truncsize)
        w = int(np.floor(npts/2))
        # Achieve automatic base line correction correction of the imaginary
        # part :
        data_imag_new, _, _, _ = basecorr1D(x=newx, y=data_imag,
                                            polyorder=1, window=w)
        if np.floor(Percent_tmax*npts)-1 <= npts:
            itmax = int(np.floor(Percent_tmax*npts)-1)
        else:
            itmax = npts
        RelBckgndDecay = 1 - data_bckgnd[itmax] / data_bckgnd[itmin]
        FinalData = (data_real - data_bckgnd)/np.max(data_real)
        FinalData = FinalData-FinalData[itmax-20:itmax].mean()
        # BckgndSubtractedData = (data_real - data_bckgnd)/(np.max(data_real - data_bckgnd))
        # Two noises-level are computed :
        # 1) With the full imaginary part "sigma_noise"
        # 2) With half of the imaginary part "sigma_noise_half"
        center = int(np.floor(npts/2))
        sigma_noise = np.std(data_imag_new[itmin:itmax,])/np.max(data_real)
        sigma_noise_half = np.std(data_imag_new[center-int(np.floor(npts/4)):
                                                center+int(np.floor(npts/4))],
                                  )/np.max(data_real)
        NoiseLevel = (sigma_noise, sigma_noise_half)
        # Calculate the Root mean square of error
        RMSE = ComputeRMSE(data_real/np.max(data_real),
                           data_bckgnd/np.max(data_real), p0)

        # Let's create a global dictionnary for storing all the data and the
        # parameters:
        # TITL = str(par['TITL'])
        TITL = fileID[1]
        if zerofilling == 0:
            fulldata = np.full((5*npts, 50), np.nan, dtype="complex_")
            fulldata[0:npts, 0] = newx.ravel()
            fulldata[0:npts, 1] = data.ravel()
            fulldata[0:npts, 2] = new_data.ravel()
            fulldata[0:npts, 3] = new_data.real.ravel()
            fulldata[0:npts, 4] = data_bckgnd.ravel()
            fulldata[0:npts, 5] = FinalData.ravel()

            DataDict.update({TITL: fulldata})
            Header = list(np.zeros((50,)))
            Header[0] = str(par['XNAM'])
            Header[1] = str(TITL+"_rawData")
            Header[2] = str(TITL+"_phased")
            Header[3] = str(TITL+"_phased_real")
            Header[4] = str(TITL+"_background_real")
            Header[5] = str(TITL+"_backgroundsubtracted_real")
            HeaderTITL = str(TITL+'_Header')
            DataDict[HeaderTITL] = Header
            Exp_parameter = {'RelBckgndDecay': RelBckgndDecay, 'tmin':  tmin,
                             'NoiseLevel': NoiseLevel, 'tmax': abscissa[itmax],
                             'itmax': itmax, 'itmin': itmin, 'RMSE': RMSE}
        elif float(zerofilling) > 0:
            totnpts = int(np.floor(zerofilling))+itmax
            dx = newx[1]-newx[0]
            tmax = newx[0]+dx*totnpts
            newx2 = np.linspace(newx[0], tmax, totnpts)
            FinalData2 = np.zeros((totnpts,), dtype="complex_")
            FinalData2[0:itmax,] = FinalData[0:itmax,]
            itmax = int(np.floor(zerofilling))+itmax
            fulldata = np.full((5*totnpts, 50), np.nan, dtype="complex_")
            fulldata[0:totnpts, 0] = newx2.ravel()
            fulldata[0:npts, 1] = data.ravel()
            fulldata[0:npts, 2] = new_data.ravel()
            fulldata[0:npts, 3] = new_data.real.ravel()
            fulldata[0:npts, 4] = data_bckgnd.ravel()
            fulldata[0:totnpts, 5] = FinalData2.ravel()

            DataDict.update({TITL: fulldata})
            Header = list(np.zeros((50,)))
            Header[0] = str(par['XNAM'])
            Header[1] = str(TITL+"_rawData")
            Header[2] = str(TITL+"_phased")
            Header[3] = str(TITL+"_phased_real")
            Header[4] = str(TITL+"_background_real")
            Header[5] = str(TITL+"_backgroundsubtracted_real")
            HeaderTITL = str(TITL+'_Header')
            DataDict[HeaderTITL] = Header
            Exp_parameter = {'RelBckgndDecay': RelBckgndDecay, 'tmin':  tmin,
                             'NoiseLevel': NoiseLevel, 'tmax': tmax,
                             'itmax': itmax, 'itmin': itmin, 'RMSE': RMSE}
        # Assign the modulation depth in the parameters
        if mode == 'strexp':
            Bckg_type = str(mode+'_'+str(Percent_tmax)+'_'+str(truncsize))
            DEER_SNR = (p0[0]/sigma_noise, p0[0]/sigma_noise_half)
            Exp_parameter.update({'ModDepth': p0[0], 'decay': p0[1],
                                  'stretch': p0[2], 'Bckg_type': Bckg_type,
                                  'DEER_SNR': DEER_SNR})

        if mode == 'strexp_fixed':
            Bckg_type = str(mode+'_'+str(Percent_tmax)+'_'+str(truncsize))
            DEER_SNR = (p0[0]/sigma_noise, p0[0]/sigma_noise_half)
            Exp_parameter.update({'ModDepth': p0[0], 'decay': p0[1],
                                  'stretch': p0[2], 'Bckg_type': Bckg_type,
                                  'DEER_SNR': DEER_SNR})
        elif mode == 'poly':
            Bckg_type = str(mode+str(len(p0)-1)+'_'+str(Percent_tmax)+'_' +
                            str(truncsize))
            Mod_Depth = FinalData[itmin-1:itmin+1].mean()
            DEER_SNR = (Mod_Depth/sigma_noise, Mod_Depth/sigma_noise_half)
            Exp_parameter.update({'ModDepth': Mod_Depth, 'polyparam': p0,
                                  'Bckg_type': Bckg_type,
                                  'DEER_SNR': DEER_SNR})
        elif mode == 'polyexp':
            Bckg_type = str(mode+str(len(p0)-1)+'_'+str(Percent_tmax)+'_' +
                            str(truncsize))
            Mod_Depth = FinalData[itmin-1:itmin+1].mean()
            DEER_SNR = (Mod_Depth/sigma_noise, Mod_Depth/sigma_noise_half)
            Exp_parameter.update({'ModDepth': Mod_Depth, 'polyparam': p0,
                                  'Bckg_type': Bckg_type,
                                  'DEER_SNR': DEER_SNR})
        ParamDict.update({TITL: Exp_parameter})
    return DataDict, ParamDict


def ManualGaussianFit(x, y, params0, bounds=None):
    '''
    Achieve the gaussian fit of data.
    The script was designed for distance domain experimental data obtained
    after a DEER experiment using pulse EPR spectrometer.
    The function find_peaks_cwt can be adjusted with two parameters, widths
    and min_snr.
    It has to be optimized for distance domaine data, check the documentation :
    on scipy.signal.find_peaks_cwt
    Parameters
    ----------
    DistanceDistribution : a two column numpy arrays, where the first column is
    the distance and the second column is the distance distribution P(r)
        DESCRIPTION.
    NumberOfGaussian : TYPE, optional
        DESCRIPTION. The default is 1.
    Returns
    -------
    fitted_data : TYPE
        DESCRIPTION.
    popt: TYPE
        DESCRIPTION.
    perr: TYPE
        DESCRIPTION.
    residual: TYPE
        DESCRIPTION.
    '''
    npts = x.shape[0]
    ymax = np.max(y)
    # Normalization of the data:
    y = np.ravel(y)/ymax
    NumberOfGaussian = int(len(params0)/3)

    def Gaussian(x, *params):
        yfit = np.zeros_like(x)
        for i in range(0, len(params), 3):
            ctr = params[i]
            wid = params[i+1]
            amp = params[i+2]
            yfit = yfit + amp*np.sqrt(1/(2*np.pi))*(1/wid)*np.exp(
                -0.5*((x - ctr)/wid)**2)
        return yfit
    # defining the bounds :
    if bounds == None:
        lb = np.full((3, NumberOfGaussian), np.nan, dtype=float)
        ub = np.full((3, NumberOfGaussian), np.nan, dtype=float)
        for i in range(NumberOfGaussian):
            lb[0:3, i] = [1, 0.01, 0]
            ub[0:3, i] = [8, 3, 10]
        lb = lb.ravel(order='F')
        ub = ub.ravel(order='F')
        bounds = (lb, ub)
    # Achieve the non-linear fit with the scipy.optimize.curve_fit function
    popt, pcov = curve_fit(Gaussian, x, y, p0=params0, sigma=None,
                           absolute_sigma=False, check_finite=None, bounds=bounds)
    perr = np.sqrt(np.diag(pcov))
    fitted_data = Gaussian(x, *popt)*ymax
    RMSE_Gauss = ComputeRMSE(y, Gaussian(x, *popt), popt)
    y = y*ymax
    residual = y-fitted_data
    return fitted_data, popt, perr, residual, RMSE_Gauss


def GaussianFit(x, y, NumberOfGaussian=1, height=1/10):
    '''
    Achieve the gaussian fit of data.
    The script was designed for distance domain experimental data obtained
    after a DEER experiment using pulse EPR spectrometer.
    The function find_peaks_cwt can be adjusted with two parameters, widths
    and min_snr.
    It has to be optimized for distance domaine data, check the documentation :
    on scipy.signal.find_peaks_cwt
    Parameters
    ----------
    DistanceDistribution : a two column numpy arrays, where the first column is
    the distance and the second column is the distance distribution P(r)
        DESCRIPTION.
    NumberOfGaussian : TYPE, optional
        DESCRIPTION. The default is 1.
    Returns
    -------
    fitted_data : TYPE
        DESCRIPTION.
    popt: TYPE
        DESCRIPTION.
    perr: TYPE
        DESCRIPTION.
    residual: TYPE
        DESCRIPTION.
    '''
    npts = x.shape[0]
    ymax = np.max(y)
    # Normalization of the data:
    y = np.ravel(y)/ymax
    # Initial guess of the gaussian position are probably to hard to do in an
    # automated way, as there is issue to detect the right peaks, but
    # we can try it to enhance the fitting or adjust to default values
    # mean = 2nm, std = 0.5 nm and amp = 1nm
    # i_pk = find_peaks_cwt(y, widths=10, min_snr=1)
    i_pk = find_peaks(y, height)
    fwhm = np.full((4, len(i_pk[0])), np.nan, dtype=float)
    fwhm[0:4, 0:len(i_pk[0])] = np.asarray(peak_widths(y, i_pk[0], 0.5))
    params0 = [None] * NumberOfGaussian*3
    if len(i_pk) >= NumberOfGaussian:
        for i in range(NumberOfGaussian):
            print(i_pk[0])
            params0[3*i] = x[i_pk[0][i]]
            params0[3*i+1] = (x[-1]-x[0])*(fwhm[3, i]-fwhm[2, i])/npts
            params0[3*i+2] = y[i_pk[0][i]]
            a = 'Gaussian fit initializes with the guess of peak position and'\
                ' fwhm evaluations with the functions scipy.signal.find_peaks'\
                ' and scipy.signal.peak_widths functions.'
    else:
        for i in range(NumberOfGaussian):
            # We attributes some default values
            params0[3*i] = float(2+2*i)
            params0[3*i+1] = float(0.5)
            params0[3*i+2] = float(1)
            a = 'Gaussian fit initializes with default parameters'
    # Gaussian function used for the fitting :
    print(a)

    def Gaussian(x, *params):
        yfit = np.zeros_like(x)
        for i in range(0, len(params), 3):
            ctr = params[i]
            wid = params[i+1]
            amp = params[i+2]
            yfit = yfit + amp*np.sqrt(1/(2*np.pi))*(1/wid)*np.exp(
                -0.5*((x - ctr)/wid)**2)
        return yfit
    # defining the bounds :
    lb = np.full((3, NumberOfGaussian), np.nan, dtype=float)
    ub = np.full((3, NumberOfGaussian), np.nan, dtype=float)
    for i in range(NumberOfGaussian):
        lb[0:3, i] = [1, 0.01, 0]
        ub[0:3, i] = [8, 3, 10]
    lb = lb.ravel(order='F')
    ub = ub.ravel(order='F')
    b = (lb, ub)
    # Achieve the non-linear fit with the scipy.optimize.curve_fit function
    popt, pcov = curve_fit(Gaussian, x, y, p0=params0, sigma=None,
                           absolute_sigma=False, check_finite=None, bounds=b)
    perr = np.sqrt(np.diag(pcov))
    fitted_data = Gaussian(x, *popt)*ymax
    RMSE_Gauss = ComputeRMSE(y, Gaussian(x, *popt), popt)
    y = y*ymax
    residual = y-fitted_data
    return fitted_data, popt, perr, residual, RMSE_Gauss


def DEERLABFitForOneFile(fullpath_to_filename, DataDict, ParamDict):
    '''
    Function to achieve Tikhonov regulrization
    Regularization parameters are chosen as 'aic' and 'gcv' as described in the
    publication : https://doi.org/10.1016/j.jmr.2018.01.021
    (Journal of Magnetic Resonance 288 (2018) 58–68)

    Returns
    -------
    Fit results

    '''
    fileID = path.split(fullpath_to_filename)
    # _, _, par = eprload(fullpath_to_filename, Scaling=None)
    # TITL = str(par['TITL'])
    TITL = fileID[1]
    Exp_parameter = ParamDict.get(TITL)
    itmin = Exp_parameter.get('itmin')
    # print(itmin)
    itmax = Exp_parameter.get('itmax')
    y = DataDict.get(TITL)[itmin:itmax, 5].ravel()
    # newy = y[itmin:itmax,]
    x = DataDict.get(TITL)[itmin:itmax, 0].ravel()
    # Experimental parameters (get the parameters from the Bruker DSC format)
    BrukerParamDict = GetExpParamValues(fullpath_to_filename)
    tau1 = BrukerParamDict['d1']/1000  # First inter-pulse delay, us
    tau2 = BrukerParamDict['d2']/1000  # Second inter-pulse delay, us
    t = (x.real/1000 + tau1).ravel()
    Vexp = np.real(y-np.max(y)+1).ravel()
    # r = dl.distancerange(t, nr=t.shape[0])
    npts = t.shape[0]
    r = dl.distancerange(t, nr=npts)
    # Construct the simulation fitting model of the distance distribution:
    Pmodel1 = dl.dd_gauss
    Pmodel1.mean.set(lb=2.5, ub=max(r), par0=3.0)
    Pmodel1.std.set(lb=0.01, ub=2, par0=0.1)

    Pmodel2 = dl.dd_gauss2
    Pmodel2.mean1.set(lb=min(r), ub=max(r), par0=2.0)
    Pmodel2.std1.set(lb=0.01, ub=2, par0=0.1)
    Pmodel2.mean2.set(lb=min(r), ub=max(r), par0=2.5)
    Pmodel2.std2.set(lb=0.01, ub=4, par0=0.1)

    # Construct the model (Tikhonov regularization)
    my4PDEER = dl.ex_4pdeer(tau1, tau2, pathways=[1])
    Vmodel_P = dl.dipolarmodel(t, r, Pmodel=None, Bmodel=None,
                               experiment=my4PDEER)
    Vmodel_Gauss1 = dl.dipolarmodel(t, r, Pmodel=Pmodel1, Bmodel=None,
                                    experiment=my4PDEER)
    # Vmodel_Gauss2 = dl.dipolarmodel(t, r, Pmodel=Pmodel2, Bmodel=None,
    #                                 experiment=my4PDEER)
    # Fit the model to the data
    results_P1 = dl.fit(Vmodel_P, Vexp, reg=True,
                        regparam=0.005)  # True 'aic'
    results_P2 = dl.fit(Vmodel_P, Vexp, reg=True,
                        regparam=0.005)  # True 'gcv'
    P1 = results_P1.P
    P2 = results_P2.P
    Vfit1 = results_P1.model
    Vfit2 = results_P2.model
    residuals1 = results_P1.regparam_stats['residuals']
    penalties1 = results_P1.regparam_stats['penalties']
    alphas1 = results_P1.regparam_stats['alphas_evaled'][1:]
    funcs1 = results_P1.regparam_stats['functional'][1:]

    # To plot alpha selection functional, do the following command :
    alphas1 = results_P1.regparam_stats['alphas_evaled'][1:]
    funcs1 = results_P1.regparam_stats['functional'][1:]
    # idx = np.argsort(alphas1)
    # plt.semilogx(alphas1[idx], funcs1[idx], '-+')
    # plt.ylabel("Regularisation Parameter")
    # plt.xlabel("Functional Value ")
    # plt.title(r"$\alpha$ selection functional")

    # To plot L-curve, do the following commands :
    # idx = np.argsort(residuals)
    # plt.loglog(residuals[idx],penalties[idx], '-+')
    # plt.ylabel("Penalties")
    # plt.xlabel("Residuals")
    # plt.title("L-Curve")
    residuals2 = results_P2.regparam_stats['residuals']
    penalties2 = results_P2.regparam_stats['penalties']
    alphas2 = results_P2.regparam_stats['alphas_evaled'][1:]
    funcs2 = results_P2.regparam_stats['functional'][1:]

    results1_Gauss1 = dl.fit(Pmodel1, P1, r)
    Pfit1_Gauss1 = results1_Gauss1.evaluate(Pmodel1, r)
    results1_Gauss2 = dl.fit(Pmodel2, P1, r)
    Pfit1_Gauss2 = results1_Gauss2.evaluate(Pmodel2, r)
    results2_Gauss1 = dl.fit(Pmodel1, P2, r)
    Pfit2_Gauss1 = results2_Gauss1.evaluate(Pmodel1, r)
    results2_Gauss2 = dl.fit(Pmodel2, P2, r)
    Pfit2_Gauss2 = results2_Gauss2.evaluate(Pmodel2, r)
    #
    scale1 = np.trapz(P1, r)*100
    P1 = P1/scale1
    Puq1 = results_P1.PUncert
    Pci95_1 = Puq1.ci(95)/scale1

    scale2 = np.trapz(P2, r)*100
    P2 = P2/scale2
    Puq2 = results_P2.PUncert
    Pci95_2 = Puq2.ci(95)/scale2

    P1_Gauss1 = Pfit1_Gauss1/scale1
    Puq1_Gauss1 = results1_Gauss1.propagate(Pmodel1, r, lb=np.zeros_like(r))
    Pci95_1_Gauss1 = Puq1_Gauss1.ci(95)/scale1

    P1_Gauss2 = Pfit1_Gauss2/scale1
    Puq1_Gauss2 = results1_Gauss2.propagate(Pmodel2, r, lb=np.zeros_like(r))
    Pci95_1_Gauss2 = Puq1_Gauss2.ci(95)/scale1

    P2_Gauss1 = Pfit2_Gauss1/scale2
    Puq2_Gauss1 = results2_Gauss1.propagate(Pmodel1, r, lb=np.zeros_like(r))
    Pci95_2_Gauss1 = Puq2_Gauss1.ci(95)/scale2

    P2_Gauss2 = Pfit2_Gauss2/scale2
    Puq2_Gauss2 = results2_Gauss2.propagate(Pmodel2, r, lb=np.zeros_like(r))
    Pci95_2_Gauss2 = Puq2_Gauss2.ci(95)/scale2

    # Save all the data in DataDict and ParamDict
    fulldata = DataDict.get(TITL)
    # 'AIC' Criterium first'
    fulldata[0:npts, 9] = r.ravel()
    fulldata[0:npts, 10] = P1.ravel()
    fulldata[0:npts, 11] = Pci95_1[:, 0].ravel()
    fulldata[0:npts, 12] = Pci95_1[:, 1].ravel()
    fulldata[0:npts, 13] = P1_Gauss1.ravel()
    fulldata[0:npts, 14] = Pci95_1_Gauss1[:, 0].ravel()
    fulldata[0:npts, 15] = Pci95_1_Gauss1[:, 1].ravel()
    fulldata[0:npts, 16] = P1_Gauss2.ravel()
    fulldata[0:npts, 17] = Pci95_1_Gauss2[:, 0].ravel()
    fulldata[0:npts, 18] = Pci95_1_Gauss2[:, 1].ravel()
    # print(alphas1)
    npts2 = len(alphas1)
    fulldata[0:npts2, 19] = alphas1[:]  # .ravel()
    fulldata[0:npts2, 20] = funcs1[:]  # .ravel()
    npts3 = len(residuals1)
    fulldata[0:npts3, 21] = residuals1[:]  # .ravel()
    fulldata[0:npts3, 22] = penalties1[:]  # .ravel()
    # 'GCV' Criterium second'
    fulldata[0:npts, 23] = r.ravel()
    fulldata[0:npts, 24] = P2.ravel()
    fulldata[0:npts, 25] = Pci95_2[:, 0].ravel()
    fulldata[0:npts, 26] = Pci95_2[:, 1].ravel()
    fulldata[0:npts, 27] = P2_Gauss1.ravel()
    fulldata[0:npts, 28] = Pci95_2_Gauss1[:, 0].ravel()
    fulldata[0:npts, 29] = Pci95_2_Gauss1[:, 1].ravel()
    fulldata[0:npts, 30] = P2_Gauss2.ravel()
    fulldata[0:npts, 31] = Pci95_2_Gauss2[:, 0].ravel()
    fulldata[0:npts, 32] = Pci95_2_Gauss2[:, 1].ravel()
    npts4 = len(alphas2)
    fulldata[0:npts4, 33] = alphas2[:]  # .ravel()
    fulldata[0:npts4, 34] = funcs2[:]  # .ravel()
    npts5 = len(residuals2)
    fulldata[0:npts5, 35] = residuals2[:]  # .ravel()
    fulldata[0:npts5, 36] = penalties2[:]  # .ravel()
    fulldata[0:npts, 37] = Vfit1[:,].ravel()
    fulldata[0:npts, 38] = Vfit2[:,].ravel()
    DataDict.update({TITL: fulldata})

    HeaderTITL = str(TITL+'_Header')
    Header = DataDict.get(HeaderTITL)
    Header[9] = str(TITL+"_Distance_(nm)")
    Header[10] = str(TITL+"_Distance_Domain_AIC")
    Header[11] = str(TITL+"_Distance_Domain_AIC_PCI95_0")
    Header[12] = str(TITL+"_Distance_Domain_AIC_PCI95_1")
    Header[13] = str(TITL+"_Simulated_Distance_Domain_OneGaussian_AIC")
    Header[14] = str(TITL+"_Simulated_Distance_Domain_OneGaussian_AIC_PCI95_0")
    Header[15] = str(TITL+"_Simulated_Distance_Domain_OneGaussian_AIC_PCI95_1")
    Header[16] = str(TITL+"_Simulated_Distance_Domain_TwoGaussian_AIC")
    Header[17] = str(TITL+"_Simulated_Distance_Domain_TwoGaussian_AIC_PCI95_0")
    Header[18] = str(TITL+"_Simulated_Distance_Domain_TwoGaussian_AIC_PCI95_1")
    Header[19] = str(TITL+"_L_Curve_AIC_alphas")
    Header[20] = str(TITL+"_L_Curve_AIC_funcs")
    Header[21] = str(TITL+"_L_Curve_AIC_residuals")
    Header[22] = str(TITL+"_L_Curve_AIC_penalties")
    Header[23] = str(TITL+"_Distance_(nm)")
    Header[24] = str(TITL+"_Distance_Domain_GCV")
    Header[25] = str(TITL+"_Distance_Domain_GCV_PCI95_0")
    Header[26] = str(TITL+"_Distance_Domain_GCV_PCI95_1")
    Header[27] = str(TITL+"_Simulated_Distance_Domain_OneGaussian_GCV")
    Header[28] = str(TITL+"_Simulated_Distance_Domain_OneGaussian_GCV_PCI95_0")
    Header[29] = str(TITL+"_Simulated_Distance_Domain_OneGaussian_GCV_PCI95_1")
    Header[30] = str(TITL+"_Simulated_Distance_Domain_TwoGaussian_GCV")
    Header[31] = str(TITL+"_Simulated_Distance_Domain_TwoGaussian_GCV_PCI95_0")
    Header[32] = str(TITL+"_Simulated_Distance_Domain_TwoGaussian_GCV_PCI95_1")
    Header[33] = str(TITL+"_L_Curve_GCV_alphas")
    Header[34] = str(TITL+"_L_Curve_GCV_funcs")
    Header[35] = str(TITL+"_L_Curve_GCV_residuals")
    Header[36] = str(TITL+"_L_Curve_GCV_penalties")
    Header[37] = str(TITL+"_TimeSignalTikhonovAIC")
    Header[38] = str(TITL+"_TimeSignalTikhonovGCV")

    GaussFitParam_aic = [results1_Gauss1.mean, results1_Gauss1.std,
                         results1_Gauss2.mean1, results1_Gauss2.std1,
                         results1_Gauss2.mean2, results1_Gauss2.std2,
                         results1_Gauss2.amp1, results1_Gauss2.amp2]
    GaussFitParam_gcv = [results2_Gauss1.mean, results2_Gauss1.std,
                         results2_Gauss2.mean1, results2_Gauss2.std1,
                         results2_Gauss2.mean2, results2_Gauss2.std2,
                         results2_Gauss2.amp1, results2_Gauss2.amp2]
    HeaderGaussFit = ['mean_OneGaussian', 'std_OneGaussian',
                      'mean1_TwoGaussian', 'std1_TwoGaussian',
                      'mean2_TwoGaussian', 'std2_TwoGaussian',
                      'amp1_TwoGaussian', 'amp2_TwoGaussian']
    Exp_parameter.update({'alpha_opt_aic': results_P1.regparam,
                          'alpha_opt_gcv': results_P2.regparam,
                          'fitting parameters_aic': GaussFitParam_aic,
                          'fitting parameters_gcv': GaussFitParam_gcv,
                          'HeaderGaussianFit': HeaderGaussFit})
    ParamDict.update({TITL: Exp_parameter})
    return DataDict, ParamDict


def DEERLABFitForMultipleFIles(ListOfFiles, DataDict, ParamDict):
    '''
    Function to achieve Tikhonov regulrization

    Returns
    -------
    None.

    '''
    NumberOfFiles = len(ListOfFiles)
    for i in range(NumberOfFiles):
        DataDict, ParamDict = BckgndSubtractionOfRealPart2(ListOfFiles[i],
                                                           DataDict, ParamDict, Dimensionorder=1, Percent_tmax=40/40,
                                                           mode='polyexp', truncsize=3/4)
        DataDict, ParamDict = GetPakePatternForOneFile(
            ListOfFiles[i], DataDict, ParamDict)
        DataDict, ParamDict = DEERLABFitForOneFile(ListOfFiles[i],
                                                   DataDict, ParamDict)
    return DataDict, ParamDict


def GetSVDFitForOneFile(fullpath_to_filename, DataDict, ParamDict):
    '''
    Function to achieve Singular Value Decomposition

    Returns
    -------
    None.

    '''
    fileID = path.split(fullpath_to_filename)
    # _, _, par = eprload(fullpath_to_filename, Scaling=None)
    # TITL = str(par['TITL'])
    TITL = fileID[1]
    y = DataDict.get(TITL)[:, 5].ravel()
    y2 = y[~np.isnan(y)].real
    x = DataDict.get(TITL)[:, 0].ravel()
    x2 = x[~np.isnan(x)].real
    # npts_init = y.shape[0]
    Exp_parameter = ParamDict.get(TITL)
    itmin = Exp_parameter.get('itmin')
    itmax = Exp_parameter.get('itmax')
    newy = y2[itmin:itmax,]
    newx = x2[itmin:itmax,]
    K, U, s, V = get_KUsV(newx, newy)
    S, sigma, PR, Pr, Picard, sum_Pic = process_data(newx, newy, K, U, s, V)
    npts = newx.shape[0]
    r = dl.distancerange(newx, nr=npts)
    # reconstruct_dipolar_signal(K, PR):
    Pfit = np.dot(K, PR.T)
    # Save all the data in DataDict and ParamDict
    fulldata = DataDict.get(TITL)
    fulldata[0:npts, 6] = r.ravel()
    fulldata[0:npts, 7] = Pfit.ravel()
    fulldata[0:npts, 8] = sum_Pic.ravel()
    fulldata[0:npts, 9] = sigma.ravel()

    # Fit the distance domain with one or two gaussians :
    fitted_data1, popt1, perr1, residual1, RMSE_Gauss1 = GaussianFit(
        r, Pfit, NumberOfGaussian=1, height=1/10)
    fitted_data2, popt2, perr2, residual2, RMSE_Gauss2 = GaussianFit(
        r, Pfit, NumberOfGaussian=2, height=1/10)
    fulldata[0:npts, 10] = fitted_data1.ravel()
    fulldata[0:npts, 11] = fitted_data2.ravel()
    DataDict.update({TITL: fulldata})
    # fulldata[0:npts, 8] = Pci95_1[0].ravel()
    # fulldata[0:npts, 10] = P1_Gauss1.ravel()
    # fulldata[0:npts, 11] = Pci95_1_Gauss1[0].ravel()
    # fulldata[0:npts, 12] = Pci95_1_Gauss1[1].ravel()
    # fulldata[0:npts, 13] = P1_Gauss2.ravel()
    # fulldata[0:npts, 14] = Pci95_1_Gauss2[0].ravel()
    # fulldata[0:npts, 15] = Pci95_1_Gauss2[1].ravel()
    # npts2 = alphas1.shape[0]
    # fulldata[0:npts2, 16] = alphas1[0].ravel()
    # fulldata[0:npts2, 17] = funcs1[0].ravel()
    # npts3 = residuals1.shape[0]
    # fulldata[0:npts3, 18] = residuals1[0].ravel()
    # fulldata[0:npts3, 19] = penalties1[0].ravel()
    # # 'AIC' Criterium first'
    # fulldata[0:npts, 20] = r.ravel()
    # fulldata[0:npts, 21] = P2.ravel()
    # fulldata[0:npts, 22] = Pci95_2[0].ravel()
    # fulldata[0:npts, 23] = Pci95_2[1].ravel()
    # fulldata[0:npts, 24] = P2_Gauss1.ravel()
    # fulldata[0:npts, 25] = Pci95_2_Gauss1[0].ravel()
    # fulldata[0:npts, 26] = Pci95_2_Gauss1[1].ravel()
    # fulldata[0:npts, 27] = P2_Gauss2.ravel()
    # fulldata[0:npts, 28] = Pci95_2_Gauss2[0].ravel()
    # fulldata[0:npts, 29] = Pci95_2_Gauss2[1].ravel()
    # npts4 = alphas2.shape[0]
    # fulldata[0:npts4, 30] = alphas2[0].ravel()
    # fulldata[0:npts4, 31] = funcs2[0].ravel()
    # npts5 = residuals2.shape[0]
    # fulldata[0:npts5, 32] = residuals2[0].ravel()
    # fulldata[0:npts5, 33] = penalties2[0].ravel()

    HeaderTITL = str(TITL+'_Header')
    Header = DataDict.get(HeaderTITL)
    Header[6] = str(TITL+"_Distance_(nm)")
    Header[7] = str(TITL+"_DistanceDomain_SVD")
    Header[8] = str(TITL+"_Picard_Sum")
    Header[9] = str(TITL+"_sigma")
    Header[10] = str(TITL+"_DistanceDomain_OneGaussianFit")
    Header[11] = str(TITL+"_DistanceDomain_TwoGaussianFit")

    DataDict.update({HeaderTITL: Header})
    results_Gauss = [popt1[0], popt1[1], popt2[0], popt2[1], popt2[3],
                     popt2[4], popt2[2], popt2[5]]
    HeaderGaussFit = ['mean_OneGaussian', 'std_OneGaussian',
                      'mean1_TwoGaussian', 'std1_TwoGaussian',
                      'mean2_TwoGaussian', 'std2_TwoGaussian',
                      'amp1_TwoGaussian', 'amp2_TwoGaussian']

    Exp_parameter.update({'GaussFit_SVD': results_Gauss,
                          # 'alpha_opt_gcv': results_P2.regparam,
                          # 'fitting parameters_aic': GaussFitParam_aic,
                          # 'fitting parameters_gcv': GaussFitParam_gcv,
                          'HeaderGaussianFit': HeaderGaussFit})
    ParamDict.update({TITL: Exp_parameter})
    return DataDict, ParamDict


def GetSVDFitForMultipleFiles(ListOfFiles, DataDict, ParamDict):
    NumberOfFiles = len(ListOfFiles)
    for i in range(NumberOfFiles):
        DataDict, ParamDict = GetSVDFitForOneFile(ListOfFiles[i], DataDict,
                                                  ParamDict)
    return DataDict, ParamDict


def GetPakePattern(t, y):
    tmax = np.max(t)
    npts = y.shape[0]
    newt = np.linspace(0, tmax*5, 5*npts)/1000  # Time axis in us
    freq = fdaxis(TimeAxis=newt)
    win = windowing(window_type='exp+', N=npts, alpha=3)
    y2 = np.zeros((npts*5,), dtype="complex_")
    y2[0:npts,] = y[0:npts,]*win[0:npts,]
    PakeSpectra = np.fft.fftshift(np.fft.fft(y2))
    Pivot = int(np.floor(PakeSpectra.shape[0]/2))
    PakeSpectraPhased, _ = automatic_phase(vector=PakeSpectra, pivot1=Pivot,
                                           funcmodel='minfunc')
    PakeAbs, _, _, _ = basecorr1D(x=freq, y=np.absolute(PakeSpectraPhased),
                                  polyorder=1, window=200)
    return PakeSpectraPhased, PakeAbs, freq


def GetPakePatternForOneFile(fullpath_to_filename, DataDict, ParamDict):
    fileID = path.split(fullpath_to_filename)
    # _, abscissa, par = eprload(fullpath_to_filename, Scaling=None)
    # npts_init = abscissa.shape[0]
    # TITL = str(par['TITL'])
    TITL = fileID[1]
    Exp_parameter = ParamDict.get(TITL)
    tmax = Exp_parameter.get('tmax')
    itmin = Exp_parameter.get('itmin')
    itmax = Exp_parameter.get('itmax')
    y = DataDict.get(TITL)[itmin:itmax, 5].ravel()
    y = y[~np.isnan(y)]
    # t = DataDict.get(TITL)[:, 0].ravel()
    # t2 = t[~np.isnan(t)].real
    npts = y.shape[0]
    newt = np.linspace(0, tmax*5, 5*npts)/1000  # Time axis in us
    freq = fdaxis(TimeAxis=newt)  # Frequency axis in MHz
    win = windowing(window_type='exp+', N=npts, alpha=3)
    y2 = np.zeros((npts*5,), dtype="complex_")  # zerofilling
    y2[0:npts,] = y[0:npts,]*win[0:npts,]
    PakeSpectra = np.fft.fftshift(np.fft.fft(y2))
    Pivot = int(np.floor(PakeSpectra.shape[0]/2))
    PakeSpectraPhased, _ = automatic_phase(vector=PakeSpectra, pivot1=Pivot,
                                           funcmodel='minfunc')
    PakeAbs, _, _, _ = basecorr1D(x=freq, y=np.abs(PakeSpectraPhased),
                                  polyorder=1, window=200)
    # Assign the data
    fulldata = DataDict.get(TITL)
    fulldata[0:5*npts, 6] = freq.ravel()
    fulldata[0:5*npts, 7] = PakeSpectra.ravel()
    fulldata[0:5*npts, 8] = PakeAbs.ravel()
    DataDict.update({TITL: fulldata})
    # Assign the Header of the data
    HeaderTITL = str(TITL+'_Header')
    Header = DataDict.get(HeaderTITL)
    Header[6] = str(TITL+"_Frequency_(MHz)")
    Header[7] = str(TITL+"_Complex_Pake_Pattern")
    Header[8] = str(TITL+"_Absolute_Pake_Pattern")
    DataDict.update({HeaderTITL: Header})
    return DataDict, ParamDict


def GetPakePatternForMultipleFiles(ListOfFiles, DataDict, ParamDict):
    NumberOfFiles = len(ListOfFiles)
    for i in range(NumberOfFiles):
        DataDict, ParamDict = GetSVDFitForOneFile(ListOfFiles[i], DataDict,
                                                  ParamDict)
    return DataDict, ParamDict


def DenoisingForOneFile(fullpath_to_filename, DataDict, ParamDict,
                        Dimensionorder=3, Percent_tmax=9/10, mode='strexp',
                        truncsize=3/4):
    fileID = path.split(fullpath_to_filename)
    _, _, par = eprload(fullpath_to_filename, Scaling=None)
    # TITL = str(par['TITL'])
    TITL = fileID[1]
    Exp_parameter = ParamDict.get(TITL)
    tmax = Exp_parameter.get('tmax')
    itmin = Exp_parameter.get('itmin')
    itmax = Exp_parameter.get('itmax')
    noiselevel = Exp_parameter.get('NoiseLevel')
    maxlevel = ThresholdGuess(noiselevel[1])
    y_phased = DataDict.get(TITL)[:itmax, 3].ravel()  # itmin
    bckg = DataDict.get(TITL)[:itmax, 4].ravel()  # itmin
    y = DataDict.get(TITL)[:itmax, 5].ravel()  # itmin
    newx = DataDict.get(TITL)[:itmax, 0].ravel()  # itmin
    newy = np.real(y)
    npts = newy.shape[0]  # [itmin:itmax,])
    maxvalue = np.max(y_phased)
    wavename = 'db6'
    denoisedmode = 'antisymmetric'  # 'antisymmetric'
    dsignaldict, coeff, thresholds = Deer_Denoising(newx, newy,
                                                    wavename=wavename,
                                                    mode=denoisedmode)

    # catch the denoised signal at the 'right' level of decomposition
    dsignal = np.flipud(dsignaldict.get(str(maxlevel)))  # -1
    # reconstruct the signal without background subtraction
    dsignal = dsignal[0:npts,]*maxvalue + bckg[0:npts,]
    # newdata_real = datasmooth(
    #     dsignal[0:npts], window_length=1, method='binom')
    # itmin = newdata_real.argmax()
    # tmin = newx[itmin]
    # newx = newx-tmin
    # print(newx[0])
    # reprocess the background subtraction as if the denoised was the original
    # one:
    data_bckgnd, p0, perr = ExponentialCorr1D_DEER(x=newx, y=dsignal.real,
                                                   Dimensionorder=Dimensionorder,
                                                   Percent_tmax=Percent_tmax, mode=mode,
                                                   truncsize=truncsize)
    RMSE = ComputeRMSE(dsignal.real/np.max(dsignal.real),
                       data_bckgnd/np.max(dsignal.real), p0)
    newnpts = data_bckgnd.shape[0]
    itmax = newnpts-1
    tmax = newx[itmax]
    FinalData = (dsignal.real - data_bckgnd)/np.max(dsignal.real)
    FinalData = FinalData-FinalData[-20:].mean()
    # newdata_real = datasmooth(
    #     BckgndSubtractedData[0:npts], window_length=1, method='binom')
    # itmin = newdata_real.argmax()
    # tmin = newx[itmin]
    # newx = newx-tmin
    RelBckgndDecay = 1 - data_bckgnd[itmax] / data_bckgnd[itmin]
    # replace the data dict for the denoised one
    # fulldata = np.full((5*newnpts, 37), np.nan, dtype="complex_")
    fulldata = DataDict.get(TITL)
    fullnpts = fulldata.shape[0]
    fulldata[:, 0:6] = np.full((fullnpts, 6), np.nan, dtype="complex_")
    fulldata[0:newnpts, 0] = newx.ravel()
    fulldata[0:newnpts, 1] = y.ravel()
    fulldata[0:newnpts, 2] = dsignal.ravel()
    fulldata[0:newnpts, 3] = dsignal.real.ravel()
    fulldata[0:newnpts, 4] = data_bckgnd.ravel()
    fulldata[0:newnpts, 5] = FinalData.ravel()
    DataDict.update({TITL: fulldata})
    HeaderTITL = str(TITL+'_Header')
    Header = DataDict.get(HeaderTITL)
    Header[:6] = list(np.zeros((6,)))
    Header[0] = str(par['XNAM'])
    Header[1] = str(TITL+"_OriginalData")
    Header[2] = str(TITL+"_denoised")
    Header[3] = str(TITL+"_denoised_real")
    Header[4] = str(TITL+"_background_real")
    Header[5] = str(TITL+"_backgroundsubtracted_real")
    DataDict.update({HeaderTITL: Header})

    Exp_parameter.update({'RelBckgndDecay_denoised': RelBckgndDecay,  # 'tmin': tmin,
                          'tmax': tmax, 'itmax': itmax,  # 'itmin': itmin,
                          'wavename': wavename, 'denoisedmode': denoisedmode})
    # Assign the modulation depth in the parameters
    sigma_noise, sigma_noise_half = noiselevel[0], noiselevel[1]
    if mode == 'strexp':
        DEER_SNR = (p0[0]/sigma_noise, p0[0]/sigma_noise_half)
        Bckg_type = str(mode+'_'+str(Percent_tmax)+'_'+str(truncsize))
        Exp_parameter.update({'ModDepth_denoised': p0[0], 'decay': p0[1],
                             'stretch': p0[2], 'Bckg_type': Bckg_type,
                              'RMSE': RMSE, 'DEER_SNR_denoised': DEER_SNR})
    if mode == 'strexp_fixed':
        DEER_SNR = (p0[0]/sigma_noise, p0[0]/sigma_noise_half)
        Bckg_type = str(mode+'_'+str(Percent_tmax)+'_'+str(truncsize))
        Exp_parameter.update({'ModDepth_denoised': p0[0], 'decay': p0[1],
                              'stretch': p0[2], 'Bckg_type': Bckg_type,
                              'RMSE': RMSE, 'DEER_SNR_denoised': DEER_SNR})
    elif mode == 'poly':
        Bckg_type = str(mode+str(len(p0)-1)+'_'+str(Percent_tmax)+'_' +
                        str(truncsize))
        Mod_Depth = (dsignal.real[0:1].mean() - data_bckgnd[0:1].mean()) / (
            dsignal.real[0:1].mean())
        DEER_SNR = (Mod_Depth/sigma_noise, Mod_Depth/sigma_noise_half)
        Exp_parameter.update({'ModDepth_denoised': Mod_Depth, 'polyparam': p0,
                              'Bckg_type': Bckg_type, 'RMSE': RMSE,
                              'DEER_SNR_denoised': DEER_SNR})
    elif mode == 'polyexp':
        Bckg_type = str(mode+str(len(p0)-1)+'_'+str(Percent_tmax)+'_' +
                        str(truncsize))
        Mod_Depth = (dsignal.real[0:1].mean() - data_bckgnd[0:1].mean()) / (
            dsignal.real[0:1].mean())
        DEER_SNR = (Mod_Depth/sigma_noise, Mod_Depth/sigma_noise_half)
        Exp_parameter.update({'ModDepth_denoised': Mod_Depth, 'polyparam': p0,
                              'Bckg_type': Bckg_type, 'RMSE': RMSE,
                              'DEER_SNR_denoised': DEER_SNR})

        Exp_parameter.update({})
    ParamDict.update({TITL: Exp_parameter})
    return DataDict, ParamDict


def DenoisingForOneFile2(fullpath_to_filename, DataDict, ParamDict,
                         Dimensionorder=3, Percent_tmax=9/10, mode='strexp',
                         truncsize=3/4, zerofilling=0):
    fileID = path.split(fullpath_to_filename)
    _, _, par = eprload(fullpath_to_filename, Scaling=None)
    # TITL = str(par['TITL'])
    TITL = fileID[1]
    Exp_parameter = ParamDict.get(TITL)
    tmax = Exp_parameter.get('tmax')
    itmin = Exp_parameter.get('itmin')
    itmax = Exp_parameter.get('itmax')
    noiselevel = Exp_parameter.get('NoiseLevel')
    maxlevel = ThresholdGuess(noiselevel[1])
    y_phased = DataDict.get(TITL)[:itmax, 3].ravel()  # itmin
    bckg = DataDict.get(TITL)[:itmax, 4].ravel()  # itmin
    y = DataDict.get(TITL)[:itmax, 5].ravel()  # itmin
    newx = DataDict.get(TITL)[:itmax, 0].ravel()  # itmin
    newy = np.real(y)
    npts = newy.shape[0]  # [itmin:itmax,])
    maxvalue = np.max(y_phased)
    wavename = 'db6'
    denoisedmode = 'symmetric'  # 'antisymmetric'
    dsignaldict, coeff, thresholds = Deer_Denoising(newx, newy,
                                                    wavename=wavename,
                                                    mode=denoisedmode)

    # catch the denoised signal at the 'right' level of decomposition
    dsignal = np.flipud(dsignaldict.get(str(maxlevel)))  # -1
    # reconstruct the signal without background subtraction
    dsignal = dsignal[0:npts,]*maxvalue + bckg[0:npts,]
    # newdata_real = datasmooth(
    #     dsignal[0:npts], window_length=1, method='binom')
    # itmin = newdata_real.argmax()
    # tmin = newx[itmin]
    # newx = newx-tmin
    # print(newx[0])
    # reprocess the background subtraction as if the denoised was the original
    # one:
    data_bckgnd, p0, perr = ExponentialCorr1D_DEER(x=newx, y=dsignal.real,
                                                   Dimensionorder=Dimensionorder,
                                                   Percent_tmax=1, mode=mode,
                                                   truncsize=truncsize)
    RMSE = ComputeRMSE(dsignal.real/np.max(dsignal.real),
                       data_bckgnd/np.max(dsignal.real), p0)
    newnpts = data_bckgnd.shape[0]
    itmax = newnpts-1
    tmax = newx[itmax]
    FinalData = (dsignal.real - data_bckgnd)/np.max(dsignal.real)
    FinalData = FinalData-FinalData[-20:].mean()
    # newdata_real = datasmooth(
    #     BckgndSubtractedData[0:npts], window_length=1, method='binom')
    # itmin = newdata_real.argmax()
    # tmin = newx[itmin]
    # newx = newx-tmin
    RelBckgndDecay = 1 - data_bckgnd[itmax] / data_bckgnd[itmin]
    # replace the data dict for the denoised one
    # TITL = str(par['TITL'])
    TITL = fileID[1]
    if zerofilling == 0:
        fulldata = DataDict.get(TITL)
        fullnpts = fulldata.shape[0]
        fulldata[:, 0:6] = np.full((fullnpts, 6), np.nan, dtype="complex_")
        fulldata[0:newnpts, 0] = newx.ravel()
        fulldata[0:newnpts, 1] = y.ravel()
        fulldata[0:newnpts, 2] = dsignal.ravel()
        fulldata[0:newnpts, 3] = dsignal.real.ravel()
        fulldata[0:newnpts, 4] = data_bckgnd.ravel()
        fulldata[0:newnpts, 5] = FinalData.ravel()
        DataDict.update({TITL: fulldata})
        HeaderTITL = str(TITL+'_Header')
        Header = DataDict.get(HeaderTITL)
        Header[:6] = list(np.zeros((6,)))
        Header[0] = str(par['XNAM'])
        Header[1] = str(TITL+"_OriginalData")
        Header[2] = str(TITL+"_denoised")
        Header[3] = str(TITL+"_denoised_real")
        Header[4] = str(TITL+"_background_real")
        Header[5] = str(TITL+"_backgroundsubtracted_real")
        DataDict.update({HeaderTITL: Header})

    elif float(zerofilling) > 0:
        fulldata = DataDict.get(TITL)
        fullnpts = fulldata.shape[0]
        fulldata[:, 0:6] = np.full((fullnpts, 6), np.nan, dtype="complex_")
        totnpts = int(np.floor(zerofilling))+itmax
        dx = newx[1]-newx[0]
        tmax = newx[0]+dx*totnpts
        newx2 = np.linspace(newx[0], tmax, totnpts)
        FinalData2 = np.zeros((totnpts,), dtype="complex_")
        FinalData2[0:itmax,] = FinalData[0:itmax,]
        itmax = int(np.floor(zerofilling))+itmax
        fulldata = np.full((5*totnpts, 50), np.nan, dtype="complex_")
        fulldata[0:totnpts, 0] = newx2.ravel()
        fulldata[0:npts, 1] = y.ravel()
        fulldata[0:npts, 2] = dsignal.ravel()
        fulldata[0:npts, 3] = dsignal.real.ravel()
        fulldata[0:npts, 4] = data_bckgnd.ravel()
        fulldata[0:totnpts, 5] = FinalData2.ravel()
        DataDict.update({TITL: fulldata})
        HeaderTITL = str(TITL+'_Header')
        Header = DataDict.get(HeaderTITL)
        Header[:6] = list(np.zeros((6,)))
        Header[0] = str(par['XNAM'])
        Header[1] = str(TITL+"_OriginalData")
        Header[2] = str(TITL+"_denoised")
        Header[3] = str(TITL+"_denoised_real")
        Header[4] = str(TITL+"_background_real")
        Header[5] = str(TITL+"_backgroundsubtracted_real")
        DataDict.update({HeaderTITL: Header})

    Exp_parameter.update({'RelBckgndDecay_denoised': RelBckgndDecay,  # 'tmin': tmin,
                          'tmax': tmax, 'itmax': itmax,  # 'itmin': itmin,
                          'wavename': wavename, 'denoisedmode': denoisedmode})
    # Assign the modulation depth in the parameters
    sigma_noise, sigma_noise_half = noiselevel[0], noiselevel[1]
    if mode == 'strexp':
        DEER_SNR = (p0[0]/sigma_noise, p0[0]/sigma_noise_half)
        Bckg_type = str(mode+'_'+str(Percent_tmax)+'_'+str(truncsize))
        Exp_parameter.update({'ModDepth_denoised': p0[0], 'decay': p0[1],
                             'stretch': p0[2], 'Bckg_type': Bckg_type,
                              'RMSE': RMSE, 'DEER_SNR_denoised': DEER_SNR})
    if mode == 'strexp_fixed':
        DEER_SNR = (p0[0]/sigma_noise, p0[0]/sigma_noise_half)
        Bckg_type = str(mode+'_'+str(Percent_tmax)+'_'+str(truncsize))
        Exp_parameter.update({'ModDepth_denoised': p0[0], 'decay': p0[1],
                              'stretch': p0[2], 'Bckg_type': Bckg_type,
                              'RMSE': RMSE, 'DEER_SNR_denoised': DEER_SNR})
    elif mode == 'poly':
        Bckg_type = str(mode+str(len(p0)-1)+'_'+str(Percent_tmax)+'_' +
                        str(truncsize))
        Mod_Depth = (dsignal.real[0:1].mean() - data_bckgnd[0:1].mean()) / (
            dsignal.real[0:1].mean())
        DEER_SNR = (Mod_Depth/sigma_noise, Mod_Depth/sigma_noise_half)
        Exp_parameter.update({'ModDepth_denoised': Mod_Depth, 'polyparam': p0,
                              'Bckg_type': Bckg_type, 'RMSE': RMSE,
                              'DEER_SNR_denoised': DEER_SNR})
    elif mode == 'polyexp':
        Bckg_type = str(mode+str(len(p0)-1)+'_'+str(Percent_tmax)+'_' +
                        str(truncsize))
        Mod_Depth = (dsignal.real[0:1].mean() - data_bckgnd[0:1].mean()) / (
            dsignal.real[0:1].mean())
        DEER_SNR = (Mod_Depth/sigma_noise, Mod_Depth/sigma_noise_half)
        Exp_parameter.update({'ModDepth_denoised': Mod_Depth, 'polyparam': p0,
                              'Bckg_type': Bckg_type, 'RMSE': RMSE,
                              'DEER_SNR_denoised': DEER_SNR})

        Exp_parameter.update({})
    ParamDict.update({TITL: Exp_parameter})
    return DataDict, ParamDict


def DenoisingForMultipleFiles(ListOfFiles, DataDict, ParamDict):
    NumberOfFiles = len(ListOfFiles)
    for i in range(NumberOfFiles):
        DataDict, ParamDict = DenoisingForOneFile(ListOfFiles[i], DataDict,
                                                  ParamDict)
    return DataDict, ParamDict


def ExportParamTable(FolderPath=getcwd(), ParamDict=None):
    #     '''
    #     The script export a table of the important parameters deduced from the
    #     export files created by Deer-Analysis. (Typically the files are named :
    #                                      "Filename_res.txt")
    #     In the following order, it To redescribe

    #     Returns
    #     -------
    #     None.
    # To do, check if a version with https://xlsxwriter.readthedocs.io/
    # would be better
    #     '''
    ListOfFiles = list(ParamDict.keys())
    Paramlist = ['tmax', 'tmin', 'NoiseLevel', 'ModDepth', 'DEER_SNR',
                 'RelBckgndDecay', 'RMSE', 'polyparam']
    PolyOrder = [0]*len(ListOfFiles)
    for i in range(len(ListOfFiles)):
        PolyOrder[i] = len(ParamDict[ListOfFiles[i]]['polyparam'])
    maxn = max(PolyOrder)
    shape = (int(len(Paramlist)+maxn), len(ListOfFiles))
    Param = np.full(shape, np.nan, dtype=float)
    for i in range(len(ListOfFiles)):
        Value = []
        for j in range(len(Paramlist)):
            Value = ParamDict[ListOfFiles[i]][Paramlist[j]]
            if type(Value) == tuple:
                Param[j, i] = Value[1]
            elif type(Value) == np.ndarray:
                for k in range(len(Value)):
                    Param[j+k, i] = Value[k]
            else:
                try:
                    Param[j, i] = Value
                except:
                    type(Value) == str
    fullfilename = path.normpath(path.join(FolderPath, 'DeerParam.csv'))
    with open(fullfilename, 'w') as file:
        wr = csv.writer(file, dialect='excel',
                        delimiter='\t', lineterminator='\n')
        wr.writerow(ListOfFiles)
        for row in Param:
            wr.writerow(row)
    return


def ExportOneAsciiBckg(fullpath_to_filename, DataDict, ParamDict, mode='time'):
    'Export a .txt ascii datafile at the same location of the datafolder file.'
    'append the keyword "waveletname_denoised" at the end of ascii filename'
    fileID = path.split(fullpath_to_filename)
    Exp_parameter = ParamDict.get(fileID[1])
    itmax = Exp_parameter.get('itmax')
    itmin = Exp_parameter.get('itmin')
    fulldata = DataDict.get(fileID[1])
    TITL = fileID[1]
    if mode == 'time':
        time = fulldata[:, 0].ravel()
        y = fulldata[:, 5].ravel()
        cleanedy = y[~np.isnan(y)][itmin:itmax].real
        cleanedx = (time[~np.isnan(time)][itmin:itmax].real)/1000
        newfilename = str(fileID[1]+'.txt')
        fullnewfilename = path.join(fileID[0], newfilename)
    elif mode == 'timeden':
        time = fulldata[:, 0].ravel()
        ydenoised = fulldata[:, 5].ravel()
        cleanedy = ydenoised[~np.isnan(ydenoised)][itmin:itmax].real
        cleanedx = (time[~np.isnan(time)][itmin:itmax].real)/1000
        wavename = Exp_parameter.get('wavename')
        mode = Exp_parameter.get('denoisedmode')
        newfilename = str(fileID[1]+'_'+wavename+'_'+mode+'_denoised.txt')
    Bckg_type = Exp_parameter.get('Bckg_type')
    fullnewfilename = path.join(fileID[0], newfilename, Bckg_type)
    HeaderTITL = str(TITL+'_Header')
    Header = DataDict.get(HeaderTITL)
    header = '\n'.join(['Filename: ' + newfilename,
                        'Bckg_type: ' + Bckg_type,
                       'FirstColumn: ' + 'Time[us]',
                        'SecondColumn: ' + 'BckgSubtractedData'])
    if cleanedy.shape == cleanedx.shape:
        data = np.column_stack([cleanedx, cleanedy])
        makedirs(path.dirname(fullnewfilename), exist_ok=True)
        np.savetxt(fullnewfilename, data, fmt=['%.5e', '%.15e'],
                   delimiter='\t', header=header)
    # , footer='', comments='# ', encoding=None)
    return fullnewfilename


def ExportOneAscii(fullpath_to_filename, DataDict, ParamDict, mode='time'):
    'Export a .txt ascii datafile at the same location of the datafolder file.'
    'append the keyword "waveletname_denoised" at the end of ascii filename'
    fileID = path.split(fullpath_to_filename)
    Exp_parameter = ParamDict.get(fileID[1])
    itmax = Exp_parameter.get('itmax')
    itmin = Exp_parameter.get('itmin')
    fulldata = DataDict.get(fileID[1])
    TITL = fileID[1]
    if mode == 'time':
        time = fulldata[:, 0].ravel()
        y = fulldata[:, 5].ravel()
        cleanedy = y[~np.isnan(y)][:itmax].real
        cleanedx = (time[~np.isnan(time)][:itmax].real)/1000
        newfilename = str(fileID[1]+'.txt')
        fullnewfilename = path.join(fileID[0], newfilename)
    elif mode == 'timeden':
        time = fulldata[:, 0].ravel()
        ydenoised = fulldata[:, 5].ravel()
        cleanedy = ydenoised[~np.isnan(ydenoised)][:itmax].real
        cleanedx = (time[~np.isnan(time)][:itmax].real)/1000
        wavename = Exp_parameter.get('wavename')
        mode = Exp_parameter.get('denoisedmode')
        newfilename = str(fileID[1]+'_'+wavename+'_'+mode+'_denoised.txt')
    fullnewfilename = path.join(fileID[0], newfilename)
    Bckg_type = Exp_parameter.get('Bckg_type')
    HeaderTITL = str(TITL+'_Header')
    Header = DataDict.get(HeaderTITL)
    header = '\n'.join(['Filename: ' + newfilename,
                        'Bckg_type: ' + Bckg_type,
                       'FirstColumn: ' + 'Time[us]',
                        'SecondColumn: ' + 'BckgSubtractedData'])
    if cleanedy.shape == cleanedx.shape:
        data = np.column_stack([cleanedx, cleanedy])
        makedirs(path.dirname(fullnewfilename), exist_ok=True)
        np.savetxt(fullnewfilename, data, fmt=['%.5e', '%.15e'],
                   delimiter='\t', header=header)
    # , footer='', comments='# ', encoding=None)
    return fullnewfilename


def ExportMultipleDenoisedAscii(ListOfFiles, DataDict, Paramdict):
    for i in range(len(ListOfFiles)):
        filename = ListOfFiles[i]
        ExportOneAscii(filename, DataDict, Paramdict, mode='time')
    return
    # ListOfFiles = ImportMultipleNameFiles(folder, Extension='.DTA')
    # maxlen = MaxLengthOfFiles(ListOfFiles)
    # # fulldata3 = OpenComplexFiles(ListOfFiles,Scaling='n',polyorder=1,window_length=200)
    # data, abscissa, par = eprload(ListOfFiles[1], Scaling=None)
    # fulldata, Header = OpenMultipleComplexFiles2(ListOfFiles)
    # ncol = 7
    # fulldata2 = np.full((maxlen, ncol), np.nan, dtype="complex_")
    # fulldata2[:, 0] = fulldata[:, 0]
    # for i in range(ncol-1):
    #     fulldata2[:, i+1] = np.sum(fulldata[:,
    #                                [i+1, ncol+1+i, ncol*2+i+1, ncol*4+i+1]], axis=1)
    # fulldata3

    # Header = list(np.zeros((6*len(ListOfFiles),)))
    # for i in range(len(ListOfFiles)):
    #     Header[6*i] = par['XNAM']
    #     Header[6*i+1] = ListOfFiles[i].split('\\')[-1]
    #     Header[6*i+2] = ListOfFiles[i].split('\\')[-1]+str("_smooth_4pts")
    #     Header[6*i+3] = ListOfFiles[i].split('\\')[-1]+str("_normalized")
    #     Header[6*i+4] = ListOfFiles[i].split('\\')[-1]+str("_normalized")

    #     Header[6*i+5] = ListOfFiles[i].split('\\')[-1]+str("_normalized")


def ExportGlobalDataAnalysis(DataDict):
    return
    # ListOfFiles = ImportMultipleNameFiles(folder, Extension='.DTA')
    # maxlen = MaxLengthOfFiles(ListOfFiles)
    # # fulldata3 = OpenComplexFiles(ListOfFiles,Scaling='n',polyorder=1,window_length=200)
    # data, abscissa, par = eprload(ListOfFiles[1], Scaling=None)
    # fulldata, Header = OpenMultipleComplexFiles2(ListOfFiles)
    # ncol = 7
    # fulldata2 = np.full((maxlen, ncol), np.nan, dtype="complex_")
    # fulldata2[:, 0] = fulldata[:, 0]
    # for i in range(ncol-1):
    #     fulldata2[:, i+1] = np.sum(fulldata[:,
    #                                [i+1, ncol+1+i, ncol*2+i+1, ncol*4+i+1]], axis=1)
    # fulldata3

    # Header = list(np.zeros((6*len(ListOfFiles),)))
    # for i in range(len(ListOfFiles)):
    #     Header[6*i] = par['XNAM']
    #     Header[6*i+1] = ListOfFiles[i].split('\\')[-1]
    #     Header[6*i+2] = ListOfFiles[i].split('\\')[-1]+str("_smooth_4pts")
    #     Header[6*i+3] = ListOfFiles[i].split('\\')[-1]+str("_normalized")
    #     Header[6*i+4] = ListOfFiles[i].split('\\')[-1]+str("_normalized")

    #     Header[6*i+5] = ListOfFiles[i].split('\\')[-1]+str("_normalized")


def ComputeRMSE(y, yfit, p0):
    '''
    Compute the normalized residual sum of square of the residual of a function
    or Root Mean Square of Error (RMSE)
    See by instance :
    https://statisticsbyjim.com/regression/root-mean-square-error-rmse/
    Script written by Timothée Chauviré 10/26/2023

    Parameters
    ----------
    y : experimental data
        TYPE : Numpy data array
    yfit : experimental data
        TYPE : Numpy data array
    p0 : paremeters used for the fit
        TYPE : Numpy data array
    Returns
    -------
    RMSE : normalized residual sum of square or Root mean square of error
        TYPE : real float value
    '''
    NumMeas = y.shape[0]
    NumParams = len(p0)
    resnorm = np.sum((y-yfit)**2)
    RMSE = np.sqrt(resnorm/(NumMeas - NumParams))
    return RMSE


def GetExpParamValues(filename=str):
    '''
    Function to catch pulse_EPR parameters from Bruker pulse experiment data.
    The function uses deerlab.deerload function to import the parameters
    description.

    Parameters
    ----------
    filename : full path to the filename
        DESCRIPTION. string

    Returns
    -------
    BrukerParamDict : Dictionnary of Bruker common parameters name.
                 By instance BrukerParamDict['d1'] will return the value of
                 d1 (tau1 in Bruker language
        DESCRIPTION. Dictionnary

    '''
    _, _, par = dl.deerload(filename, full_output=True)
    PulseParameters = par['DSL']['ftEpr']['PlsSPELGlbTxt']
    ListBrukerNames = ['d0', 'd1', 'd2', 'd3', 'd4', 'd5', 'd6', 'd7', 'd8',
                       'd9', 'd10', 'd11', 'd12', 'd30', 'd31', 'p0', 'p1',
                       'p2', 'p3', 'p4', 'p5', 'DAF', 'af0', 'af1', 'af2']
    BrukerParamDict = {}
    for i in range(len(ListBrukerNames)):
        KeywordToFind = ListBrukerNames[i]
        keywordlength = int(len(ListBrukerNames[i]))
        a = int()
        b = int()
        c = str()
        a = PulseParameters.find(KeywordToFind)  # find 1st index of keyword
        if a != -1:
            # find second index with adjacent ;
            b = PulseParameters.find(';', a)
            if b != -1:
                c = PulseParameters[a+keywordlength:b-1]
                try:
                    value = float(c.replace('=', '').strip())
                    BrukerParamDict.update({KeywordToFind: value, })
                except ValueError:
                    pass  # do nothing
    return BrukerParamDict


def ReadDenoisedData(fullpath_to_filename):
    Header = {}
    x, y, y_den = [], [], []
    with open(fullpath_to_filename, 'r') as f:
        lines = f.readlines()
        for i in range(len(lines)):
            if lines[i][0] == '%':
                pairs = lines[i].split('\t')
                Header.update({str(i): pairs})
            else:
                pairs = lines[i].split('\t')
                x.append(float(pairs[0]))
                y_den.append(float(pairs[1]))
                y.append(float(pairs[2]))
    return np.asarray(y), np.asarray(y_den), np.asarray(x), Header


def is_float(s):
    if re.match(r'(?i)^\s*[+-]?(?:inf(inity)?|nan|(?:\d+\.?\d*|\.\d+)(?:e[+-]'
                '?\d+)?)\s*$', s) is None:
        return False
    else:
        return True


def ReadDEERAnalysisAsciiFiles(ListOfFiles):
    '''
    Read the ascii File generated by the DEER Analysis 2022 matlab program.
    and generate a python dictionnary with all the tikhonov regularisation and
    uncertaintie parameters.

    Parameters
    ----------
    Filename : TYPE
        DESCRIPTION.

    Returns
    -------
    parameter : Dictionnary containing the para

    '''
    parameter = {}
    with open(ListOfFiles[0], 'r') as f:
        lines = f.readlines()
        for i in range(len(lines)):
            print(i)
            pairs = lines[i].split(":")
            print(pairs)
            parameter.update({pairs[0]: pairs[1].strip()
                              for pair in pairs if len(pairs) == 2})
            parameter.update({pairs[0]: float(pairs[1].strip().split(' ')[0])
                              for pair in pairs if len(pairs) == 2
                              and is_float(pairs[1].strip().split(' ')[0])})
    return parameter


def importasciiSVDData(fullpath_to_filename):
    Header = {}
    x, y = [], []
    with open(fullpath_to_filename, 'r') as f:
        lines = f.readlines()
        for i in range(len(lines)):
            if lines[i][0] == '%':
                pairs = lines[i].split('\t')
                Header.update({str(i): pairs})
            else:
                pairs = lines[i].split('\t')
                x.append(float(pairs[0]))
                y.append(float(pairs[1]))
    return np.asarray(y), np.asarray(x), Header


def importasciiTimeData(fullpath_to_filename):
    Header = {}
    t, y1, y2 = [], [], []
    with open(fullpath_to_filename, 'r') as f:
        lines = f.readlines()
        for i in range(len(lines)):
            if lines[i][0] == '%':
                pairs = lines[i].split('\t')
                Header.update({str(i): pairs})
            else:
                pairs = lines[i].split('\t')
                t.append(float(pairs[0]))
                y1.append(float(pairs[1]))
                y2.append(float(pairs[2]))
    return np.asarray(y1), np.asarray(y2), np.asarray(t), Header


def importasciiTimeDataPB(fullpath_to_filename):
    #     Header = {}
    t, y1 = [], []
    with open(fullpath_to_filename, 'r') as f:
        lines = f.readlines()
        for i in range(len(lines)):
            pairs = lines[i].split(' ')
            t.append(float(pairs[0]))
            y1.append(float(pairs[1]))
    return np.asarray(t), np.asarray(y1)
# def GetNoiseLevelFromImaginaryPart(Filename):
#     '''
#     Noise level estimated from the root-mean-square
#     amplitude of the imaginary part after phase correction.

#     Parameters
#     ----------
#     Filename : Bruker filename for DEER DAta analysis (.DTA/.DSC or .par/.spc)
#         DESCRIPTION.

#     Raises
#     ------
#     ValueError
#         DESCRIPTION.

#     Returns
#     -------
#     sigma_noise : Noise level parameterne.

#     '''
#     # Parameters Initialization
#     sigma_noise = float()
#     # Import the data
#     data, abscissa, par = eprload(FileName, Scaling=None)
#     if (data.shape[0] != np.ravel(data).shape[0]):
#         raise ValueError('The file {0} is\'t a column vector'
#                          .format(par['TITL']))
#     else:
#         # Achieve automatic phase correction
#         data = np.ravel(data)
#         new_data, _ = automatic_phase(vector=data, pivot1=int(data.shape[0]/2),
#                                       funcmodel='minfunc')
#         data_real = new_data.real
#         data_imag = new_data.imag
#         # Achieve automatic base line correction correction
#         data_real_new, _, _, _ = basecorr1D(x=abscissa, y=data_real,
#                                             polyorder=polyorder, window=window)
#         data_imag_new, _, _, _ = basecorr1D(x=abscissa, y=data_imag,
#                                             polyorder=polyorder, window=window)

#         # Two noises-level are computed :
#         # 1) With the full imaginary part "sigma_noise"
#         # 2) With half of the imaginary part "sigma_noise_half"

#         npts = abscissa.shape[0]
#         center = np.floor(npts/2)
#         sigma_noise = np.std(data_imag_new[:, 0])
#         sigma_noise_half = np.std(data_imag_new[center-np.floor(npts/4):
#                                                 center+np.floor(npts/4)], 0)

#         return sigma_noise, sigma_noise_half
