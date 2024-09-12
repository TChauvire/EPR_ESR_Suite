# -*- coding: utf-8 -*-
"""
Created on Thu Nov 16 11:46:11 2023
Analysis of T1 data

This script enables to import multiple T1 relaxation measurements with files 
in DSC/DTA Bruker format.
This script is intended to reconstruct the non linear timescale with the d2,
d30 bruker parameters and the number of points with the following loop:
    d2 = 200; d30 = 2; newx = [d2];
    for j in range(npts-1):
        dx = float(newx[j]+(j+1)*d30)
        newx.extend([dx])
The curve is then fitted on the real part and the absolute part with the 
following equation (see Hung et al. 2000):
1)    I(t) = I0 - A1*exp(-t/T1) for the monoexponential model
2)    I(t) = I0 - AS*exp(-t/TS)-AL*exp(-t/TL) for the
                biexponential model (S=Short component and L = long component)

@author: Timothee Chauvire tsc84@cornell.edu
"""
import matplotlib.pyplot as plt
import numpy as np
from ImportMultipleFiles import eprload, ImportMultipleNameFiles, datasmooth
from ImportMultipleFiles import MaxLengthOfFiles
from basecorr1D import basecorr1D
from automatic_phase import automatic_phase
from scipy.optimize import curve_fit
from os import path


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


plt.close('all')
# File location in folder
# folder = 'C:\\Users\\tim_t\\Python\\EPRImportScripts\\DataFilesRelaxationTime'\
#          '\\T1\\T1LogScale'  # Importing global path to files
folder = 'D:\\Documents\\Recherches\\ResearchAssociatePosition\\Data\\'\
    'PulsedEPR\\2024\\2024_Rebecca\\March2023\\240307\\T1'
ListOfFiles = ImportMultipleNameFiles(FolderPath=folder, Extension='DSC')
# Check for the maximum length datafiles
npts = MaxLengthOfFiles(ListOfFiles)
# ninit = int(float(npts/16))
# Initialize the variable:
Y = np.full((npts, 4*len(ListOfFiles)), np.nan, dtype="float")
Yname = list(np.zeros((4*len(ListOfFiles,))))
FitValue = np.full((npts, 5*len(ListOfFiles)), np.nan, dtype="float")
FitHeader = list(np.zeros((5*len(ListOfFiles),)))
FitParam = np.full((14, 2*len(ListOfFiles)), np.nan, dtype="float")
FitParamHeader = list(np.zeros((15,)))
NameHeader = list(np.zeros((len(ListOfFiles,))))
# import all and achieve datatreatment
for i in range(len(ListOfFiles)):
    fileID = path.split(ListOfFiles[i])
    data, x, par = eprload(ListOfFiles[i])
    npts = data.shape[0]
    # Generate the non linear time scale:
    d2 = 200
    d30 = 400
    newx = [d2]
    # newx = x.real.ravel()
    for j in range(npts-1):
        dx = float(newx[j]+(j+1)*d30)
        newx.extend([dx])
    # Achieve the phase correction
    pivot = int(np.floor(data.shape[0]/2))
    new_data, _ = automatic_phase(
        vector=data, pivot1=pivot, funcmodel='minfunc')  # data are 180° phased
    yreal = -new_data.real.ravel()  # Correction of the phase with the factor
    yimag = -new_data.imag.ravel()  # exp(pi*j) = -1
    Y[0:npts, 4*i] = newx
    Y[0:npts, 4*i+1] = yreal.ravel()
    Y[0:npts, 4*i+2] = yimag.ravel()
    Y[0:npts, 4*i+3] = np.abs(data[0:npts]).ravel()
    Yname[4*i] = 'Time (ns)'
    Yname[4*i+1] = str(fileID[1]+'_real')
    Yname[4*i+2] = str(fileID[1]+'_imag')
    Yname[4*i+3] = str(fileID[1]+'_abs')
    ## Monoexponential Data fitting ##
    # Starting guess value for the parameters
    x01 = [7e5, 8e5, 5e6]
    #x01 = [1e8, 1e8, 1e8]
    ub1 = [1e11, 1e11, 1e11]
    lb1 = [0, 0, 0]
    b1 = (lb1, ub1)
    # Monoexponential function definition

    def monoexp(x, a, b, c):
        y = a - (b)*(np.exp((-1.0)*(x / c)))
        return y
    # Monoexponential Fitting of the real part:
    popt1, pcov1 = curve_fit(monoexp, newx, yreal[:,].ravel(),
                             p0=x01, sigma=None, absolute_sigma=False,
                             check_finite=None, bounds=b1)
    perr1 = np.sqrt(np.diag(pcov1))
    yfit1 = monoexp(newx, popt1[0], popt1[1], popt1[2])
    residual1 = np.subtract(yreal[0:npts,].ravel(), yfit1[0:npts,].ravel())
    RMSE1 = ComputeRMSE(yreal[0:npts,], yfit1[0:npts,], popt1)
    ## Double exponential fitting ##
    # Starting guess value for the parameters
    #x02 = [popt1[0], popt1[1], popt1[2], 1e3, 1e3]
    x02 = [7e5, 4e4, 2e3, 8e5, 5e6]
    ub2 = [1e11, 1e11, 1e11, 1e11, 1e11]
    lb2 = [0, 0, 0, 0, 0]
    b2 = (lb2, ub2)
    # Double exponential function definition

    def biexp(x, a, b, c, d, e):
        y = a - b*(np.exp((-1.0)*(x/c))) - d*(np.exp((-1.0)*(x/e)))
        return y
    # Double exponential Fitting of the real part:
    popt2, pcov2 = curve_fit(biexp, newx, yreal[:,].ravel(),
                             p0=x02, sigma=None, absolute_sigma=False,
                             check_finite=None, bounds=b2)
    perr2 = np.sqrt(np.diag(pcov2))
    yfit2 = biexp(newx, popt2[0], popt2[1], popt2[2], popt2[3], popt2[4])
    residual2 = np.subtract(yreal[:,].ravel(), yfit2[:,].ravel())
    RMSE2 = ComputeRMSE(yreal[:,].ravel(), yfit2[:,], popt2)
    ## Triple exponential fitting ##
    # Starting guess value for the parameters
    #x03 = [popt2[0], popt2[1], popt2[2], popt2[3], popt2[4], 1e1, 1e1]
    x03 = x02 = [7e5, 4e4, 2e3, 2e5, 5e5, 8e5, 5e6]
    ub3 = [1e11, 1e11, 1e11, 1e11, 1e11, 1e11, 1e11]
    lb3 = [0, 0, 0, 0, 0,  0, 0]
    b3 = (lb3, ub3)
    # Triple exponential function definition

    def triexp(x, a, b, c, d, e, f, g):
        y = a - b*(np.exp((-1.0)*(x/c))) - d * \
            (np.exp((-1.0)*(x/e))) - f*(np.exp((-1.0)*(x/g)))
        return y
    # Double exponential Fitting of the real part:
    popt3, pcov3 = curve_fit(triexp, newx, yreal[:,].ravel(),
                             p0=x03, sigma=None, absolute_sigma=False,
                             check_finite=None, bounds=b3)
    perr3 = np.sqrt(np.diag(pcov3))
    yfit3 = triexp(newx, popt3[0], popt3[1], popt3[2],
                   popt3[3], popt3[4], popt3[5], popt3[6])
    residual3 = np.subtract(yreal[:,].ravel(), yfit3[:,].ravel())
    RMSE3 = ComputeRMSE(yreal[:,].ravel(), yfit3[:,], popt3)
    # Output the Fitted Data
    FitValue[0:npts, 5*i] = newx
    FitValue[0:npts, 5*i+1] = yfit1[0:npts,].ravel()
    FitValue[0:npts, 5*i+2] = residual1[0:npts,].ravel()
    FitValue[0:npts, 5*i+3] = yfit2[0:npts,].ravel()
    FitValue[0:npts, 5*i+4] = residual2[0:npts,].ravel()
    FitHeader[5*i] = 'Time (ns)'
    FitHeader[5*i+1] = 'Monoexponential Fitting of the Real Part'
    FitHeader[5*i+2] = 'Residual'
    FitHeader[5*i+3] = 'Biexponential Fitting of the Real Part'
    FitHeader[5*i+4] = 'Residual'
    # Output of the fitting parameters
    FitParam[0, i] = popt1[1]
    FitParam[1, i] = perr1[1]
    FitParam[2, i] = popt1[2]
    FitParam[3, i] = perr1[2]
    FitParam[4, i] = RMSE1
    FitParam[5, i] = popt2[1]
    FitParam[6, i] = perr2[1]
    FitParam[7, i] = popt2[2]
    FitParam[8, i] = perr2[2]
    FitParam[9, i] = popt2[3]
    FitParam[10, i] = perr2[3]
    FitParam[11, i] = popt2[4]
    FitParam[12, i] = perr2[4]
    FitParam[13, i] = RMSE2
    NameHeader[i] = str(fileID[1]+'real_part')
    # plot the data in a figure for inspection of the goodness of fit
    plt.figure(i)
    plt.suptitle(fileID[1])
    line1, = plt.plot(Y[0:npts, 4*i]/1000, Y[0:npts, 4*i+1],
                      'k', label='real part')
    line2, = plt.plot(FitValue[:, 5*i]/1000, FitValue[:, 5*i+1], 'r',
                      label='monoexponential fit')
    # line3, = plt.plot(FitValue[:, 5*i], FitValue[:, 5*i+3], 'b',
    #                   label='biexponential fit')
    line3, = plt.plot(FitValue[:, 5*i]/1000, yfit2[0:npts,].ravel(), 'b',
                      label='biexponential fit')
    line4, = plt.plot(Y[0:npts, 4*i]/1000, Y[0:npts, 4*i+2], 'k--', linewidth=2,
                      label='imaginary part')
    plt.xlabel('Time [us]')
    plt.ylabel('Inverted Echo Intensity [a.u.]')
    plt.legend()
    figManager = plt.get_current_fig_manager()
    figManager.window.showMaximized()
    plt.show()
    plt.tight_layout()

## Header for the fitting parameters ##
FitParamHeader[0] = 'Name of the Sample'
FitParamHeader[1] = 'A1'
FitParamHeader[2] = 'Error (A1)'
FitParamHeader[3] = 'T1'
FitParamHeader[4] = 'Error (T1)'
FitParamHeader[5] = 'RMSE Root mean squared of Error Monoexponential'
FitParamHeader[6] = 'AS'
FitParamHeader[7] = 'Error (AS)'
FitParamHeader[8] = 'TS'
FitParamHeader[9] = 'Error (TS)'
FitParamHeader[10] = 'AL'
FitParamHeader[11] = 'Error (AL)'
FitParamHeader[12] = 'TL'
FitParamHeader[13] = 'Error (TL)'
FitParamHeader[14] = 'RMSE Root mean squared of Error Biexponential'
