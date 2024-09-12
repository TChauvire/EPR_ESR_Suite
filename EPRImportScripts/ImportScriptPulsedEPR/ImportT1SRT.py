# -*- coding: utf-8 -*-
"""
Created on Thu Nov 16 11:46:11 2023
Analysis of T1 data

This script enables to import multiple T1 relaxation measurements with files 
in DSC/DTA Bruker format. 
An option for smoothing the curve with a number of point Npoint is proposed.
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
from os import path, makedirs


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
folder = 'D:\\Documents\\Recherches\\ResearchAssociatePosition\\Data\\'\
    'PulsedEPR\\2024\\BorisSamples\\240506_TC_WeakSample\\T1SRT\\'
Name = '240506_TC_WeakSample\\T1SRT'
# Importing global path to files
ListOfFiles = ImportMultipleNameFiles(FolderPath=folder, Extension='DSC')
# Check for the maximum length datafiles
npts = MaxLengthOfFiles(ListOfFiles)
nINTG = 60
# ninit = int(float(npts/16))
# Initialize the variable:
Y = np.full((npts, 4*len(ListOfFiles)), np.nan, dtype="float")
Yname = list(np.zeros((4*len(ListOfFiles,))))
FitValue = np.full((len(ListOfFiles), 4), np.nan, dtype="float")
FitValue2 = np.full((len(ListOfFiles), 2), np.nan, dtype="float")
FitHeader = list(np.zeros((4,)))
FitParam = np.full((5,), np.nan, dtype="float")
FitParamHeader = list(np.zeros((5,)))
# NameHeader = list(np.zeros((len(ListOfFiles,))))
# import all and achieve datatreatment
for i in range(len(ListOfFiles)):
    fileID = path.split(ListOfFiles[i])
    SRTValue = fileID[1].split('_')[-1]
    FitValue[i, 0] = float(SRTValue[:-4])
    data, x, par = eprload(ListOfFiles[i])
    npts = data.shape[0]
    newx = x.real.ravel()
    # Achieve the phase correction
    pivot = int(np.floor(data.shape[0]/2))
    new_data, _ = automatic_phase(
        vector=data, pivot1=pivot, funcmodel='minfunc')  # data are 180° phased
    yreal = new_data.real.ravel()  # Correction of the phase with the factor
    yimag = new_data.imag.ravel()  # exp(pi*j) = -1
    Y[0:npts, 4*i] = newx.ravel()
    Y[0:npts, 4*i+1] = yreal.ravel()
    Y[0:npts, 4*i+2] = yimag.ravel()
    Y[0:npts, 4*i+3] = np.abs(data[0:npts]).ravel()
    Yname[4*i] = 'Time (ns)'
    Yname[4*i+1] = str(fileID[1]+'_real')
    Yname[4*i+2] = str(fileID[1]+'_imag')
    Yname[4*i+3] = str(fileID[1]+'_abs')
    IntgValue1 = np.trapz(yreal[0:nINTG])
    IntgValueBaseline = np.trapz(yreal[-nINTG:])
    FitValue[i, 1] = IntgValue1-IntgValueBaseline
    plt.figure(i)
    plt.suptitle(fileID[1])
    line1, = plt.plot(Y[0:npts, 4*i], Y[0:npts, 4*i+1], 'k', label='real part')
    line2, = plt.plot(Y[0:npts, 4*i], Y[0:npts, 4*i+2], 'r', label='imag part')
    xvalue = int(Y[nINTG, 4*i])
    line3 = plt.axvline(x=xvalue, color='black', ls='--', lw=1)
    plt.legend()
    plt.xlabel('Time [ns]', fontsize=16, fontweight='bold')
    plt.ylabel('Intensity [a.u]', fontsize=16, fontweight='bold')
    figManager = plt.get_current_fig_manager()
    figManager.window.showMaximized()
    plt.show()
    plt.tight_layout()
    makedirs(fileID[0] + '\\Figures\\', exist_ok=True)
    plt.savefig(fileID[0] + '\\Figures\\figure{0}.png'.format(i))

FitValue2 = np.sort(FitValue, axis=0)
plt.figure()
plt.suptitle(Name)
l1 = plt.plot(FitValue2[:, 0], FitValue2[:, 1]/max(FitValue2[:, 1]), 'k-o')
plt.ylim(0, )
plt.xlabel('Shot repetition time [us]', fontsize=16, fontweight='bold')
plt.ylabel('Echo Intensity [a.u]', fontsize=16, fontweight='bold')
## Monoexponential Data fitting ##
# Starting guess value for the parameters
x01 = [1, 1e2, 1e2]
ub1 = [2, 1e11, 1e11]
lb1 = [0, 0, 0]
b1 = (lb1, ub1)
# Monoexponential function definition


def monoexp(x, a, b, c):
    y = a - (b)*(np.exp((-1.0)*(x / c)))
    return y
# Monoexponential Fitting of the real part:


popt1, pcov1 = curve_fit(monoexp, FitValue2[:, 0], FitValue2[:, 1] /
                         max(FitValue2[:, 1]), p0=x01, sigma=None,
                         absolute_sigma=False, check_finite=None, bounds=b1)
perr1 = np.sqrt(np.diag(pcov1))
yfit1 = monoexp(FitValue2[:, 0], popt1[0], popt1[1], popt1[2])
residual1 = np.subtract(FitValue2[:, 1]/max(FitValue2[:, 1]), yfit1[:,])
RMSE1 = ComputeRMSE(FitValue2[:, 1]/max(FitValue2[:, 1]), yfit1[:,], popt1)
l2 = plt.plot(FitValue2[:, 0], yfit1, 'r',
              label='Monoexponential Fit: T1= {0} us'.format(popt1[2]))
plt.savefig(fileID[0] + '\\Figures\\T1_SRTFit.png')
#     ## Double exponential fitting ##
#     # Starting guess value for the parameters
#     x02 = [popt1[0], popt1[1], popt1[2], 1e3, 1e3]
#     ub2 = [1e11, 1e11, 1e11, 1e11, 1e11]
#     lb2 = [0, 0, 0, 0, 0]
#     b2 = (lb2, ub2)
#     # Double exponential function definition

#     def biexp(x, a, b, c, d, e):
#         y = a - b*(np.exp((-1.0)*(x/c))) - d*(np.exp((-1.0)*(x/e)))
#         return y
#     # Double exponential Fitting of the real part:
#     popt2, pcov2 = curve_fit(biexp, newx[:,], yreal[:,].ravel(),
#                              p0=x02, sigma=None, absolute_sigma=False,
#                              check_finite=None, bounds=b2)
#     perr2 = np.sqrt(np.diag(pcov2))
#     yfit2 = biexp(newx, popt2[0], popt2[1], popt2[2], popt2[3], popt2[4])
#     residual2 = np.subtract(yreal[:,].ravel(), yfit2[:,].ravel())
#     RMSE2 = ComputeRMSE(yreal[:,].ravel(), yfit2[:,], popt2)
# Output the Fitted Data
FitValue[:, 0] = FitValue2[:, 0]
FitValue[:, 1] = FitValue2[:, 1]
FitValue[:, 2] = yfit1[0:npts,].ravel()
FitValue[:, 3] = residual1[0:npts,].ravel()

FitHeader[0] = 'Shot Repetition Time (us)'
FitHeader[1] = Name+'Experimental Data'
FitHeader[2] = Name+'Monoexponential Fit Data'
FitHeader[3] = Name+'Residual Fit'


FitParam[0] = popt1[1]
FitParam[1] = perr1[1]
FitParam[2] = popt1[2]
FitParam[3] = perr1[2]
FitParam[4] = RMSE1

FitParamHeader[0] = 'A1'
FitParamHeader[1] = 'Error (A1)'
FitParamHeader[2] = 'T1'
FitParamHeader[3] = 'Error (T1)'
FitParamHeader[4] = 'RMSE Root mean squared of Error Monoexponential'
# FitParamHeader[6] = 'AS'
# FitParamHeader[7] = 'Error (AS)'
# FitParamHeader[8] = 'TS'
# FitParamHeader[9] = 'Error (TS)'
# FitParamHeader[10] = 'AL'
# FitParamHeader[11] = 'Error (AL)'
# FitParamHeader[12] = 'TL'
# FitParamHeader[13] = 'Error (TL)'
# FitParamHeader[14] = 'RMSE Root mean squared of Error Biexponential'
