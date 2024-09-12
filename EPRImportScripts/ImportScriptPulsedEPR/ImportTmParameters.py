# -*- coding: utf-8 -*-
"""
Created on Thu Nov 16 11:46:11 2023
Analysis of Tm data
Be carful of the factor 2 (or not) in Pulsespel and in this script for the
fitting

This script enables to import one file in DSC/DTA Bruker format. 
No baseline correction is applied.
An option for smoothing the curve with a number of point Npoint is proposed.
The curve is then fitted on the real part and the absolute part with the 
following equation (Biological Magnetic Resonance Vol.19 Distance
Measurements in Biological Systems by EPR p 372:
I(t)=I0 * exp(-(2t/Tm)^x)+y0 %(Be careful some data have a 2tau plotted)
So check the pulsespel program ; Here I multiplied the abscissa by two ;) )

Two simulations are done, one with x = 1, and one with x varying from 0
to 5 for both real part and absolute part...
@author: tim_t
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
folder = 'D:\\Documents\\Recherches\\Postdoc\\Data\\CraneLab_Flavoprotein\\'\
    'Sidddarth\\2019\\Apr23_iLAW_iLov_EcA_EcW_BL21\\Tm\\'
#folder = 'C:\\Users\\tim_t\\Python\\EPRImportScripts\\DataFilesRelaxationTime\\Tm'
# Importing global path to files
ListOfFiles = ImportMultipleNameFiles(FolderPath=folder, Extension='DSC')
# Check for the maximum length datafiles
npts = MaxLengthOfFiles(ListOfFiles)
ninit = int(float(npts/16))
# Initialize the variable:
Y = np.full((npts, 4*len(ListOfFiles)), np.nan, dtype="float")
Yname = list(np.zeros((4*len(ListOfFiles,))))
FitValue = np.full((npts, 9*len(ListOfFiles)), np.nan, dtype="float")
FitHeader = list(np.zeros((9*len(ListOfFiles),)))
FitParam = np.full((12, 2*len(ListOfFiles)), np.nan, dtype="float")
FitParamHeader = list(np.zeros((13,)))
NameHeader = list(np.zeros((2*len(ListOfFiles,))))
# import all and achieve datatreatment
for i in range(len(ListOfFiles)):
    fileID = path.split(ListOfFiles[i])
    data, x, par = eprload(ListOfFiles[i])
    x = (x.real.ravel()-x[0])/2+x[0]  # 2 tau scale to 1 tau scale conversion
    npts = data.shape[0]
    pivot = int(np.floor(data.shape[0]/2))
    new_data, _ = automatic_phase(
        vector=data, pivot1=pivot, funcmodel='minfunc')
    yreal = new_data.real.ravel()
    yimag = new_data.imag.ravel()
    Y[0:npts, 4*i] = x.ravel()
    Y[0:npts, 4*i+1] = yreal.ravel()
    Y[0:npts, 4*i+2] = yimag.ravel()
    Y[0:npts, 4*i+3] = np.abs(data[0:npts]).ravel()
    Yname[4*i] = 'Time (ns)'
    Yname[4*i+1] = str(fileID[1]+'_real')
    Yname[4*i+2] = str(fileID[1]+'_imag')
    Yname[4*i+3] = str(fileID[1]+'_abs')

    ## Monoexponential Data fitting ##
    # Starting guess value for the parameters
    endvalue = np.mean(Y[-4:, 4*i+1].real)
    if endvalue < 0:
        endvalue = 1
    x01 = [endvalue, np.mean(Y[0:4, 4*i+1].real), 5000]
    x02 = [endvalue, np.mean(Y[0:4, 4*i+3].real), 5000]
    ub1 = [1e11, 1e11, 1e11]
    lb1 = [0, 0, 0]
    b1 = (lb1, ub1)
    # Monoexponential function definition

    def monoexp(x, a, b, c):
        y = a + (b)*(np.exp((-2) * (x / c)))
        return y
    # Monoexponential Fitting of the real part:
    popt1, pcov1 = curve_fit(monoexp, x[ninit:,].ravel(), yreal[ninit:,].ravel(),
                             p0=x01, sigma=None, absolute_sigma=False,
                             check_finite=None, bounds=b1)
    perr1 = np.sqrt(np.diag(pcov1))
    yfit1 = monoexp(x, popt1[0], popt1[1], popt1[2])
    residual1 = np.subtract(yreal[0:npts,].ravel(), yfit1[0:npts,].ravel())
    RMSE1 = ComputeRMSE(yreal[0:npts,], yfit1[0:npts,], popt1)
    # Monoexponential Fitting of the absolute part:
    absdata = np.abs(data[0:npts,].ravel())
    popt2, pcov2 = curve_fit(monoexp, x[ninit:,].ravel(), absdata[ninit:,],
                             p0=x02, sigma=None, absolute_sigma=False,
                             check_finite=None, bounds=b1)
    perr2 = np.sqrt(np.diag(pcov2))
    yfit2 = monoexp(x, popt2[0], popt2[1], popt2[2])
    residual2 = np.subtract(yreal[0:npts,].ravel(), yfit2[0:npts,].ravel())
    RMSE2 = ComputeRMSE(np.abs(data[0:npts,]), yfit2[0:npts,], popt2)
    ## Stretched exponential Data fitting ##
    # Starting guess value for the parameters
    x03 = [popt1[0], popt1[1], popt1[2], 1]
    x04 = [popt2[0], popt2[1], popt2[2], 1]
    ub2 = [1e11, 1e11, 1e11, 6]
    lb2 = [0, 0, 0, 0]
    b2 = (lb2, ub2)
    # Stretched exponential function definition

    def strexp(x, a, b, c, d):
        y = a + b*(np.exp((-2)*(x/c))**(d))
        return y
    # Stretched exponential Fitting of the real part:
    popt3, pcov3 = curve_fit(strexp, x[ninit:,].ravel(), yreal[ninit:,].ravel(),
                             p0=x03, sigma=None, absolute_sigma=False,
                             check_finite=None, bounds=b2)
    perr3 = np.sqrt(np.diag(pcov3))
    yfit3 = strexp(x, popt3[0], popt3[1], popt3[2], popt3[3])
    residual3 = np.subtract(yreal[0:npts,].ravel(), yfit3[0:npts,].ravel())
    RMSE3 = ComputeRMSE(yreal[0:npts,], yfit3[0:npts,], popt3)
    # Stretched exponential Fitting of the absolute part:
    popt4, pcov4 = curve_fit(strexp, x[ninit:,].ravel(), absdata[ninit:,],
                             p0=x04, sigma=None, absolute_sigma=False,
                             check_finite=None, bounds=b2)
    perr4 = np.sqrt(np.diag(pcov4))
    yfit4 = strexp(x, popt4[0], popt4[1], popt4[2], popt4[3])
    residual4 = np.subtract(yreal[0:npts,].ravel(), yfit4[0:npts,].ravel())

    RMSE4 = ComputeRMSE(np.abs(data[0:npts,]), yfit4[0:npts,], popt4)
    # Output the Fitted Data
    FitValue[0:npts, 9*i] = x[:,].ravel()
    FitValue[0:npts, 9*i+1] = yfit1[0:npts,].ravel()
    FitValue[0:npts, 9*i+2] = residual1[0:npts,].ravel()
    FitValue[0:npts, 9*i+3] = yfit3[0:npts,].ravel()
    FitValue[0:npts, 9*i+4] = residual3[0:npts,].ravel()
    FitValue[0:npts, 9*i+5] = yfit2[0:npts,].ravel()
    FitValue[0:npts, 9*i+6] = residual2[0:npts,].ravel()
    FitValue[0:npts, 9*i+7] = yfit4[0:npts,].ravel()
    FitValue[0:npts, 9*i+8] = residual4[0:npts,].ravel()

    FitHeader[9*i] = 'Time (ns)'
    FitHeader[9*i+1] = 'Monoexponential Decay Fitting of the Real Part'
    FitHeader[9*i+2] = 'Residual'
    FitHeader[9*i+3] = 'Stretched Decay Fitting of the Real Part'
    FitHeader[9*i+4] = 'Residual'
    FitHeader[9*i+5] = 'Monoexponential Decay Fitting of the Absolute Part'
    FitHeader[9*i+6] = 'Residual'
    FitHeader[9*i+7] = 'Stretched Decay Fitting of the Absolute Part'
    FitHeader[9*i+8] = 'Residual'

    FitParam[0, 2*i] = popt1[1]
    FitParam[1, 2*i] = perr1[1]
    FitParam[2, 2*i] = popt1[2]
    FitParam[3, 2*i] = perr1[2]
    FitParam[4, 2*i] = RMSE1
    FitParam[5, 2*i] = popt3[1]
    FitParam[6, 2*i] = perr3[1]
    FitParam[7, 2*i] = popt3[2]
    FitParam[8, 2*i] = perr3[2]
    FitParam[9, 2*i] = popt3[3]
    FitParam[10, 2*i] = perr3[3]
    FitParam[11, 2*i] = RMSE3
    FitParam[0, 2*i+1] = popt2[1]
    FitParam[1, 2*i+1] = perr2[1]
    FitParam[2, 2*i+1] = popt2[2]
    FitParam[3, 2*i+1] = perr2[2]
    FitParam[4, 2*i+1] = RMSE2
    FitParam[5, 2*i+1] = popt4[1]
    FitParam[6, 2*i+1] = perr4[1]
    FitParam[7, 2*i+1] = popt4[2]
    FitParam[8, 2*i+1] = perr4[2]
    FitParam[9, 2*i+1] = popt4[3]
    FitParam[10, 2*i+1] = perr4[3]
    FitParam[11, 2*i+1] = RMSE4
    NameHeader[2*i] = str(fileID[1]+'real_part')
    NameHeader[2*i+1] = str(fileID[1]+'abs_part')
    fig, axes = plt.subplots(2, 1)
    plt.suptitle(fileID[1])
    axes[0].plot(Y[0:npts, 4*i], Y[0:npts, 4*i+1], 'k',
                 FitValue[:, 9*i], FitValue[:, 9*i+1], 'r',
                 FitValue[:, 9*i], FitValue[:, 9*i+3], 'b',
                 Y[0:npts, 4*i], Y[0:npts, 4*i+2], 'k')
    axes[0].set_title('real part')
    axes[1].plot(Y[0:npts, 4*i], Y[0:npts, 4*i+3], 'k',
                 FitValue[:, 9*i], FitValue[:, 9*i+5], 'r',
                 FitValue[:, 9*i], FitValue[:, 9*i+7], 'b',)
    axes[1].set_title('abs part')
    figManager = plt.get_current_fig_manager()
    figManager.window.showMaximized()
    plt.subplots_adjust(left=0.045, right=0.991, top=0.951, bottom=0.062,
                        hspace=0.2, wspace=0.178)
    plt.show()
    plt.tight_layout()
    makedirs(fileID[0] + '\\Figures\\', exist_ok=True)

    plt.savefig(fileID[0] + '\\Figures\\figure{0}.png'.format(i))

FitParamHeader[0] = 'Name of the Sample'
FitParamHeader[1] = 'A'
FitParamHeader[2] = 'Error (A)'
FitParamHeader[3] = 'Tm'
FitParamHeader[4] = 'Error (Tm)'
FitParamHeader[5] = 'Chi square Monoexponential'
FitParamHeader[6] = 'A'
FitParamHeader[7] = 'Error (A)'
FitParamHeader[8] = 'Tm'
FitParamHeader[9] = 'Error (Tm)'
FitParamHeader[10] = 'x'
FitParamHeader[11] = 'Error (x)'
FitParamHeader[12] = 'Chi square Monoexponential with Variable exponent'
