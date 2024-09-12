# -*- coding: utf-8 -*-
"""
Created on Thu Nov 16 11:46:11 2023
Analysis of Signal To Noise Ratio of a pulse echo detected before a 4PDEER 
pulse experiment recorded with a Bruker spectrometer.
Different parameters has to be taken account for:
    n = Number of scans
    h/a = Number of repetition in the loop. Normally a is used for the recorded
    transient echo and h is used for the 4PDeer experiement
    Phase cycling = Normally 16 steps
    pg = Pulse gate used for the integration
So the signal for a scan has to be deduced by dividing by a factor of:
    n*a*16
This script enables to import multiple Transient SNR measurements with files 
in DSC/DTA Bruker format.
Signal to noise ratio can be evaluated by two different method, and I'm still 
not sure which one is the better. The two methods implied different type of
raw data recording:
    1) For a 2D experiment, you record multiple transient of the echod and you
    divide the mean of the echo intensity by the standard deviation of the echo
    intensity.
    2) For a 1D experiment, you compare the echo intensity to the noise 
    (standard deviation) at the end of the transient.
     
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


def ComputeSNR_1D(y=None, pg1=[0, 1], pg2=[-2, -1]):
    '''
    Compute the SNR of a 1D-Data with pulse gate range pg1 (in datapoint) for
    averaging the intensity (Int) of the signal
    and pg2 for the determination of the standard deviation (Std) value.
    The SNR is calculated by the formula: Int/Std
    Script written by Timothée Chauviré 10/26/2023

    Parameters
    ----------
    y : experimental data
        TYPE : Numpy data array
    pg1 : pulse gate range for calculation of the intensity
        TYPE : [start,end] list
    pg2 : pulse gate range for calculation of the standard deviation
        TYPE : [start,end] list
    Returns
    -------
    SNR: Signal to noise ratio
        TYPE : real float value
    '''
    Int = np.mean(y[pg1[0]:pg1[1],], axis=0)
    Std = np.std(y[pg2[0]:pg2[1],], axis=0)
    return float(Int/Std)


def ComputeSNR_2D(y=None, pg1=[0, 1]):
    '''
    Compute the SNR of a 2D-Data with pulse gate range pg1 (in datapoint) for
    averaging the intensity (Int) of the signal.
    The determination of the standard deviation (Std) value, is made by 
    calculating the variation in intensity of Int Value.
    The SNR is calculated by the formula: Int/Std
    Script written by Timothée Chauviré 10/26/2023

    Parameters
    ----------
    y : 2D experimental data, column vector in axis 1 dimension
        TYPE : Numpy data array
    pg1 : pulse gate range for calculation of the intensity
        TYPE : [start,end] list
    Returns
    -------
    SNR: Signal to noise ratio
        TYPE : real float value
    '''
    IntMat = np.mean(y[pg1[0]:pg1[1]], axis=0)
    Int = np.mean(IntMat)
    StdMat = np.std(y[pg1[0]:pg1[1]], axis=0)
    Std = np.mean(StdMat)
    return float(Int/Std)


plt.close('all')
# File location in folder
folder = 'D:\\Documents\\Recherches\\ResearchAssociatePosition\\Data\\'\
    'PulsedEPR\\2024\\BorisSamples\\SNR_4PDEER\\'
# Importing global path to files
ListOfFiles = ImportMultipleNameFiles(FolderPath=folder, Extension='DSC')
# Check for the maximum length datafiles
npts = MaxLengthOfFiles(ListOfFiles)
nINTG = 78

# Initialize the variable:
Y = np.full((npts, 4*len(ListOfFiles)), np.nan, dtype="float")
Yname = list(np.zeros((4*len(ListOfFiles,))))
FitValue = np.full((len(ListOfFiles), 2), np.nan, dtype="float")
FitHeader = list(np.zeros((len(ListOfFiles),)))
# NameHeader = list(np.zeros((len(ListOfFiles,))))
# import all and achieve datatreatment
for i in range(len(ListOfFiles)):
    fileID = path.split(ListOfFiles[i])
    data, x, par = eprload(ListOfFiles[i])
    s = par['PlsSPELLISTSlct']  # Catch the phase cycling number
    phasecyc = float(''.join((ch if ch in '0123456789' else ' ') for ch in s))

    nscans = par['AVGS']*par['ShotsPLoop']*phasecyc
    npts = data.shape[0]
    newx = x.real.ravel()
    # Achieve the phase correction
    pivot = int(np.floor(data.shape[0]/2))
    new_data, _ = automatic_phase(
        vector=data, pivot1=pivot, funcmodel='minfunc')  # data are 180° phased
    yreal = new_data.real.ravel()  # Correction of the phase with the factor
    yimag = new_data.imag.ravel()  # exp(pi*j) = -1
    Y[0:npts, 4*i] = newx.ravel()
    Y[0:npts, 4*i+1] = yreal.ravel()-np.mean(yreal[-50:,], axis=0)
    Y[0:npts, 4*i+2] = yimag.ravel()-np.mean(yimag[-50:,], axis=0)
    Y[0:npts, 4*i+3] = np.abs(Y[0:npts, 4*i+1]+1j*Y[0:npts, 4*i+2]).ravel()
    Yname[4*i] = 'Time (ns)'
    Yname[4*i+1] = str(fileID[1]+'_real')
    Yname[4*i+2] = str(fileID[1]+'_imag')
    Yname[4*i+3] = str(fileID[1]+'_abs')
    if len(data.shape) > 1:
        ExpType = 2
        SNRValue = ComputeSNR_2D(Y[0:npts, 4*i+1], [0, nINTG])
        print(SNRValue)
        SNRValue2 = ComputeSNR_2D(Y[0:npts, 4*i+2], [0, nINTG])
    else:
        ExpType = 1
        SNRValue = ComputeSNR_1D(Y[0:npts, 4*i+1], [0, nINTG],
                                 [-nINTG, -1])
        print(SNRValue)
        SNRValue2 = ComputeSNR_1D(Y[0:npts, 4*i+2], [0, nINTG],
                                  [-nINTG, -1])
    FitValue[i, 0] = SNRValue
    FitValue[i, 1] = nscans
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
    ax = plt.gca()
    textstr = 'SNR value ({0}D_experiment) = {1} per scan\n'\
        'SNR value ({0}D_experiment) = {2}'.format(
            ExpType, round(SNRValue/np.sqrt(nscans), 4), round(SNRValue, 1))
    ax.text(0.3, 0.8, textstr, transform=ax.transAxes, fontsize=14,
            verticalalignment='top', bbox=dict(boxstyle="round", fc="none"))
    plt.show()
    plt.tight_layout()
    makedirs(fileID[0] + '\\Figures\\', exist_ok=True)
    plt.savefig(fileID[0] + '\\Figures\\figure{0}.png'.format(i))
    FitHeader[i] = fileID[1][:-4]
