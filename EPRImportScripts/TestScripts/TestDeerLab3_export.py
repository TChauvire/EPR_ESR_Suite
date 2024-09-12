# -*- coding: utf-8 -*-
"""
Created on Mon Oct 23 16:33:12 2023

@author: tim_t
"""
import matplotlib.pyplot as plt
import numpy as np
from copy import deepcopy
from os import path
from ImportDeerAnalysisParameters import BckgndSubtractionOfRealPart
from ImportDeerAnalysisParameters import DenoisingForOneFile
from ImportDeerAnalysisParameters import ExportOneAscii
from ImportDeerAnalysisParameters import ImportMultipleDEERAsciiFiles
from ImportDeerAnalysisParameters import DEERLABFitForOneFile
from ImportDeerAnalysisParameters import GetPakePatternForOneFile
from ImportMultipleFiles import datasmooth
from automatic_phase import automatic_phase
from basecorr1D import basecorr1D
from windowing import windowing
from fdaxis import fdaxis
import deerlab as dl
plt.close('all')
# File location in folder
folder = 'C:\\Users\\tim_t\\Python\\EPRImportScripts\\DataFilesDEER'
# Importing global path to files
ListOfFiles = ImportMultipleDEERAsciiFiles(FolderPath=folder, Extension='DSC')
# Initialize dictionnaries
DataDict, ParamDict = {}, {}
DataDict2, ParamDict2 = {}, {}
for j in range(len(ListOfFiles)):
    print(j)
    filename = ListOfFiles[j]
    fileID = path.split(filename)
    # Import data and achieve baseline subtraction
    DataDict, ParamDict = BckgndSubtractionOfRealPart(
        filename, DataDict, ParamDict, Dimensionorder=1, Percent_tmax=9/10,
        mode='poly', truncsize=1/2)
    DataDict, ParamDict = GetPakePatternForOneFile(
        filename, DataDict, ParamDict)
    DataDict, ParamDict = DenoisingForOneFile(filename, DataDict, ParamDict, Dimensionorder=1,
                                              Percent_tmax=9/10, mode='poly',
                                              truncsize=1/2)
    DataDict, ParamDict = DEERLABFitForOneFile(filename, DataDict, ParamDict)

plt.close('all')
for j in range(len(ListOfFiles)):
    print(j)

    filename = ListOfFiles[j]
    fileID = path.split(filename)
    # FullData = DataDict.get(fileID[1])
    # r = FullData[:, 9]
    # P1 = FullData[:, 10]
    # P1_OneGaussian = FullData[:, 13]
    # P1_TwoGaussians = FullData[:, 16]

    # P2 = FullData[:, 24]
    # P2_OneGaussian = FullData[:, 27]
    # P2_TwoGaussians = FullData[:, 30]
    # fig, axes = plt.subplots(2, 1)
    # fig.suptitle(fileID[1])
    # # https://matplotlib.org/stable/api/axes_api.html
    # l1, = axes[0].plot(r, P1, 'k', label='Tikhonov_AIC', linewidth=1)
    # l2, = axes[0].plot(r, P1_OneGaussian, 'r',
    #                    label='One Gaussian fit', linewidth=1)
    # l3, = axes[0].plot(r, P1_TwoGaussians, 'b',
    #                    label='Two Gaussians fit', linewidth=1)
    # axes[0].grid()
    # axes[0].set_xlabel("Distance Domain [nm]")
    # axes[0].set_ylabel("P(r)_aic")
    # #axes[0].set_title('Criterium AIC')
    # axes[0].legend()

    # l1, = axes[1].plot(r, P2, 'k', label='Tikhonov_GCV', linewidth=1)
    # l2, = axes[1].plot(r, P2_OneGaussian, 'r',
    #                    label='One Gaussian fit', linewidth=1)
    # l3, = axes[1].plot(r, P2_TwoGaussians, 'b',
    #                    label='Two Gaussians fit', linewidth=1)
    # axes[1].grid()
    # axes[1].set_ylabel("P(r)_gcv")
    # axes[1].set_xlabel("Distance Domain [nm]")
    # #axes[1].set_title('Criterium GCV')

    Exp_parameter = ParamDict.get(fileID[1])
    itmax = Exp_parameter.get('itmax')
    itmin = Exp_parameter.get('itmin')
    plt.figure(j)
    fig, axes = plt.subplots(1, 3)
    fig.suptitle(fileID[1])

    FullData = DataDict.get(fileID[1])
    t = FullData[:, 0]
    t = t[~np.isnan(t)]
    y = FullData[:, 5]
    y = y[~np.isnan(y)]
    ymax = np.max(y)
    tmax = t[itmax]
    yfit = FullData[:, 37].ravel()
    yfit = yfit[~np.isnan(yfit)]
    yfit = yfit-1+ymax
    npts = yfit.shape[0]
    newt = np.linspace(0, tmax*5, 5*npts)/1000  # Time axis in us
    f2 = fdaxis(TimeAxis=newt)  # Frequency axis in MHz
    win = windowing(window_type='exp+', N=npts, alpha=3)
    yfit2 = np.zeros((npts*5,), dtype="complex_")  # zerofilling
    yfit2[0:npts,] = yfit[0:npts,]*win[0:npts,]
    PakeSpectra = np.fft.fftshift(np.fft.fft(yfit2))
    Pake2, _, _, _ = basecorr1D(x=f2, y=np.abs(PakeSpectra),
                                polyorder=1, window=50)
    Pake2 = np.abs(Pake2)/np.max(Pake2)
    f2pts = f2.shape[0]
    f2pts_init = int(np.floor(1*f2pts/4))
    f2pts_max = int(np.floor(3*f2pts/4))

    f = FullData[:, 6]
    f = f[~np.isnan(f)]
    Pake = np.abs(FullData[:, 8])
    Pake = Pake[~np.isnan(Pake)]
    Pake = Pake/np.max(Pake)
    fpts = f.shape[0]
    fpts_init = int(np.floor(1*fpts/4))
    fpts_max = int(np.floor(3*fpts/4))
    #Pake = Pake[~np.isnan(Pake)]
    r = FullData[:, 9]
    #r = r[~np.isnan(r)]
    P1 = FullData[:, 10]
    #P1 = P1[~np.isnan(P1)]
    P1_OneGaussian = FullData[:, 13]
    P1_TwoGaussians = FullData[:, 16]
    axes[0].grid()
    axes[0].set_xlabel("Time Domain [ns]")
    axes[0].set_ylabel("S(t)")
    l1, = axes[0].plot(t[:itmax,], y[:itmax,], 'k',
                       label='signal', linewidth=1)
    l2, = axes[0].plot(t[itmin:itmax,], yfit[:,], 'r', label='TikhonovSignal',
                       linewidth=1)
    axes[0].legend()
    axes[1].grid()
    axes[1].set_xlabel("Distance Domain [nm]")
    axes[1].set_ylabel("P(r)_aic")
    l1, = axes[1].plot(r, P1, 'k', label='Tikhonov', linewidth=1)
    # l2, = axes[1].plot(r, P2, 'k', label='SVD', linewidth=1)
    l2, = axes[1].plot(r, P1_OneGaussian, 'r',
                       label='One Gaussian fit', linewidth=1)
    l3, = axes[1].plot(r, P1_TwoGaussians, 'b',
                       label='Two Gaussians fit', linewidth=1)
    axes[1].legend()
    axes[2].grid()
    axes[2].set_xlabel("Frequency Domain [MHz]")
    axes[2].set_ylabel("Pake(f)")
    l1, = axes[2].plot(f[fpts_init:fpts_max,],
                       Pake[fpts_init:fpts_max,], 'k', label='Signal', linewidth=1)
    l2, = axes[2].plot(f2[fpts_init:fpts_max,], Pake2[fpts_init:fpts_max,],
                       'r', label='TikhonovSignal', linewidth=1)
    axes[2].legend()

    figManager = plt.get_current_fig_manager()
    figManager.window.showMaximized()
    plt.subplots_adjust(left=0.045, right=0.991, top=0.951, bottom=0.062,
                        hspace=0.2, wspace=0.178)
    plt.show()
    plt.tight_layout()
