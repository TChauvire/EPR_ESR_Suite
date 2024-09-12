# -*- coding: utf-8 -*-
"""
Created on Fri Nov 17 17:48:00 2023

@author: tim_t
"""

import numpy as np
import deerlab as dl
from basecorr1D import basecorr1D
from ImportDeerAnalysisParameters import importasciiSVDData, GaussianFit
from ImportDeerAnalysisParameters import DEERLABFitForMultipleFIles
from ImportDeerAnalysisParameters import importasciiTimeData
from ImportMultipleFiles import ImportMultipleNameFiles
from ImportDeerAnalysisParameters import GetPakePattern
import matplotlib.pyplot as plt
from scipy.signal import find_peaks, peak_widths
from os import path

folder = 'C:\\Users\\tim_t\\Python\\EPRImportScripts\\DataFilesDEER\\' \
    'AsciiSVDFiles\\poly1\\Pr'
Listoffilename = ImportMultipleNameFiles(folder, 'txt')

folder2 = 'C:\\Users\\tim_t\\Python\\EPRImportScripts\\DataFilesDEER\\' \
    'AsciiSVDFiles\\poly1\\ReconstructedSignal'
Listoffilename2 = ImportMultipleNameFiles(folder2, 'txt')

folder3 = 'C:\\Users\\tim_t\\Python\\EPRImportScripts\\DataFilesDEER\\' \
    'BrukerFiles'
Listoffilename3 = ImportMultipleNameFiles(folder3, 'DSC')
DataDict, ParamDict, DataDict2, ParamDict2 = {}, {}, {}, {}
DataDict2, ParamDict2 = DEERLABFitForMultipleFIles(Listoffilename3, DataDict2,
                                                   ParamDict2)
plt.close('all')
for i in range(len(Listoffilename)):
    filename = Listoffilename[i]
    fileID = path.split(filename)

    # Get Time Domain data and Time domain reconstructed Data
    TITL1 = str(fileID[0][:-2]+'ReconstructedSignal\\Reconstructed_'
                + fileID[1][3:])
    y1, yfit1, t1, _ = importasciiTimeData(TITL1)
    t1 = (t1-t1[0])*1000  # Time domain in nanoseconds
    y1norm = y1/np.max(y1)
    yfit1_norm = yfit1/np.max(yfit1)
    # Get Pake pattern Data
    fft1, fft1_abs, freq1 = GetPakePattern(t1, y1norm)
    fft1_fit, fft1_abs_fit, freq1_fit = GetPakePattern(t1, yfit1_norm)
    fpts1 = freq1.shape[0]
    fpts1_init = int(np.floor(1*fpts1/8))
    fpts1_max = int(np.floor(7*fpts1/8))
    max1 = np.max(fft1_abs[fpts1_init:fpts1_max].real)
    Pake1 = fft1_abs[fpts1_init:fpts1_max].real/max1
    fpts2 = freq1_fit.shape[0]
    fpts2_init = int(np.floor(1*fpts2/8))
    fpts2_max = int(np.floor(7*fpts2/8))
    max2 = np.max(fft1_abs_fit[fpts1_init:fpts2_max].real)
    Pake2 = fft1_abs_fit[fpts2_init:fpts2_max].real/max2

    # Get Tikhonov Time Domain reconstructed spectra:
    TITL2 = fileID[1][3:-4]
    Exp_parameter2 = ParamDict2.get(TITL2)
    itmax2 = Exp_parameter2.get('itmax')
    itmin2 = Exp_parameter2.get('itmin')
    FullData2 = DataDict2.get(TITL2)
    t2 = FullData2[:, 0].ravel().real
    t2 = t2[~np.isnan(t2)]
    t2 = t2[itmin2:itmax2]
    y2 = FullData2[:, 5].ravel().real
    y2 = y2[~np.isnan(y2)]
    ymax2 = np.max(y2)
    tmax2 = t2[-1]
    yfit2 = FullData2[:, 37].ravel().real
    yfit2 = yfit2[~np.isnan(yfit2)]
    yfit2 = yfit2-1+ymax2
    y2norm = yfit2/np.max(yfit2)

    # Get Pake pattern Data
    fft2_fit, fft2_abs_fit, freq2_fit = GetPakePattern(t2, y2norm)
    fpts3 = freq2_fit.shape[0]
    fpts3_init = int(np.floor(1*fpts3/8))
    fpts3_max = int(np.floor(7*fpts3/8))
    max3 = np.max(fft2_abs_fit[fpts3_init:fpts3_max].real)
    Pake3 = fft2_abs_fit[fpts3_init:fpts3_max].real/max3

    # Tikhonov Distance Domain data
    npts = yfit2.shape[0]
    r2 = FullData2[:, 9].real
    r2 = r2[~np.isnan(r2)]
    P2 = FullData2[:, 10].real
    P2 = P2[~np.isnan(P2)]
    P2 = np.ravel(P2)/np.max(P2)
    i_pk2 = find_peaks(P2, 0.01)
    # fitted_data2, popt2, perr2, residual2, RMSE_Gauss2 = GaussianFit(
    #     x=r2, y=P2, NumberOfGaussian=len(i_pk2[0]), height=0.01)

    # Get SVD Distance Domain spectra:
    P1, r1, header = importasciiSVDData(Listoffilename[i])
    P1 = np.ravel(P1)/np.max(P1)

    # i_pk = find_peaks(P1, 0.01)
    # fitted_data1, popt1, perr1, residual1, RMSE_Gauss1 = GaussianFit(x=r1,
    #                                                                  y=P1, NumberOfGaussian=len(i_pk[0]), height=0.01)
    # Plot the data : Time Domain // Distance Domain // Pake Pattern
    plt.figure(i)
    fig, axes = plt.subplots(1, 3)
    fig.suptitle(fileID[1])
    #axes[0].plot(t1, y1, 'k', t1, yfit1, 'r', t2, yfit2, 'b')
    l1, = axes[0].plot(t1, y1, 'k', label='signal', linewidth=1)
    l2, = axes[0].plot(t1, yfit1, 'r', label='SVDReconstructedSignal',
                       linewidth=1)
    l3, = axes[0].plot(t2, yfit2, 'b', label='TikhonovReconstructedSignal',
                       linewidth=1)
    axes[0].set_xlabel("Time Domain [ns]")
    axes[0].set_ylabel("S(t)")
    axes[0].legend(loc='upper right')
    #axes[1].plot(r1, P1, 'r', r2, P2, 'b')
    l4, = axes[1].plot(r1, P1, 'r', label='SVDRegularization',
                       linewidth=1)
    l5, = axes[1].plot(r2, P2, 'b', label='TikhonovRegularization',
                       linewidth=1)
    axes[1].set_xlabel("Distance Domain [nm]")
    axes[1].set_ylabel("P(r)")
    axes[1].legend(loc='upper right')
    # axes[2].plot(freq1[fpts1_init:fpts1_max], fft1[fpts1_init:fpts1_max].real, 'k',
    #              freq1_fit[fpts2_init:fpts2_max], fft1_fit[fpts2_init:fpts2_max].real, 'r',
    #              freq2_fit[fpts3_init:fpts3_max], fft2_fit[fpts3_init:fpts3_max].real, 'b')
    l6, = axes[2].plot(freq1[fpts1_init:fpts1_max], Pake1,
                       'k', label='Pake Pattern', linewidth=1)
    l7, = axes[2].plot(freq1_fit[fpts2_init:fpts2_max], Pake2,
                       'r', label='SVDRegularization', linewidth=1)
    l8, = axes[2].plot(freq2_fit[fpts3_init:fpts3_max], Pake3,
                       'b', label='TikhonovRegularization', linewidth=1)
    axes[2].set_xlabel("Frequency Domain [MHz]")
    axes[2].set_ylabel("FFT amplitude (a.u.)")
    axes[2].legend(loc='upper right')
    figManager = plt.get_current_fig_manager()
    figManager.window.showMaximized()
    plt.subplots_adjust(left=0.057, right=0.991, top=0.948, bottom=0.062,
                        hspace=0.2, wspace=0.18)
    plt.show()
    # plt.tight_layout()
