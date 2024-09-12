# -*- coding: utf-8 -*-
"""
Created on Tue Jan  2 12:10:59 2024

@author: tim_t
"""
from ImportMultipleFiles import ImportMultipleNameFiles, MaxLengthOfFiles, OpenMultipleComplexFiles2
import numpy as np
from fieldmodulation import fieldmodulation
from os import path
import matplotlib.pyplot as plt
folder1 = 'D:\\Documents\\Recherches\\ResearchAssociatePosition\\Data\\'\
    'PulsedEPR\\2024\\2024_Jess\\FSE\\80K'
ListOfFiles = ImportMultipleNameFiles(folder1, Extension='.DSC')
maxlen = MaxLengthOfFiles(ListOfFiles)
fulldata, header = OpenMultipleComplexFiles2(
    ListOfFiles, Scaling=None, polyorder=1, window=50)
# Get the pseudoderivative spectrum with the function fieldmodulation.py
# This script is freely inspired by the easyspin suite from the S.Stoll lab
# (https://github.com/StollLab/EasySpin/)
# (https://easyspin.org/easyspin/
plt.close('all')
modamp = 20  # amplitude modulation in Gauss
ncol = len(ListOfFiles)*2
FirstDeriv = np.full((maxlen, ncol), np.nan, dtype=float)
for i in range(len(ListOfFiles)):
    fileID = path.split(ListOfFiles[i])
    x = fulldata[:, 7*i]
    x = x[~np.isnan(x)]
    yreal = fulldata[:, 7*i+1]
    yreal = yreal[~np.isnan(yreal)]
    yimag = fulldata[:, 7*i+2]
    yimag = yimag[~np.isnan(yimag)]
    npts = x.shape[0]
    if np.trapz(yreal) < 0:
        yreal = -1*yreal
    firstderiv = fieldmodulation(
        x, yreal, ModAmpl=modamp, Harmonic=1)
    FirstDeriv[0:npts, 2*i] = x[0:npts]
    FirstDeriv[0:npts, 2*i+1] = firstderiv[0:npts]
    fig, axes = plt.subplots(2, 1)
    fig.suptitle(fileID[1], fontsize=20, weight='bold')
    axes[0].plot(x, yreal, 'k', label='Field-Swept Echo Real Part')
    axes[0].plot(x, yimag, 'r', label='Field-Swept Echo Imaginary Part')
    axes[0].legend(loc="upper left", fontsize='large')
    axes[1].plot(x, firstderiv, 'k',
                 label='First Derivative with {0}G pseudo-modulation'.format(str(modamp)))
    axes[1].legend(loc="lower left", fontsize='large')
    font = {'family': 'tahoma',
            'weight': 'normal',
            'size': 16}
    plt.rc('grid', linestyle="--", color='grey')
    plt.rc('lines', linewidth=2)
    plt.rc('font', **font)
    plt.rc('xtick', labelsize=16)
    plt.rc('ytick', labelsize=16)
    for j, ax in enumerate(axes):
        axes[j].set_xlabel('Magnetic Field [G]', fontsize=16, weight='bold')
        axes[j].set_ylabel('Intensity [a.u.]', fontsize=16, weight='bold')
        #axes[j].set_xlim(12120, 12220)
        axes[j].grid()
