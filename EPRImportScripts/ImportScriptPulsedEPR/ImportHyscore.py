# -*- coding: utf-8 -*-
"""
Created on Thu Nov 16 11:46:11 2023
Analysis of T1 data

This script enables to import Hyscore data in DSC/DTA Bruker format.

@author: Timothee Chauvire tsc84@cornell.edu
"""
import matplotlib.pyplot as plt
import numpy as np
from ImportMultipleFiles import eprload, datasmooth
from basecorr2D import basecorr2D
# from automatic_phase import automatic_phase
from os import path
# from mpl_toolkits.mplot3d import Axes3D
from windowing import windowing
from Plot3DUtilities import mayavi3D, Contour3D
# from datasmooth import datasmooth
from mayavi import mlab
from tvtk.api import tvtk
from fdaxis import fdaxis
plt.close('all')
mlab.close(all=True)
# File location in folder
# folder = 'C:\\Users\\tim_t\\Python\\EPRImportScripts\\DataFilesRelaxationTime'\
#          '\\T1\\T1LogScale'  # Importing global path to files
# FullFileName = 'D:\\Documents\\Recherches\\ResearchAssociatePosition\\Data\\'\
#     'PulsedEPR\\2024\\2024_Jess\\20240208\\Hyscore\\'\
#     '20240208_100uM_WTCheA_pH8.5_1to1_CuNTA_Hyscore_20K.DSC'
# FullFileName = 'D:\\Documents\\Recherches\\ResearchAssociatePosition\\Data\\'\
#     'PulsedEPR\\2024\\2024_Jess\\20240201\\20K_CheA_7_5\\Hyscore\\'\
#     '20240131_wtCheA_pH7_5_150uM_Hyscore_7dB_20K.DSC'
FullFileName = 'D:\\Documents\\Recherches\\ResearchAssociatePosition\\Data\\'\
    'PulsedEPR\\2024\\2024_Jess\\20240202\\CuHis2_20K\\Hyscore\\'\
    '20240202_CuHis2_500uM_Hyscore3380G_7dB_20K.DSC'
# import the hyscore data and achieve data phasing (automatic_phase) and
# baseline treatment with 2D polyomials baseline correction with basecorr2D.py
fileID = path.split(FullFileName)
data, x, par = eprload(FileName=FullFileName)
npts = data.shape[0]
npts2 = data.shape[1]

t1 = x[:, 0].real/1000  # np.arange(0, 256, 1)
t2 = x[:, 1].real/1000  # np.arange(0, 256, 1)
z = data.real
mayavi3D(x1=t1, x2=t2, z=z, mode='time', name=fileID[1])
Contour3D(x1=t1, x2=t2, z=z, level=1, mode='time', name=fileID[1])

# Correct the phase of time domain data
new_data = np.full(data.shape, np.nan, dtype="complex")
new_data2 = np.full(data.shape, np.nan, dtype="float")
# for i in range(data.shape[1]):
#     pivot = int(np.floor(data.shape[0]/2))
#     # new_data[:, i], _ = automatic_phase(vector=data[:, i], pivot1=pivot,
#    #                                     funcmodel='minfunc')
#     new_data[:, i] = datasmooth(
#         data[:, i].real, window_length=1, method='flat')

# Achieve the baseline correction
CorrSpectrum, Baseline = basecorr2D(data.real, [1], [2])
CorrSpectrum, Baseline = basecorr2D(CorrSpectrum, [2], [0])

mayavi3D(x1=t1, x2=t2, z=CorrSpectrum,
         mode='time', name='Bckgnd corrected data')
mayavi3D(x1=t1, x2=t2, z=Baseline, mode='time', name='Baseline correction')
# Carry out windowing
win = np.full(data.shape, np.nan, dtype=float)
# for i in range(data.shape[1]):
#     CorrSpectrum[:, i] = CorrSpectrum[:, i]-np.mean(CorrSpectrum[-20:,])
win1D_a = windowing(window_type='ham+', N=npts)
win1D_b = windowing(window_type='ham+', N=npts2)
win2D = np.sqrt(np.outer(win1D_a, win1D_b))  # 2D windowing
CorrSpectrum2 = CorrSpectrum*win2D
mayavi3D(x1=t1, x2=t2, z=win2D, mode='time', name='2D apodization window')
mayavi3D(x1=t1, x2=t2, z=CorrSpectrum2, mode='time', name='Apodized data')

# Apply zeroffilling
CorrSpectrum3 = np.zeros((npts*5, npts2*5), dtype=float)
CorrSpectrum3[0:npts, 0:npts2] = CorrSpectrum3[0:npts, 0:npts2] + CorrSpectrum2
t1_b = np.linspace(t1[0], t1[-1]*5, npts*5)
t2_b = np.linspace(t2[0], t2[-1]*5, npts2*5)
mayavi3D(x1=t1_b, x2=t2_b, z=CorrSpectrum3,
         mode='time', name='zerofilled data')

# Generate the frequency scale:
f1 = fdaxis(TimeAxis=t1_b)
f2 = fdaxis(TimeAxis=t2_b)
# Fourier Transform the data
data_fft = np.fft.fftshift(np.fft.fft2(CorrSpectrum3))
fftabs = np.absolute(data_fft)  # /np.max(np.absolute(data_fft))
mayavi3D(x1=f1, x2=f2, z=fftabs, mode='freq', name=fileID[1])
# Multiplicative level number to change to get the white background on the
# flat area
Contour3D(x1=f1, x2=f2, z=fftabs, level=2.1, mode='freq', name=fileID[1])
