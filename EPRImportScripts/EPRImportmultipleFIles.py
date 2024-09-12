# -*- coding: utf-8 -*-
"""
Created on Mon Oct 16 14:09:16 2023

@author: tim_t
"""

# -*- coding: utf-8 -*-R
"""
Created on Thu Jun 11 12:24:22 2020

Test files to check if datasmooth.py, automatic_phase_correction.py and apowin.py
are well working.

@author: Tim
"""

import automatic_phase, basecorr1D, datasmooth, eprload_BrukerBES3T 
import eprload_BrukerESP, fdaxis, fieldmodulation, hilberttrans
import rcfilt, statistic_tools, windowing
from scipy.signal import find_peaks
import matplotlib.pyplot as plt
import numpy as np
#############################################################################
folder = 'D:\\Documents\\Recherches\\ResearchAssociatePosition\\PubliToWrite_Aerotaxis\\Jan30_Aer_EcAEcW_15N_D2O_BL21_Celtone\\DEER'
#filename1 = '20190930_TC_20190826_NOSLY173F_1e_t=0s_T60K.par'
# filename2 = '20200305_NOSLwT_EDNMR_1.DSC'
# filename3 = '20200212_20191210_NOSLN175E_Hyscore_20K.DSC'
# #filename4 ='190214_NosL_D5_291014_20K_03.DSC'
# filename4 ='290615_NosL_F_181014_20K_04.DSC'
#filename3b = '20200311_NOSLwt_FSE_SRT_200us_100Shot_2.DSC'

#y1,x1,par1 = eprload_BrukerESP(folder+filename1)
# y2,x2,par2 = eprload_BrukerBES3T(folder+filename2)
# y3,x3,par3 = eprload_BrukerBES3T(folder+filename3)
# y3.shape
# y4,x4,par4 = eprload_BrukerBES3T(folder2+filename4)
#y3mod,x3mod,par3mod = eprload_BrukerBES3T(folder+filename3b)
#______________________________________________________________________________
from datasmooth import *

# window_length = 13
# method1 = 'binom' 
# method2 ='flat'
# method3 = 'savgol'
# method4 = 'test'
# y_smooth12 = datasmooth(y1,window_length,method2)

from automatic_phase import *
# from windowing import *
#______________________________________________________________________________
# from fieldmodulation import *
# ymodreal = fieldmodulation(x3,y3,3.5,2)
# from rescale import *
# ynew,scalefactor = rescale(np.real(y3),Mode='None')
# from basecorr import *
# ynew,scalefactor = basecorr2D(np.real(y3),Dimension = [1,1], Order = [1,3])
from ImportMultipleFiles import *
from ImportMultipleFiles import OpenMultipleComplexFiles2

ListOfFiles = ImportMultipleNameFiles(folder, Extension='.DTA')
maxlen = MaxLengthOfFiles(ListOfFiles)
# fulldata3 = OpenComplexFiles(ListOfFiles,Scaling='n',polyorder=1,window_length=200)
data, abscissa, par = eprload(ListOfFiles[1],Scaling=None)
fulldata, Header = OpenMultipleComplexFiles2(ListOfFiles)
ncol = 7;
fulldata2 = np.full((maxlen,ncol),np.nan,dtype="complex_")
fulldata2[:,0]=fulldata[:,0] 
for i in range(ncol-1):
    fulldata2[:,i+1] = np.sum(fulldata[:,[i+1,ncol+1+i,ncol*2+i+1,ncol*4+i+1]],axis=1)
fulldata3

# Header = list(np.zeros((6*len(ListOfFiles),)))
# for i in range(len(ListOfFiles)):
#     Header[6*i] = par['XNAM']
#     Header[6*i+1] = ListOfFiles[i].split('\\')[-1]
#     Header[6*i+2] = ListOfFiles[i].split('\\')[-1]+str("_smooth_4pts")
#     Header[6*i+3] = ListOfFiles[i].split('\\')[-1]+str("_normalized")
#     Header[6*i+4] = ListOfFiles[i].split('\\')[-1]+str("_normalized")

#     Header[6*i+5] = ListOfFiles[i].split('\\')[-1]+str("_normalized")

# from basecorr2D import *
# from windowing import *

#ListOfFiles = ImportMultipleNameFiles(FolderPath = getcwd(), Extension = 'DTA')
#OpenMultipleComplexFiles(ListOfFiles,Scaling='maxabs')


# y3b = np.full((128,128),np.nan)
# y3c = np.full((128,128),np.nan)
# ynew3 = np.full((128,128),np.nan)

# for i in range(128):
#     y3b[i,:],_ = automatic_phase(y3[i,:],pivot1=127,funcmodel='acme')
# #     #y3c[:,i] = datasmooth(y=y3b.real[:,i],window_length=1,method='binom',polyorder=2)

# ynew3,basecorr = basecorr2D(Spectrum=np.real(y3b),dimension=[0,1],polyorder = [3,1])
# win = np.zeros((128*4,))
# win = windowing(window_type='gau+',M=128*4,alpha=0.6)
# y3d = np.zeros((128*4,128))
# y3e = np.full((128*4,128),np.nan)
# y3d[0:128,0:128] = ynew3[0:128,0:128]
# for i in range(128):
#     y3e[:,i] = win*y3d[:,i]
    
# fft = np.fft.fftshift(np.fft.fft2(y3e))
# fft2 = np.full(y3e.shape,np.nan,dtype='complex')
# fft2 = np.fft.fftshift(np.fft.rfft2(y3e))

# Npts1,Npts2 = y4.shape
# y4b = np.zeros((Npts1,Npts2))
# for i in range(128):
#     y4b[i,:],_ = automatic_phase(y4[i,:],pivot1=127,funcmodel='acme')
#     #y3c[:,i] = datasmooth(y=y3b.real[:,i],window_length=1,method='binom',polyorder=2)

# ynew4,basecorr4 = basecorr2D(Spectrum=np.real(y4b),dimension=[0,1],polyorder = [3,3])

# win = windowing(window_type='exp+',M=Npts1*4,alpha=3)
# y4d = np.zeros((Npts1*4,Npts2))
# y4e = np.full((Npts1*4,Npts2),np.nan)
# y4d[0:Npts1,0:Npts2] = ynew4[0:Npts1,0:Npts2]
# for i in range(Npts2):
#     y4e[:,i] = win*y4d[:,i]
    
# fft4 = np.fft.fftshift(np.fft.fft2(y4e))
# fft4b = np.full(y4e.shape,np.nan,dtype='complex')
# fft4b = np.fft.fftshift(np.fft.rfft2(y4e))