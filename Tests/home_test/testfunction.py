# -*- coding: utf-8 -*-
"""
Created on Thu Jun 11 12:24:22 2020

Test files to check if datasmooth.py, automatic_phase_correction.py and apowin.py
are well working.

@author: Tim
"""
from eprload_BrukerBES3T import *
from eprload_BrukerESP import * 
from scipy.signal import find_peaks
import matplotlib.pyplot as plt
import numpy as np
#############################################################################
folder = 'C:\\Users\\Tim\\Jupyter\\CoreProject_EPRSuite\\Tests\\home_test\\'
#filename1 = '20190930_TC_20190826_NOSLY173F_1e_t=0s_T60K.par'
filename2 = '20200305_NOSLwT_EDNMR_1.DSC'
#filename3 = '20200311_NOSLwt_FSE_SRT_200us_100Shot_2.DSC'
#filename3b = '20200311_NOSLwt_FSE_SRT_200us_100Shot_2.DSC'

#y1,x1,par1 = eprload_BrukerESP(folder+filename1)
y2,x2,par2 = eprload_BrukerBES3T(folder+filename2)
#y3,x3,par3 = eprload_BrukerBES3T(folder+filename3)
#y3mod,x3mod,par3mod = eprload_BrukerBES3T(folder+filename3b)
#______________________________________________________________________________
# from datasmooth import *

# window_length = 13
# method1 = 'binom' 
# method2 ='flat'
# method3 = 'savgol'
# method4 = 'test'
# y_smooth12 = datasmooth(y1,window_length,method2)

# from automatic_phase import *
# from windowing import *
#______________________________________________________________________________
# from fieldmodulation import *
# ymodreal = fieldmodulation(x3,y3,3.5,2)
# from rescale import *
# ynew,scalefactor = rescale(np.real(y3),Mode='None')
# from basecorr import *
# ynew,scalefactor = basecorr2D(np.real(y3),Dimension = [1,1], Order = [1,3])
from ImportMultipleFiles import *
ListOfFiles = ImportMultipleNameFiles(folder, Extension='.DSC')
maxlen = MaxLengthOfFiles(ListOfFiles)
fulldata = OpenMultipleFiles(ListOfFiles,Scaling=None)