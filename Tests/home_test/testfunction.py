# -*- coding: utf-8 -*-
"""
Created on Thu Jun 11 12:24:22 2020

Test files to check if datasmooth.py, automatic_phase_correction.py and apowin.py
are well working.

@author: Tim
"""
from eprload_BrukerBES3T import *
from eprload_BrukerESP import * 
import matplotlib.pyplot as plt

folder = 'C:\\Users\\Tim\\Jupyter\\CoreProject_EPRSuite\\Tests\\home_test\\'
filename1 = '20190930_TC_20190826_NOSLY173F_1e_t=0s_T60K.par'
filename2 = '20200305_NOSLwT_EDNMR_1.DSC'

y1,abscissa1,par1 = eprload_BrukerESP(folder+filename1)
y2,abscissa2,par2 = eprload_BrukerBES3T(folder+filename2)

#import numpy as np
#import scipy.optimize
#First test datasmooth
#______________________________________________________________________________
from datasmooth import *

window_length = 13
method1 = 'binom' 
method2 ='flat'
method3 = 'savgol'
method4 = 'test'
y_smooth12 = datasmooth(y1,window_length,method2)

from automatic_phase import *
from windowing import *