# -*- coding: utf-8 -*-
"""
Created on Mon Oct 23 16:33:12 2023

@author: tim_t
"""
import numpy as np
import matplotlib.pyplot as plt
import deerlab as dl
from ImportMultipleFiles import eprload
from ImportDeerAnalysisParameters import BckgndSubtractionOfRealPart, DEERLABFitForOneFile
from copy import deepcopy

# File location
filename4 = 'C:\\Users\\tim_t\\Python\\EPRImportScripts\\DataFilesDEER\\'\
    '\\BrukerFiles\\04102019_Aer_EcAW_15N_2H_150K_4PDeer_4500ns.DSC'
filename1 = 'C:\\Users\\tim_t\\Python\\EPRImportScripts\\DataFilesDEER\\'\
    'TestFiles\\DEER_Anthr_50K_af1+42.DSC'
filename2 = 'C:\\Users\\tim_t\\Python\\EPRImportScripts\\DataFilesDEER\\'\
    'TestFiles\\DEER_Anthr_50K_af1-66.DSC'
filename3 = 'C:\\Users\\tim_t\\Python\\EPRImportScripts\\DataFilesDEER\\'\
    'TestFiles\\M_GdnHCl_DEER_2700ns.DSC'
DataDict, ParamDict = {}, {}
filename = filename2
DataDict, ParamDict = BckgndSubtractionOfRealPart(
    filename, DataDict, ParamDict, Scaling=None, Dimensionorder=3,
    Percent_tmax=9/10, mode='poly')
DataDict, ParamDict = DEERLABFitForOneFile(filename, DataDict, ParamDict)

y2, abscissa, par = eprload(filename, Scaling=None)
TITL = str(par['TITL'])
npts = abscissa.shape[0]
FullData = DataDict.get(TITL)
Header = DataDict.get(str(TITL+'_Header'))
Exp_parameter = ParamDict.get(TITL)

r = FullData[:, 6]
P1 = FullData[:, 7]
P1_OneGaussian = FullData[:, 10]
P1_TwoGaussians = FullData[:, 13]

P2 = FullData[:, 21]
P2_OneGaussian = FullData[:, 24]
P2_TwoGaussians = FullData[:, 27]

fig, axes = plt.subplots(2, 1)
fig.suptitle("Test reg_criterium selection")
# https://matplotlib.org/stable/api/axes_api.html
l1, = axes[0].plot(r, P1, 'k', label='Tikhonov_AIC', linewidth=1)
l2, = axes[0].plot(r, P1_OneGaussian, 'r',
                   label='One Gaussian fit', linewidth=1)
l3, = axes[0].plot(r, P1_TwoGaussians, 'b',
                   label='Two Gaussians fit', linewidth=1)
axes[0].grid()
axes[0].set_xlabel("Distance Domain [nm]")
axes[0].set_ylabel("P(r)_aic")
#axes[0].set_title('Criterium AIC')
axes[0].legend()

l1, = axes[1].plot(r, P2, 'k', label='Tikhonov_GCV', linewidth=1)
l2, = axes[1].plot(r, P2_OneGaussian, 'r',
                   label='One Gaussian fit', linewidth=1)
l3, = axes[1].plot(r, P2_TwoGaussians, 'b',
                   label='Two Gaussians fit', linewidth=1)
axes[1].grid()
axes[1].set_ylabel("P(r)_gcv")
axes[1].set_xlabel("Distance Domain [nm]")
#axes[1].set_title('Criterium GCV')

axes[1].legend()
plt.tight_layout()
