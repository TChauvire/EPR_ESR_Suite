# -*- coding: utf-8 -*-
"""
Created on Mon Oct 30 12:33:55 2023

@author: tim_t
"""
import matplotlib.pyplot as plt
import numpy as np
from ImportMultipleFiles import eprload
from ImportDeerAnalysisParameters import BckgndSubtractionOfRealPart
from SVD_scripts import get_KUsV, process_data, rminrmaxcalculus
import deerlab as dl

# import matplotlib.cm as cm

filename1 = 'C:\\Users\\tim_t\\Python\\EPRImportScripts\\DataFilesDEER\\'\
    'TestFiles\\DEER_Anthr_50K_af1+42.DSC'
filename2 = 'C:\\Users\\tim_t\\Python\\EPRImportScripts\\DataFilesDEER\\'\
    '\\BrukerFiles\\04102019_Aer_EcAW_15N_2H_150K_4PDeer_4500ns.DSC'
DataDict, ParamDict = {}, {}
filename = filename1
DataDict, ParamDict = BckgndSubtractionOfRealPart(
    filename1, DataDict, ParamDict)
y2, abscissa, par = eprload(filename1, Scaling=None)
TITL = str(par['TITL'])
npts = abscissa.shape[0]
y = DataDict.get(str(par['TITL']))[0:npts, 5].ravel()

Exp_parameter = ParamDict.get(par['TITL'])
tmin = Exp_parameter.get('zerotime')
tmax = Exp_parameter.get('tmax')
itmin = Exp_parameter.get('itmin')
itmax = Exp_parameter.get('itmax')
tmax = abscissa[itmax, 0]/1000
newy = np.real(y[itmin:itmax,])
new_npts = newy.shape[0]
t = ((abscissa[itmin:itmax,]-abscissa[itmin])/1000).ravel()
r = dl.distancerange(t, nr=new_npts)
K, U, s, V = get_KUsV(t)
rrange = rminrmaxcalculus(t)
rmin = rrange[0]
rmax = rrange[1]
S, sigma, PR, Pr, Picard, sum_Pic = process_data(t, newy, K,
                                                 U, s, V)

fig, axes = plt.subplots(2, 2)
fig.suptitle("Test SVD Reconstruction")
# https://matplotlib.org/stable/api/axes_api.html
axes[0, 0].plot(t, newy)
axes[0, 0].set_xlabel("Time [us]")
axes[0, 0].grid()
axes[0, 0].set_ylabel("Intensity (a.u.)")
axes[0, 0].set_title('Time Domain')
# axes[0].legend()

axes[0, 1].plot(sigma)
axes[0, 1].set_yscale('log')
axes[0, 1].grid()
axes[0, 1].set_ylabel("Singular Values")
axes[0, 1].set_xlabel("SVC number")
axes[0, 1].set_title('Singular Values')

axes[1, 0].plot(r, Pr.T[:, 40])
axes[1, 0].grid()
axes[1, 0].set_ylabel("P(r)")
axes[1, 0].set_xlabel("Distance r [nm]")
axes[1, 0].set_title('Distance Domain')

axes[1, 1].plot(sum_Pic)
axes[1, 1].set_yscale('log')
axes[1, 1].grid()
axes[1, 1].set_ylabel("Picard_summed")
axes[1, 1].set_xlabel("SVC number")
axes[1, 1].set_title('Picard plot')

plt.tight_layout()
