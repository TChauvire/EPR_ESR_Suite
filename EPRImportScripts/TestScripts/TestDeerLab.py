# -*- coding: utf-8 -*-
"""
Created on Mon Oct 23 16:33:12 2023

@author: tim_t
"""
import numpy as np
import matplotlib.pyplot as plt
import deerlab as dl
from ImportMultipleFiles import eprload
from ImportDeerAnalysisParameters import *
from copy import deepcopy

# File location
filename1 = 'C:\\Users\\tim_t\\Python\\EPRImportScripts\\DataFilesDEER\\'\
    'TestFiles\\DEER_Anthr_50K_af1+42.DSC'
filename2 = 'C:\\Users\\tim_t\\Python\\EPRImportScripts\\DataFilesDEER\\'\
    'TestFiles\\DEER_Anthr_50K_af1-66.DSC'
filename3 = 'C:\\Users\\tim_t\\Python\\EPRImportScripts\\DataFilesDEER\\'\
    'TestFiles\\M_GdnHCl_DEER_2700ns.DSC'
filename4 = 'C:\\Users\\tim_t\\Python\\EPRImportScripts\\DataFilesDEER\\'\
    '\\BrukerFiles\\04102019_Aer_EcAW_15N_2H_150K_4PDeer_4500ns.DSC'
filename5 = 'C:\\Users\\tim_t\\Python\\EPRImportScripts\\DataFilesDEER\\'\
    'BrukerFiles\\051019_AerECAW_N15_4PDEER_3000ns_150K.DSC'
filename6 = 'C:\\Users\\tim_t\\Python\\EPRImportScripts\\DataFilesDEER\\'\
    'BrukerFiles\\031419_Aer_EcCheA_EcCheW_BL21_150K_DEER_2500ns.DSC'
filename7 = 'C:\\Users\\tim_t\\Python\\EPRImportScripts\\DataFilesDEER\\'\
    'BrukerFiles\\041019_Aer_EcAW_15N_2H_150K_4PDeer_6us_Sum.DSC'
filename8 = 'C:\\Users\\tim_t\\Python\\EPRImportScripts\\DataFilesDEER\\'\
    'BrukerFiles\\022421_AerAW15N2H_B21_150K_DEE_4000ns_42MHz.DSC'
filename9 = 'C:\\Users\\tim_t\\Python\\EPRImportScripts\\DataFilesDEER\\'\
    'BrukerFiles\\022421_Aer_AW_15N2H_BL21_150K_DEER_5000ns_LF60.DSC'
filename10 = 'C:\\Users\\tim_t\\Python\\EPRImportScripts\\DataFilesDEER\\'\
    'BrukerFiles\\012221_Tar_iLAW_BL21_5sirrad_150K_DEER_1600ns.DSC'
filename11 = 'C:\\Users\\tim_t\\Python\\EPRImportScripts\\DataFilesDEER\\'\
    'BrukerFiles\\012021_iLAW_TarA160CMemb_4mMAsp_170K_DEER1p6us.DSC'
filename12 = 'C:\\Users\\tim_t\\Python\\EPRImportScripts\\DataFilesDEER\\'\
    'BrukerFiles\\062423_iLAWpACYC_TarpET28a_BL21_150K_DEE_1600ns_2029scans.DSC'
DataDict, ParamDict = {}, {}
filename = filename1
DataDict, ParamDict = BckgndSubtractionOfRealPart(
    filename, DataDict, ParamDict, Scaling=None, Dimensionorder=5,
    Percent_tmax=9/10, mode='poly')
DataDict, ParamDict = GetPakePatternForOneFile(filename, DataDict, ParamDict)
y2, abscissa, par = eprload(filename, Scaling=None)
TITL = str(par['TITL'])
Exp_parameter = ParamDict.get(par['TITL'])
tmax = Exp_parameter.get('tmax')
itmin = Exp_parameter.get('itmin')
itmax = Exp_parameter.get('itmax')
npts = abscissa.shape[0]
y = DataDict.get(str(par['TITL']))[0:npts, 5].ravel()
t = DataDict.get(str(par['TITL']))[0:npts, 0].ravel()/1000
newt = t[itmin:itmax,]
newy = y[itmin:itmax,]
freq = DataDict.get(str(par['TITL']))[:, 6].ravel()
freq = freq[~np.isnan(freq)]
Pake = DataDict.get(str(par['TITL']))[:, 7].ravel()
Pake = Pake[~np.isnan(Pake)]
PakeAbs = DataDict.get(str(par['TITL']))[:, 8].ravel()
PakeAbs = PakeAbs[~np.isnan(PakeAbs)]


nu, fft_dl = dl.fftspec(
    y, t, mode='abs', zerofilling=5*npts, apodization=True)
# Exp_parameter = ParamDict.get(par['TITL'])
# tmin = Exp_parameter.get('zerotime')
# tmax = Exp_parameter.get('tmax')
# itmin = Exp_parameter.get('itmin')
# itmax = Exp_parameter.get('itmax')

# y3 = dl.correctphase(y2.ravel())
# newy3 = np.real(y3[itmin:itmax,]).ravel()
# Vexp1 = newy3/np.max(newy3)

# tmax = abscissa[itmax, 0]/1000
# newy = np.real(y[itmin:itmax,])
# Vexp2 = np.real(newy-np.max(newy)+1)
# Vexp2 = Vexp2/np.max(Vexp2)

# new_npts = newy.shape[0]
# # Experimental parameters (get the parameters from the Bruker DSC format)
# BrukerParamDict = GetExpParamValues(filename)
# tau1 = BrukerParamDict['d1']/1000  # First inter-pulse delay, us
# tau2 = BrukerParamDict['d2']/1000

# t = (abscissa[itmin:itmax,]/1000 + tau1).ravel()

# Pre-processing
# Vexp = np.real(newy-np.max(newy)+1)    # Rescaling (aesthetic)
# Vexp = Vexp/np.max(Vexp)
# Vexp = np.real(newy)
# Distance vector
# r = np.arange(1, 6, 0.025)  # nm
# r = dl.distancerange(t, nr=200)
# Pmodel = dl.dd_gauss
# Pmodel.mean.set(lb=min(r), ub=max(r), par0=4.0)
# Pmodel.std.set(lb=0.01, ub=0.3, par0=0.1)

# Bmodel = dl.bg_strexp
# Bmodel.decay.set(lb=0, ub=1e6, par0=2.0)
# Bmodel.stretch.set(lb=2.99, ub=3.01, par0=3)
# # Construct the model
# Vmodel_P = dl.dipolarmodel(t, r, Pmodel=Pmodel, Bmodel=None,
#                            experiment=dl.ex_4pdeer(tau1, tau2, pathways=[1]))
# Vmodel = dl.dipolarmodel(t, r, Pmodel=None, Bmodel=None,
#                          experiment=dl.ex_4pdeer(tau1, tau2, pathways=[1]))
# Vmodel_B = dl.dipolarmodel(t, r, Pmodel=None, Bmodel=Bmodel,
#                            experiment=dl.ex_4pdeer(tau1, tau2, pathways=[1]))
# #compactness = dl.dipolarpenalty(Pmodel=None, r=r, type='compactness')

# # Fit the model to the data
# results_Bmodel = dl.fit(Vmodel_B, Vexp1, penalties=None, bootstrap=0, noiselvl=None,
#                         mask=None, weights=None, regparam='aic', reg=True)
# results = dl.fit(Vmodel, Vexp2, penalties=None, bootstrap=0, noiselvl=None,
#                  mask=None, weights=None, regparam='aic', reg=True)


# # results_P2 = dl.fit(Vmodel_P, Vexp, regparam='gcv', reg=True)

# # Extract distance domain data
# Pfit2 = results.P
# # P_scale = results.P_scale
# scale2 = np.trapz(Pfit2, r)  # *= 100  #
# # P = P  # *P_scale
# Puq_fit2 = results.PUncert
# Pci95_fit2 = Puq_fit2.ci(95)  # *P_scale

# # P2 = results_P2.P
# # Puq2 = results_P2.PUncert
# # Pci95_2 = Puq2.ci(95)
# # Print results summary
# Pfit1 = results_Bmodel.P
# Puq_fit1 = results_Bmodel.PUncert
# Pci95_fit1 = Puq_fit1.ci(95)

# #Pfit = results_Bmodel.evaluate(Pmodel, r)
# # scale = np.trapz(Pfit, r)
# #Puncert = results_Pmodel.propagate(Pmodel, r, lb=np.zeros_like(r))
# # Pfit = Pfit  # /scale
# # Pci95 = Puncert.ci(95)  # /scale
# # Pci50 = Puncert.ci(50)  # /scale

# Vfit1 = results_Bmodel.model
# Vfit2 = results.model
# plt.figure(1)
# plt.plot(abscissa[itmin:itmax], Vexp1, 'k', abscissa[itmin:itmax], Vfit1, 'k-',
#          abscissa[itmin:itmax], Vexp2, 'r', abscissa[itmin:itmax], Vfit2, 'r-')
# plt.xlabel('Time (ns)')
# plt.ylabel('Normalized Intensity (a.u.)')

# plt.figure(2)
# plt.plot(r, Pfit1, 'k', r, Pfit2, 'r')
# # plt.fill_between(r, Pci95[:, 0], Pci95[:, 1], alpha=0.3)
# plt.xlabel('Distance (nm)')
# plt.ylabel('P ($nm^{-1}$)')
# plt.ylim(-0.5, np.max(Pfit1)+1)
# plt.figure(3)
# plt.plot(r, P2, 'k')
# plt.fill_between(r, Pci95_2[:, 0], Pci95_2[:, 1], alpha=0.3)
# plt.xlabel('Distance (nm)')
# plt.ylabel('P ($nm^{-1}$)')
# plt.ylim(-1, 25)

# , freq, np.real(Pake), 'r')
plt.plot(freq, np.real(Pake), 'k-+', nu, fft_dl, 'r-+')
