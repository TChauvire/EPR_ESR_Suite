# -*- coding: utf-8 -*-
"""
Created on Tue Oct 31 16:44:55 2023

@author: tim_t
"""
import matplotlib.pyplot as plt
import numpy as np
from ImportMultipleFiles import eprload
from ImportDeerAnalysisParameters import BckgndSubtractionOfRealPart
from DeerDenoising import wavedec, apply_thresholding
# Deer_Denoising, generate_denoised_signal
from pywt import waverec

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
filename = filename8
DataDict, ParamDict = {}, {}

DataDict, ParamDict = BckgndSubtractionOfRealPart(
    filename, DataDict, ParamDict)
y2, abscissa, par = eprload(filename, Scaling=None)
TITL = str(par['TITL'])
npts = abscissa.shape[0]
Exp_parameter = ParamDict.get(par['TITL'])
tmin = Exp_parameter.get('zerotime')
tmax = Exp_parameter.get('tmax')
itmin = Exp_parameter.get('itmin')
itmax = Exp_parameter.get('itmax')
y_phased = DataDict.get(str(par['TITL']))[0:npts, 3].ravel()[itmin:itmax,]
bckg = DataDict.get(str(par['TITL']))[0:npts, 4].ravel()[itmin:itmax,]
x_y = DataDict.get(str(par['TITL']))[0:npts, 5].ravel()[itmin:itmax,]
newy = np.real(x_y)  # [itmin:itmax,])
maxvalue = np.max(y_phased)
noise = Exp_parameter.get('NoiseLevel')
SNR = Exp_parameter.get('DEER_SNR')
# -max(np.real(y2[itmin:itmax,]))).ravel()
# newy = (np.real(y2[itmin:itmax,])/np.real(np.max(y2[itmin:itmax,]))).ravel()
wavename = 'db6'
x_x = abscissa.ravel()
newx = np.real(x_x)  # [itmin:itmax,])
newnpts = newx.shape[0]
# raw_signal, signal, N, coeffs, full_coeffs, thresholds, denoised_signal, denoised_coeffs = Deer_Denoising(
#     x_x, x_y, wavename)
N = int(np.floor(np.log2(len(newy))))
full_coeffs = wavedec(newy, wavename, mode='periodic',  # 'smooth',  # 'symmetric',  # ' #periodic',  # 'antisymmetric',  # 'symmetric',
                      level=N, trim_approx=True)
# mode 'zero' and 'periodic' seem to work the best
thresholds = np.zeros(N+1)
coeffdict = {}
for i in range(N):
    size = len(full_coeffs[-i])
    thresholds[-i] = size
    new_coeffs = apply_thresholding(full_coeffs, thresholds)
    coeffdict.update({str(i): new_coeffs})
ds0, ds1, ds2, ds3, ds4, ds5 = [], [], [], [], [], []
ds0 = waverec(coeffdict.get(str(0)), wavename)[
    0:newnpts]*maxvalue + bckg
ds1 = waverec(coeffdict.get(str(1)), wavename)[
    0:newnpts]*maxvalue + bckg
ds2 = waverec(coeffdict.get(str(2)), wavename)[
    0:newnpts]*maxvalue + bckg
ds3 = waverec(coeffdict.get(str(3)), wavename)[
    0:newnpts]*maxvalue + bckg
ds4 = waverec(coeffdict.get(str(4)), wavename)[
    0:newnpts]*maxvalue + bckg
ds5 = waverec(coeffdict.get(str(5)), wavename)[
    0:newnpts]*maxvalue + bckg

# ds0 = waverec(coeffdict.get(str(0)), wavename)[0:newnpts]
# ds1 = waverec(coeffdict.get(str(1)), wavename)[0:newnpts]
# ds2 = waverec(coeffdict.get(str(2)), wavename)[0:newnpts]
# ds3 = waverec(coeffdict.get(str(3)), wavename)[0:newnpts]
# ds4 = waverec(coeffdict.get(str(4)), wavename)[0:newnpts]
# ds5 = waverec(coeffdict.get(str(5)), wavename)[0:newnpts]


fig, axes = plt.subplots(3, 2)
fig.suptitle('Test WavPDSDenoising: \n Noiselevel = ' + str(round(noise[1], 4))
             + ", DEER_SNR = " + str(round(SNR[1], 2)))
axes[0, 0].plot(newx, y_phased, 'k', newx, ds0, 'r')
axes[0, 0].set_xlabel("Time [us]")
axes[0, 0].grid()
axes[0, 0].set_ylabel("Intensity (a.u.)")
axes[0, 0].set_title('Firstlevel', fontsize=10)


axes[0, 1].plot(newx, y_phased, 'k', newx, ds1, 'r')
axes[0, 1].set_xlabel("Time [us]")
axes[0, 1].grid()
axes[0, 1].set_ylabel("Intensity (a.u.)")
axes[0, 1].set_title('Secondlevel', fontsize=10)

axes[1, 0].plot(newx, y_phased, 'k', newx, ds2, 'r')
axes[1, 0].set_xlabel("Time [us]")
axes[1, 0].grid()
axes[1, 0].set_ylabel("Intensity (a.u.)")
axes[1, 0].set_title('Thirdlevel', fontsize=10)

axes[1, 1].plot(newx, y_phased, 'k', newx, ds3, 'r')
axes[1, 1].set_xlabel("Time [us]")
axes[1, 1].grid()
axes[1, 1].set_ylabel("Intensity (a.u.)")
axes[1, 1].set_title('Fourthlevel', fontsize=10)

axes[2, 0].plot(newx, y_phased, 'k', newx, ds4, 'r')
axes[2, 0].set_xlabel("Time [us]")
axes[2, 0].grid()
axes[2, 0].set_ylabel("Intensity (a.u.)")
axes[2, 0].set_title('Fifthlevel', fontsize=10)

axes[2, 1].plot(newx, y_phased, 'k', newx, ds5, 'r')
axes[2, 1].set_xlabel("Time [us]")
axes[2, 1].grid()
axes[2, 1].set_ylabel("Intensity (a.u.)")
axes[2, 1].set_title('Sixthlevel', fontsize=10)

plt.tight_layout()

# np.flipud(
