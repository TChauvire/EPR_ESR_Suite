# -*- coding: utf-8 -*-
"""
Created on Tue Jan  2 12:10:59 2024

@author: tim_t
"""
from ImportMultipleFiles import ImportMultipleNameFiles, MaxLengthOfFiles, OpenEDNMRFiles
import numpy as np
from ImportMultipleFiles import eprload
from scipy.special import voigt_profile
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from os import path

folder = 'D:\\Documents\\Recherches\\ResearchAssociatePosition\\Data\\'\
    'PulsedEPR\\2024\\2024_Rebecca\\March2023\\EDNMR'
ListOfFiles = ImportMultipleNameFiles(folder, Extension='.DSC')
maxlen = MaxLengthOfFiles(ListOfFiles)
fulldata, header, snr = OpenEDNMRFiles(
    ListOfFiles, Scaling=None, polyorder=1, window_length=50)
y1, x1, par1 = eprload(ListOfFiles[1])
y2, x2, par2 = eprload(ListOfFiles[4])
y3, x3, par3 = eprload(ListOfFiles[7])
FileIndex = 7
yexp = fulldata[:, (FileIndex)*6+5].ravel()
fileID = path.split(ListOfFiles[FileIndex])
plt.close('all')
plt.figure(0)
plt.plot(x1, fulldata[:, 11], 'k')
plt.plot(x2, fulldata[:, 29], 'r')
plt.plot(x3, fulldata[:, 47], 'b')
plt.xlabel('Frequency [MHz]')
plt.ylabel('Echo Intensity [a.u.]')


def gaussian(x, sigma):
    # np.amax(np.exp(-0.5 * x**2/sigma**2)/(sigma * np.sqrt(2*np.pi)))
    maxi = 0.8
    return (np.exp(-0.5 * x**2/sigma**2)/(sigma * np.sqrt(2*np.pi)))/maxi


def lorentzian(x, gamma):
    maxi = 0.8  # np.amax(gamma/(np.pi * (np.square(x)+gamma**2)))
    return (gamma/(np.pi * (np.square(x)+gamma**2)))/maxi


def bilorentzian(x, gamma1, gamma2):

    yfit = lorentzian(x, np.abs(gamma1))+lorentzian(x, np.abs(gamma2))
    return yfit


def voigt(x, sigma, gamma):
    maxi = 0.9  # np.amax(voigt_profile(x, sigma, gamma, out=None))
    return voigt_profile(x, sigma, gamma, out=None)/maxi


popt1, pcov1 = curve_fit(gaussian, x1[:, ].ravel(), yexp,
                         p0=[10], sigma=None, absolute_sigma=False,
                         check_finite=None)
yfit1 = gaussian(x1, popt1[0]).ravel()
popt2, pcov2 = curve_fit(lorentzian, x1[:,].ravel(), yexp,
                         p0=[2], sigma=None, absolute_sigma=False,
                         check_finite=None)
yfit2 = lorentzian(x1, popt1[0]).ravel()
x0 = [1, 1000]
popt3, pcov3 = curve_fit(bilorentzian, x1[:,].ravel(), yexp,
                         p0=x0, sigma=None, absolute_sigma=False,
                         check_finite=None)
yfit3 = bilorentzian(x1, popt3[0], popt3[1]).ravel()
plt.figure(1)
plt.plot(x1.ravel(), yexp, 'k', x1.ravel(), yfit1,
         'r', x1.ravel(), yfit2, 'g', x1.ravel(), yfit3, 'b')
plt.xlabel('Frequency [MHz]')
plt.ylabel('Echo Intensity [a.u.]')
plt.suptitle(fileID[1])

# plt.figure(2)
fig, axes = plt.subplots(2, 1)
plt.suptitle(fileID[1], fontsize=18)
axes[0].plot(x1.ravel(), yexp-yfit2, 'k', )
axes[0].set_title('Lorentzian fitting subtraction', fontsize=16)
axes[1].plot(x1.ravel(), yexp-yfit3, 'k')
axes[1].set_title('BiLorentzian fitting subtraction', fontsize=16)
for j, ax in enumerate(axes):
    axes[j].set_xlabel('Frequency [MHz]', fontsize=14, weight='bold')
    axes[j].set_ylabel('Echo Intensity [a.u.]', fontsize=14, weight='bold')
    axes[j].grid()
plt.tight_layout()
