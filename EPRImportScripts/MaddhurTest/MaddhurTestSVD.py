# -*- coding: utf-8 -*-
"""
Created on Wed Jan  3 11:33:14 2024
Script for analizing 4PDeer distance domain with the SVD scripts develop under
Matlab by Maddhur Srivastava (ms2736@cornell.edu).
It was then adapted by Chris Myer, staff at CAC at Cornell
(http://www.cac.cornell.edu/).
So the original software working with a Bokh graphical user interface is found
at https://github.com/CornellCAC/denoising.
And you may want the access to the orginal more advanced software hosted at
this address : https://denoising.cornell.edu/

The idea is to decompose the signal in the distance domain and put to zero the
signal of no interest.

In the dataset I obtained, I got 3 major peaks: 4.1 nm, 4.7 nm and 6.5nm. We
are doubting of the truth nature of the last one, and may believe it's an
artefact coming from the experience.

Let's prepare 3 curves in the distance domain and plot it in fig1:
    1) the original one with all the peak
    2) the original one with the 6.5 nm peak set to zero
    3) the original one with the 4 + 4.7 nm peaks set to zero

Let's compare the reconstructed signals in the time domain in fig2.

@author: Timothee Chauvire tsc84@cornell.edu
"""

import matplotlib.pyplot as plt
import numpy as np
from ImportMultipleFiles import eprload
from ImportDeerAnalysisParameters import BckgndSubtractionOfRealPart, DenoisingForOneFile
from SVD_scripts import get_KUsV, process_data, rminrmaxcalculus
import deerlab as dl
from os import path, makedirs

plt.close('all')

# File location
f1 = 'C:\\Users\\tim_t\\Python\\EPRImportScripts\\MaddhurTest\\'\
    '\\BrukerFiles\\040619_Aer_EcA_EcW_100pD2O_150K_4PDeer_4500ns.DSC'
f2 = 'C:\\Users\\tim_t\\Python\\EPRImportScripts\\MaddhurTest\\'\
    '\\BrukerFiles\\022421_Aer_AW_15N2H_BL21_150K_DEER_5000ns.DSC'

fullname = [f1, f2]
RMSD = np.full((2000, len(fullname)), np.nan, dtype=float)
diff = np.full((2000, len(fullname)), np.nan, dtype=float)
a = np.full((2000,), np.nan, dtype=float)
idx = [0] * len(fullname)
maxiP0 = [0] * len(fullname)
maxiP1 = [0] * len(fullname)
maxiP2 = [0] * len(fullname)
maxiP3 = [0] * len(fullname)
maxiP4 = [0] * len(fullname)
maxiP5 = [0] * len(fullname)
maxiP6 = [0] * len(fullname)
maxiP7 = [0] * len(fullname)
maxiP8 = [0] * len(fullname)
maxiP9 = [0] * len(fullname)
maxiP10 = [0] * len(fullname)

DataDict, ParamDict = {}, {}
trunc1 = [59, 1, 274, 55, 95]  # first cutoff range, cutoff number
trunc2 = [68, 1, 291, 55, 99]
trunc = [trunc1, trunc2]
for j in range(len(fullname)):
    fileID = path.split(fullname[j])
    TITL = fileID[1]
    filename = fullname[j]
    DataDict, ParamDict = BckgndSubtractionOfRealPart(
        filename, DataDict, ParamDict, Dimensionorder=1, Percent_tmax=19/20,
        mode='polyexp', truncsize=1/2)
    _, x, _ = eprload(filename, Scaling=None)
    npts = x.shape[0]
    Exp_parameter = ParamDict.get(TITL)
    itmax = Exp_parameter.get('itmax')
    y = DataDict.get(TITL)[0:npts, 5].ravel()
    phasedy = DataDict.get(TITL)[0:npts, 3].ravel()
    ModDepth = Exp_parameter.get('ModDepth')
    tmin = Exp_parameter.get('zerotime')
    tmax = Exp_parameter.get('tmax')
    itmin = Exp_parameter.get('itmin')
    newy = np.real(y[itmin:itmax,])
    t = ((DataDict.get(TITL)[0:npts, 0])/1000).ravel()
    newt = np.real(t[itmin:itmax,])
    newnpts = newt.shape[0]
    ### process the data to obtain the distance domain with the SVD method ###
    # distance range has to be in nanometer
    r = np.linspace(1, 14, num=newnpts)
    rmin = r.min()  # distance in nanometer
    rmax = r.max()  # distance in nanometer
    K, U, s, V = get_KUsV(newt, rmin, rmax)  # get the kernel + sigma
    S, sigma, PR, Pr, Picard, sum_Pic = process_data(newt, newy, K,
                                                     U, s, V, rmin, rmax)
    # get the different decomposition Pr
    Pr_trunc = []
    # Apply the truncation cutoff method
    idx_1 = trunc[j][0]  # index 1
    trunc_1 = trunc[j][1]  # truncation 1
    idx_2 = trunc[j][2]  # index 2
    trunc_2 = trunc[j][3]  # truncation 2
    Pr_trunc.extend(Pr.T[0:idx_1, trunc_1])
    Pr_trunc.extend(Pr.T[idx_1:idx_2, trunc_2])
    rsignal = np.real(np.dot(K, Pr_trunc))  # reconstruct the truncated signal
    # in the time domain
    # Plot the time and distance domain
    plt.figure(3*j)
    fig, axes = plt.subplots(2, 1)
    fig.suptitle(TITL)
    axes[0].plot(newt, newy, 'k', newt, np.real(np.dot(K, Pr.T[:, trunc_2])),
                 'r', newt, rsignal, 'b')
    axes[1].plot(r, Pr.T[:, trunc_2], 'r',
                 label='Untruncated distance domain')
    axes[1].plot(r, Pr_trunc, 'b',
                 label='Truncated distance domain')

    axes[1].legend(fontsize='large', )
    plt.tight_layout()

    ### Apply the zero filling and reconstruct signal in the time domain ###
    Pr_trunc1 = []  # with 4 and 4.7 nm
    Pr_trunc2 = []  # with 6 nm only
    idx_3 = trunc[j][4]
    Pr_trunc1.extend([0]*idx_1)
    Pr_trunc1.extend(Pr.T[idx_1:idx_3, trunc_2])
    Pr_trunc1.extend([0]*(idx_2-idx_3))
    Pr_trunc2.extend([0]*idx_3)
    Pr_trunc2.extend(Pr.T[idx_3:idx_2, trunc_2])
    rsignal1 = np.real(np.dot(K, Pr_trunc1))
    rsignal2 = np.real(np.dot(K, Pr_trunc2))
    plt.figure(3*j+1)
    fig, axes = plt.subplots(2, 1)
    fig.suptitle(TITL)
    axes[0].plot(newt, rsignal, 'k', newt, rsignal1,
                 'r', newt, rsignal2, 'b')
    axes[1].plot(r,  Pr_trunc, 'k', r,
                 Pr_trunc1, 'r', r, Pr_trunc2, 'b')
    ### Save the raw files for cwt analysis in .dat format ###
    # Save the raw data
    raw_data = np.column_stack([newt.real, phasedy[itmin:itmax].real])
    makedirs(path.dirname(fullname[j]), exist_ok=True)
    header = '\n'.join(['Filename: ' + fileID[1],
                        'Bckg_type: ' + 'NoBckg',
                        'FirstColumn: ' + 'Time[us]',
                        'SecondColumn: ' + 'RawData'])
    np.savetxt(str(fullname[j]+'.dat'), raw_data, fmt=['%.15e', '%.15e'],
               delimiter='\t', header=header)
    # Save the bckgnd subtracted data
    data = np.column_stack([newt.real, newy.real])
    Bckg_type = Exp_parameter.get('Bckg_type')
    header2 = '\n'.join(['Filename: ' + fileID[1],
                        'Bckg_type: ' + Bckg_type,
                         'FirstColumn: ' + 'Time[us]',
                         'SecondColumn: ' + 'BckgSubtractedData'])
    np.savetxt(str(fullname[j]+'_BckgSubtracted.dat'), data, fmt=['%.15e', '%.15e'],
               delimiter='\t', header=header2)
    plt.figure(3*j+2)
    fig, axes = plt.subplots(2, 1)
    fig.suptitle(TITL)
    axes[0].plot(newt.real, phasedy[itmin:itmax].real, 'k')
    axes[1].plot(newt.real, newy.real, 'k')
