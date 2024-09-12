# -*- coding: utf-8 -*-
"""
Created on Fri Jan 26 15:38:18 2024
Script to import Eseem versus Tau dataset from Bruker EPR Elexsys apparatus.
It accept datafiles in the Bruker DSC/DTA format.

Dependencies: You need the scripts "ImportMultipleFiles.py", windowing.py,
fdaxis.py, automatic_phase.py and eprload_BrukerBES3T.py. 
Check the Github repository:
    https://github.com/TChauvire/EPR_ESR_Suite

It follows the routine method describes in the paper
Sergei A. Dzuba and Derek Marshc: DOI: 10.1039/9781782620280-00102
Electron Paramag. Reson., 2015, 24, 102–121
 
The following succesive data treatment is achieved: 
1) automatic zero order phase correction in time damain of the data 
2) extraction of the real part
3) background correction with n_order polynomial of the logarithm 
   of the real part 
4) hamming windowing
5) and zero filling (5x the initial number of point)
6) fft of the corrected real part
7) absolute value of the fft.

The following output are returned:
1) a matrix column of the data "DATA_Time" for each tau value:
    first column: time axis (ns), 
    second column: phase corrected real part of the time domain
    third column: phase corrected imaginary part of the time domain
    fourth column: the background fitting of the real part
    fifth column: the background corrected of the real part
 
2) a matrix column of the data "DATA_Freq" for each tau value:
    first column: frequency axis (MHz)
    second column: real part of the fourier transform of the data
    third column: imaginary part of the fourier transform of the data
    fourth column: absolute value of the fourier transform of the data
    fifth column: absolute value normalized of the fourier transform of the data

3) a header matrix "HEADER_Time" for each column of output data matrix 
    "DATA" with updated name with the initial name of each files

4) a header matrix "HEADER_Freq" for each column of output data matrix 
   "DATA" with updated name with the initial name of each files

5) a matrix output "Data_Sum":
    first row: time domain in ns
    second row: eseem time data summed over tau 
    third row: frequency domain in MHz
    fourth row: eseem fft data summed over tau

In the meantime, the figures are automatically saved in the same folder
where tha data are located.
@author: tim_t
"""
import numpy as np
from ImportMultipleFiles import OpenRidmeFile
import matplotlib.pyplot as plt
from os import path, makedirs
plt.close('all')
# File location in folder
# FullFileName = 'D:\\Documents\\Recherches\\ResearchAssociatePosition\\Data\\'\
#     'PulsedEPR\\2024\\2024_Jess\\20240202\\CuHis2_20K\\Eseem\\'\
#     '20240202_CuHis2_500uM_Eseem3380G_7dB_20K.DSC'
# FullFileName = 'D:\\Documents\\Recherches\\ResearchAssociatePosition\\Data\\'\
#     'PulsedEPR\\2024\\2024_Jess\\20240204\\'\
#     '20240204_500uM_CuNTA_30pGly_Eseem.DSC'

FullFileName = 'D:\\Documents\\Recherches\\ResearchAssociatePosition\\Data\\'\
    'PulsedEPR\\2024\\2024_Rebecca\\April2024\\240418\\240418_0.5To1_Ridme8Steps_80K_Longerd1.DSC'

# FullFileName = 'D:\\Documents\\Recherches\\ResearchAssociatePosition\\Data\\'\
#     'PulsedEPR\\2024\\2024_Jess\\20240208\\Eseem\\'\
#     '100uM_WTCheA_pH8.5_1to1_CuNTA_Eseem_20K.DSC'

# FullFileName = 'D:\\Documents\\Recherches\\ResearchAssociatePosition\\Data\\'\
#     'PulsedEPR\\2024\\2024_Jess\\20240131\\80K\\Eseem\\'\
#     '20240131_CuNTAHis_500uM_Eseem.DSC'
# FullFileName = 'D:\\Documents\\Recherches\\ResearchAssociatePosition\\Data\\'\
#     'PulsedEPR\\2024\\2024_Rebecca\\240103\\110K\\3PEseem\\12171G\\'\
#     '240106_10to1_Cobalt_110K_3PEseem12171G.DSC'
fileID = path.split(FullFileName)
poly = 2
Data_Time, Header_Time = OpenRidmeFile(
    FullFileName, window=20, poly=3, mode='poly')
npts = Data_Time.shape[0]

for i in range(int(Data_Time.shape[1]/5)):
    x = Data_Time[0:npts, 5*i]
    y0 = Data_Time[0:npts, 1].real
    yreal = Data_Time[0:npts, 5*i+1]
    yreal2 = (yreal/(y0))/np.max(yreal/(y0))
    yimag = Data_Time[0:npts, 5*i+2]
    ypoly = Data_Time[0:npts, 5*i+3]
    yridme = Data_Time[0:npts, 5*i+4]
    # Plot the time domain
    plt.figure(i)
    fig, axes = plt.subplots(1, 3)
    fig.suptitle(fileID[1]+'_'+Header_Time[5*i+1].rsplit('_', 1)[1])
    l1, = axes[0].plot(x.ravel()/1000, yreal.ravel(), 'k',
                       label='Real Part', linewidth=2)
    l2, = axes[0].plot(x.ravel()/1000, yimag.ravel(), 'r',
                       label='Imaginary Part', linewidth=2)
    axes[0].set_xlabel("Time [us]", fontsize=14, weight='bold')
    axes[0].set_ylabel("Intensity [a.u.]", fontsize=14, weight='bold')
    axes[0].legend(loc='upper right')
    axes[0].set_title('Phased Data')
    axes[0].set_xlim(-0.2, 1.2)
    l3, = axes[1].plot(x.ravel()/1000, yreal2.ravel(), 'k',
                       label='Real Part', linewidth=2)
    l4, = axes[1].plot(x.ravel()/1000, ypoly.ravel(), 'r',
                       label='Bckg Subtracted', linewidth=2)
    axes[1].set_xlabel("Time [us]", fontsize=14, weight='bold')
    axes[1].set_ylabel("Log(Intensity [a.u.])", fontsize=14, weight='bold')
    axes[1].legend(loc='upper right')
    axes[1].set_title('Bckg Fitting, polyexporder = {0}'.format(poly))
    axes[1].set_xlim(-0.2, 1.2)
    l5, = axes[2].plot(x.ravel()/1000, yridme.ravel(), 'k',
                       label='Bckg Subtracted Real Part', linewidth=2)
    axes[2].set_xlabel("Time [us]", fontsize=14, weight='bold')
    axes[2].set_ylabel("Intensity [a.u.]", fontsize=14, weight='bold')
    axes[2].legend(loc='upper right')
    axes[2].set_title('Bckg Subtracted Data, polyexporder = {0}'.format(poly))
    axes[2].set_xlim(-0.2, 1.2)
    font = {'family': 'tahoma',
            'weight': 'normal',
            'size': 14}
    plt.rc('grid', linestyle="--", color='grey')
    plt.rc('font', **font)
    plt.rc('xtick', labelsize=14)
    plt.rc('ytick', labelsize=14)
    for j, ax in enumerate(axes):
        axes[j].grid()
    figManager = plt.get_current_fig_manager()
    figManager.window.showMaximized()
    plt.subplots_adjust(left=0.07, right=0.984, top=0.893, bottom=0.085,
                        hspace=0.2, wspace=0.26)
    fig.set_size_inches(18, 10)
    fig.set_dpi(200)
    makedirs(fileID[0] + '\\Figures\\', exist_ok=True)

    plt.savefig(fileID[0] + '\\Figures\\figure{0}.png'.format(2*i))

# Carry out the sum over the mixing time:
plt.close('all')
ymean = np.full((npts, 2), np.nan)
extractdata = np.full((npts,), 0.0)
for i in range(int(Data_Time.shape[1]/5)):
    ymean[:, 0] = Data_Time[0:npts, 0]
    yridme = Data_Time[0:npts, 5*i+4]
    extractdata += yridme[:,]
    ymean[:, 1] = extractdata/int(Data_Time.shape[1]/5)

plt.figure()
plt.title(fileID[1]+'_Ridme over Tmix')
plt.plot(ymean[:, 0].ravel()/1000, ymean[:, 1], 'k', label='Ridme over Tmix',
         linewidth=2)
plt.xlabel('Time [us]', fontsize=16)
plt.ylabel('S(t)', fontsize=16)
plt.rc('grid', linestyle="--", color='grey')
plt.rc('font', **font)
plt.rc('xtick', labelsize=16)
plt.rc('ytick', labelsize=16)

# Export the ascii file for Deer analysis
fullnewfilename = path.join(fileID[0], str(fileID[1]+'_RidmeSumOverTmix.dat'))
makedirs(path.dirname(fullnewfilename), exist_ok=True)
np.savetxt(fullnewfilename, ymean, fmt=['%.5e', '%.15e'],
           delimiter='\t', header=fileID[1])
