# -*- coding: utf-8 -*-
"""
Created on Mon Oct 30 12:33:55 2023

@author: tim_t
"""
import matplotlib.pyplot as plt
import numpy as np
from SVD_scripts_Aritro import get_KUsV, process_data, rminrmaxcalculus
from SVD_scripts_Aritro import BckgndSubtractionOfRealPart, eprload
import deerlab as dl
from os import path

plt.close('all')

# File location to change
f1 = 'C:\\Users\\tim_t\\Python\\EPRImportScripts\\Data\\DataFilesDEER\\'\
    'TestFiles\\DEER_Anthr_50K_af1+42.DSC'
f2 = 'C:\\Users\\tim_t\\Python\\EPRImportScripts\\\Data\\DataFilesDEER\\'\
    'TestFiles\\DEER_Anthr_50K_af1-66.DSC'
f3 = 'C:\\Users\\tim_t\\Python\\EPRImportScripts\\\Data\\DataFilesDEER\\'\
    'TestFiles\\M_GdnHCl_DEER_2700ns.DSC'

fullname = [f1, f2, f3]
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

for j in range(len(fullname)):
    DataDict, ParamDict = {}, {}
    filename = fullname[j]
    DataDict, ParamDict = BckgndSubtractionOfRealPart(
        filename, DataDict, ParamDict, Dimensionorder=2, Percent_tmax=9/10,
        mode='poly')
    y2, abscissa, par = eprload(filename, Scaling=None)
    fileID = path.split(fullname[j])
    TITL = fileID[1]
    npts = abscissa.shape[0]
    y = DataDict.get(TITL)[0:npts, 5].ravel()

    Exp_parameter = ParamDict.get(TITL)
    ModDepth = Exp_parameter.get('ModDepth')
    tmin = Exp_parameter.get('zerotime')
    tmax = Exp_parameter.get('tmax')
    itmin = Exp_parameter.get('itmin')
    itmax = Exp_parameter.get('itmax')
    tmax = abscissa[itmax, 0]/1000
    newy = np.real(y[itmin:itmax,])
    new_npts = newy.shape[0]
    t = ((abscissa[itmin:itmax,]-abscissa[itmin])/1000).ravel()
    r = dl.distancerange(t, nr=new_npts)
    ModDepthLine = np.asarray([ModDepth]*new_npts)
    rrange = rminrmaxcalculus(t)
    rmin = rrange[0]
    rmax = rrange[1]
    K, U, s, V = get_KUsV(t, rmin, rmax)
    S, sigma, PR, Pr, Picard, sum_Pic = process_data(t, newy, K,
                                                     U, s, V, rmin, rmax)

    for i in range(Pr.shape[1]):
        rsignal = np.dot(K, Pr.T[:, i])
        RMSD[i, j] = np.sqrt(np.sum((newy-rsignal)**2))

    diff[:, j] = np.abs(np.gradient(RMSD[:, j]))
    a = diff[:, j]
    idx[j] = np.where(a < 0.0002)[0][0]
    maxiP0[j] = [np.argmax(Pr.T[:, idx[j]]), np.max(Pr.T[:, idx[j]])]
    maxiP1[j] = [np.argmax(Pr.T[:, idx[j]+1]), np.max(Pr.T[:, idx[j]+1])]
    maxiP2[j] = [np.argmax(Pr.T[:, idx[j]+2]), np.max(Pr.T[:, idx[j]+2])]
    maxiP3[j] = [np.argmax(Pr.T[:, idx[j]+3]), np.max(Pr.T[:, idx[j]+3])]
    maxiP4[j] = [np.argmax(Pr.T[:, idx[j]+4]), np.max(Pr.T[:, idx[j]+4])]
    maxiP5[j] = [np.argmax(Pr.T[:, idx[j]+5]), np.max(Pr.T[:, idx[j]+5])]
    # .colors.Colormap(name, N=256)
    fig, axes = plt.subplots(3, 1)
    plt.suptitle(TITL, fontsize=20, fontweight='bold')
    axes[0].plot(t, newy, 'k', label='Signal S(t)', linewidth=1.5)
    axes[0].plot(t, np.dot(K, Pr.T[:, i]), 'r',
                 label='Reconstituted Signal', linewidth=1.5)
    axes[0].set_xlabel('Time [us]', fontsize=16, fontweight='bold')
    axes[0].set_ylabel('S(t)', fontsize=16, fontweight='bold')
    axes[0].legend(fontsize='small')
    axes[1].plot(RMSD[:, j], linewidth=1.5)
    axes[1].set_xlabel('Time [us]', fontsize=16, fontweight='bold')
    axes[1].set_ylabel('Residual', fontsize=16, fontweight='bold')
    axes[2].plot(r, ModDepthLine, '--', linewidth=1.5)
    axes[2].plot(r, Pr.T[:, idx[j]], 'C0',
                 label='SVDCutoff={0}'.format(idx[j]), linewidth=1.5)
    axes[2].plot(r[maxiP0[j][0],], maxiP0[j][1], '+', label=str())
    axes[2].plot(r, Pr.T[:, idx[j]+1], 'C1',
                 label='SVDCutoff={0}'.format(idx[j]+1), linewidth=1.5)
    axes[2].plot(r[maxiP1[j][0],], maxiP1[j][1], '+', label=str())
    axes[2].plot(r, Pr.T[:, idx[j]+2], 'C2',
                 label='SVDCutoff={0}'.format(idx[j]+2), linewidth=1.5)
    axes[2].plot(r[maxiP2[j][0],], maxiP2[j][1], '+', label=str())
    axes[2].plot(r, Pr.T[:, idx[j]+3], 'C3',
                 label='SVDCutoff={0}'.format(idx[j]+3), linewidth=1.5)
    axes[2].plot(r[maxiP3[j][0],], maxiP3[j][1], '+', label=str())
    axes[2].plot(r, Pr.T[:, idx[j]+4], 'C4',
                 label='SVDCutoff={0}'.format(idx[j]+4), linewidth=1.5)
    axes[2].plot(r[maxiP4[j][0],], maxiP4[j][1], '+', label=str())
    axes[2].plot(r, Pr.T[:, idx[j]+5], 'C5',
                 label='SVDCutoff={0}'.format(idx[j]+5), linewidth=1.5)
    axes[2].plot(r[maxiP5[j][0],], maxiP5[j][1], '+', label=str())
    axes[2].set_xlabel('Distance [nm]', fontsize=16, fontweight='bold')
    axes[2].set_ylabel('P(r)', fontsize=16, fontweight='bold')
    axes[2].legend(fontsize='xx-small')
    font = {'family': 'tahoma',
            'weight': 'normal',
            'size': 16}
    plt.rc('grid', linestyle="--", color='grey')
    plt.rc('font', **font)
    plt.rc('xtick', labelsize=16)
    plt.rc('ytick', labelsize=16)
    for j, ax in enumerate(axes):
        axes[j].grid()
    plt.tight_layout()
