# -*- coding: utf-8 -*-
"""
Created on Mon Oct 30 12:33:55 2023

@author: tim_t
"""
import matplotlib.pyplot as plt
import numpy as np
from ImportMultipleFiles import eprload
from ImportDeerAnalysisParameters import BckgndSubtractionOfRealPart, DenoisingForOneFile
from SVD_scripts import get_KUsV, process_data, rminrmaxcalculus
import deerlab as dl

plt.close('all')

# File location
f1 = 'C:\\Users\\tim_t\\Python\\EPRImportScripts\\DataFilesDEER\\'\
    '\\BrukerFiles\\112118_AerInvivoBL21_4Pdeer2500ns_150K.DSC'
f2 = 'C:\\Users\\tim_t\\Python\\EPRImportScripts\\DataFilesDEER\\'\
    '\\BrukerFiles\\120918_AerUU2610_DEER_2us_150K.DSC'
f3 = 'C:\\Users\\tim_t\\Python\\EPRImportScripts\\DataFilesDEER\\'\
    '\\BrukerFiles\\121418_AerBL21_2us_150K.DSC'
f4 = 'C:\\Users\\tim_t\\Python\\EPRImportScripts\\DataFilesDEER\\'\
    '\\BrukerFiles\\121418_AerBL21_2500ns_150K.DSC'
f5 = 'C:\\Users\\tim_t\\Python\\EPRImportScripts\\DataFilesDEER\\'\
    '\\BrukerFiles\\031519_Aer_EcCheA_EcCheW_BL21_150K_DEER_3200ns.DSC'
f6 = 'C:\\Users\\tim_t\\Python\\EPRImportScripts\\DataFilesDEER\\'\
    'BrukerFiles\\031419_Aer_EcCheA_EcCheW_BL21_150K_DEER_2500ns.DSC'
f7 = 'C:\\Users\\tim_t\\Python\\EPRImportScripts\\DataFilesDEER\\'\
    '\\BrukerFiles\\011619_AerCheACheW_Bl21_4PDeer_2500ns_150K.DSC'
f8 = 'C:\\Users\\tim_t\\Python\\EPRImportScripts\\DataFilesDEER\\'\
    '\\BrukerFiles\\011619_AerCheACheW_Bl21_4PDeer_3200ns_150K.DSC'
f9 = 'C:\\Users\\tim_t\\Python\\EPRImportScripts\\DataFilesDEER\\'\
    '\\BrukerFiles\\032119_Aer_Tar_EcA_EcW_BL21_150K_Deer_2500ns.DSC'
f10 = 'C:\\Users\\tim_t\\Python\\EPRImportScripts\\DataFilesDEER\\'\
    '\\BrukerFiles\\032219_Aer_Tar_EcA_EcW_BL21_150K_Deer_3200ns.DSC'
f11 = 'C:\\Users\\tim_t\\Python\\EPRImportScripts\\DataFilesDEER\\'\
    '\\BrukerFiles\\051519_AerECAW_150K_2500ns_4PDEER.DSC'

fullname = [f1, f2, f3, f4, f5, f6, f7, f8, f9, f10, f11]
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

for j in range(len(fullname)):
    DataDict, ParamDict = {}, {}
    filename = fullname[j]
    DataDict, ParamDict = BckgndSubtractionOfRealPart(
        filename, DataDict, ParamDict, Dimensionorder=3, Percent_tmax=39/40,
        mode='strexp')
    # DataDict, ParamDict = DenoisingForOneFile(fullname[j], DataDict, ParamDict,
    #                                           Dimensionorder=2, Percent_tmax=9/10, mode='strexp')
    _, x, par = eprload(filename, Scaling=None)
    TITL = str(par['TITL'])
    npts = x.shape[0]
    Exp_parameter = ParamDict.get(par['TITL'])
    itmax = Exp_parameter.get('itmax')
    y = DataDict.get(str(par['TITL']))[0:npts, 5].ravel()
    ModDepth = Exp_parameter.get('ModDepth')
    tmin = Exp_parameter.get('zerotime')
    tmax = Exp_parameter.get('tmax')
    itmin = Exp_parameter.get('itmin')
    newy = np.real(y[itmin:itmax,])

    t = ((DataDict.get(str(par['TITL']))[0:npts, 0])/1000).ravel()
    newt = np.real(t[itmin:itmax,])
    newnpts = newt.shape[0]
    r = dl.distancerange(newt, nr=newnpts)
    K, U, s, V = get_KUsV(newt)
    rrange = rminrmaxcalculus(newt)
    rmin = rrange[0]
    rmax = rrange[1]
    S, sigma, PR, Pr, Picard, sum_Pic = process_data(newt, newy, K,
                                                     U, s, V)

    for i in range(Pr.shape[1]):
        rsignal = np.real(np.dot(K, Pr.T[:, i]))
        RMSD[i, j] = np.sqrt(np.sum((newy-rsignal)**2))
    diff[:, j] = np.abs(np.gradient(RMSD[:, j]))
    a = diff[:, j]
    idx[j] = np.where(a < 0.0002)[0][0]
    maxiP0[j] = [np.argmax(Pr.T[:, idx[j]]), np.max(Pr.T[:, idx[j]])]
    maxiP1[j] = [np.argmax(Pr.T[:, idx[j]+5]), np.max(Pr.T[:, idx[j]+5])]
    maxiP2[j] = [np.argmax(Pr.T[:, idx[j]+10]), np.max(Pr.T[:, idx[j]+10])]
    maxiP3[j] = [np.argmax(Pr.T[:, idx[j]+15]), np.max(Pr.T[:, idx[j]+15])]
    maxiP4[j] = [np.argmax(Pr.T[:, idx[j]+20]), np.max(Pr.T[:, idx[j]+20])]
    maxiP5[j] = [np.argmax(Pr.T[:, idx[j]+25]), np.max(Pr.T[:, idx[j]+25])]
    # maxiP6[j] = [np.argmax(Pr.T[:, idx[j]+6]), np.max(Pr.T[:, idx[j]+6])]
    # maxiP7[j] = [np.argmax(Pr.T[:, idx[j]+7]), np.max(Pr.T[:, idx[j]+7])]
    # maxiP8[j] = [np.argmax(Pr.T[:, idx[j]+8]), np.max(Pr.T[:, idx[j]+8])]
    # maxiP9[j] = [np.argmax(Pr.T[:, idx[j]+9]), np.max(Pr.T[:, idx[j]+9])]
    # maxiP10[j] = [np.argmax(Pr.T[:, idx[j]+10]), np.max(Pr.T[:, idx[j]+10])]
    # .colors.Colormap(name, N=256)
    fig, axes = plt.subplots(2, 1)
    fig.suptitle(TITL)
    axes[0].plot(t, y, 'k', newt, newy, 'r', newt,
                 np.dot(K, Pr.T[:, idx[j]]), 'b')
    #axes[1].plot(RMSD[:, j])
    #axes[2].plot(diff[:, j])
    # axes[3].plot(r, ModDepthLine, '--')
    axes[1].plot(r, Pr.T[:, idx[j]], 'C0',
                 label='SVDCutoff={0}'.format(idx[j]))
    axes[1].plot(r[maxiP0[j][0],], maxiP0[j][1], '+', label=str())
    axes[1].plot(r, Pr.T[:, idx[j]+5], 'C1',
                 label='SVDCutoff={0}'.format(idx[j]+5))
    axes[1].plot(r[maxiP1[j][0],], maxiP1[j][1], '+', label=str())
    axes[1].plot(r, Pr.T[:, idx[j]+10], 'C2',
                 label='SVDCutoff={0}'.format(idx[j]+10))
    axes[1].plot(r[maxiP2[j][0],], maxiP2[j][1], '+', label=str())
    axes[1].plot(r, Pr.T[:, idx[j]+15], 'C3',
                 label='SVDCutoff={0}'.format(idx[j]+15))
    axes[1].plot(r[maxiP3[j][0],], maxiP3[j][1], '+', label=str())
    axes[1].plot(r, Pr.T[:, idx[j]+20], 'C4',
                 label='SVDCutoff={0}'.format(idx[j]+20))
    axes[1].plot(r[maxiP4[j][0],], maxiP4[j][1], '+', label=str())
    axes[1].plot(r, Pr.T[:, idx[j]+25], 'C5',
                 label='SVDCutoff={0}'.format(idx[j]+25))
    axes[1].plot(r[maxiP5[j][0],], maxiP5[j][1], '+', label=str())
    # r, Pr.T[:, idx[j]+6], 'C6', label='SVDCutoff={0}'.format(idx[j]+2), r[maxiP6[j][0],], maxiP6[j][1], '+',
    # r, Pr.T[:, idx[j]+7], 'C7', label='SVDCutoff={0}'.format(idx[j]+2), r[maxiP7[j][0],], maxiP7[j][1], '+',
    # r, Pr.T[:, idx[j]+8], 'C8', r[maxiP8[j]
    #                               [0],], maxiP8[j][1], '+',
    # r, Pr.T[:, idx[j]+9], 'C9', r[maxiP9[j]
    #                               [0],], maxiP9[j][1], '+',
    # r, Pr.T[:, idx[j]+10], 'C10', r[maxiP10[j][0],], maxiP10[j][1], '+')
    axes[1].legend(fontsize='xx-small', )
    plt.tight_layout()
# fig, axes = plt.subplots(2, 2)
# fig.suptitle("Test SVD Reconstruction")
# # https://matplotlib.org/stable/api/axes_api.html
# axes[0, 0].plot(t, newy, t, rsignal)
# axes[0, 0].set_xlabel("Time [us]")
# axes[0, 0].grid()
# axes[0, 0].set_ylabel("Intensity (a.u.)")
# axes[0, 0].set_title('Time Domain')
# # axes[0].legend()

# axes[0, 1].plot(sigma)
# axes[0, 1].set_yscale('log')
# axes[0, 1].grid()
# axes[0, 1].set_ylabel("Singular Values")
# axes[0, 1].set_xlabel("SVC number")
# axes[0, 1].set_title('Singular Values')

# axes[1, 0].plot(r, Pr.T[:, 40])
# axes[1, 0].grid()
# axes[1, 0].set_ylabel("P(r)")
# axes[1, 0].set_xlabel("Distance r [nm]")
# axes[1, 0].set_title('Distance Domain')

# axes[1, 1].plot(sum_Pic)
# axes[1, 1].set_yscale('log')
# axes[1, 1].grid()
# axes[1, 1].set_ylabel("Picard_summed")
# axes[1, 1].set_xlabel("SVC number")
# axes[1, 1].set_title('Picard plot')

# plt.tight_layout()
