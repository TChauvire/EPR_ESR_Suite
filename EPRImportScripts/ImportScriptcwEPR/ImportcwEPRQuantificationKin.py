# -*- coding: utf-8 -*-
"""
Created on Tue Jan  2 12:10:59 2024

@author: tim_t
"""
from ImportMultipleFiles import ImportMultipleNameFiles, OpenDoubleIntegralKin2
import numpy as np
from os import path, makedirs
import matplotlib.pyplot as plt
folder = 'D:\\Documents\\Recherches\\ResearchAssociatePosition\\Data\\cwEPR'\
    '\\2024\\AerProject\\240430\\'
ListOfFiles = ImportMultipleNameFiles(folder, Extension='.DSC')
nfile = 4
Filename = ListOfFiles[nfile]
FirstDeriv, ZeroDeriv, Header, IntgValue, HeaderInt = OpenDoubleIntegralKin2(
    Filename, Scaling=None, col=[6, 7, 8, 9, 10], polyorder=[1, 1], window=200)

fileID = path.split(ListOfFiles[nfile])
# plot the first derivative
x = FirstDeriv[:, 0]/10
y = FirstDeriv[:, 1]
x2 = ZeroDeriv[:, 0]/10
y2 = ZeroDeriv[:, 1]
plt.figure()
fig, axes = plt.subplots(2, 1)
fig.suptitle(fileID[1])
l1, = axes[0].plot(x.ravel(), y.ravel(), 'k',
                   label='First Derivative', linewidth=2)
axes[0].set_xlabel("Magnetic Field [mT]", fontsize=14, weight='bold')
axes[0].set_ylabel("Intensity [a.u.]", fontsize=14, weight='bold')
# axes[0].legend(loc='upper right')
axes[0].set_title('First Derivative')
l2, = axes[1].plot(x2.ravel(), y2.ravel(), 'k',
                   label='First Derivative', linewidth=2)
axes[1].set_xlabel("Magnetic Field [mT]", fontsize=14, weight='bold')
axes[1].set_ylabel("Intensity [a.u.]", fontsize=14, weight='bold')
# axes[0].legend(loc='upper right')
axes[1].set_title('Zero Derivative')
figManager = plt.get_current_fig_manager()
figManager.window.showMaximized()
plt.subplots_adjust(left=0.046, right=0.976, top=0.945, bottom=0.085,
                    hspace=0.2, wspace=0.2)
plt.tight_layout()
makedirs(fileID[0] + '\\Figures\\', exist_ok=True)
plt.savefig(fileID[0] + '\\Figures\\figure{0}'.format(nfile))


# for i in range(len(ListOfFiles)):
#     fileID = path.split(ListOfFiles[i])
#     # plot the first derivative
#     x = FirstDeriv[:, 4*i]/10
#     y = FirstDeriv[:, 4*i+1]
#     x2 = ZeroDeriv[:, 4*i]/10
#     y2 = ZeroDeriv[:, 4*i+1]
#     plt.figure(i)
#     fig, axes = plt.subplots(2, 1)
#     fig.suptitle(fileID[1])
#     l1, = axes[0].plot(x.ravel(), y.ravel(), 'k',
#                        label='First Derivative', linewidth=2)
#     axes[0].set_xlabel("Magnetic Field [mT]", fontsize=14, weight='bold')
#     axes[0].set_ylabel("Intensity [a.u.]", fontsize=14, weight='bold')
#     #axes[0].legend(loc='upper right')
#     axes[0].set_title('First Derivative')
#     l2, = axes[1].plot(x2.ravel(), y2.ravel(), 'k',
#                        label='First Derivative', linewidth=2)
#     axes[1].set_xlabel("Magnetic Field [mT]", fontsize=14, weight='bold')
#     axes[1].set_ylabel("Intensity [a.u.]", fontsize=14, weight='bold')
#     #axes[0].legend(loc='upper right')
#     axes[1].set_title('Zero Derivative')
#     figManager = plt.get_current_fig_manager()
#     figManager.window.showMaximized()
#     plt.subplots_adjust(left=0.046, right=0.976, top=0.945, bottom=0.085,
#                         hspace=0.2, wspace=0.2)
#     plt.tight_layout()
#     makedirs(fileID[0] + '\\Figures\\', exist_ok=True)
#     plt.savefig(fileID[0] + '\\Figures\\figure{0}'.format(i+1))

del figManager, fileID, folder, l1, l2, x, x2, y, y2, axes, fig
