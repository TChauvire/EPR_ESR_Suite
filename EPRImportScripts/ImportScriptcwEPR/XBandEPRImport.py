# -*- coding: utf-8 -*-
"""
Created on Fri Nov 10 12:41:25 2023

Import Multiple cw-ESR files
"""

from datasmooth import *
from ImportMultipleFiles import ImportMultipleNameFiles, OpenMultipleFiles, eprload
from basecorr1D import *

import matplotlib.pyplot as plt
import numpy as np
#############################################################################
folder = 'D:\\Documents\\Recherches\\ResearchAssociatePosition\\Data\\cwEPR\\'\
    '2023\\2023_rebecca\\101\\DataToPlot'
fulldata, Header = [], []
ListOfFiles = ImportMultipleNameFiles(FolderPath=folder, Extension='DSC')
fulldata, Header = OpenMultipleFiles(ListOfFiles=ListOfFiles, Scaling=None,
                                     polyorder=1, window=100)
DataForGFactor = np.full((len(ListOfFiles), 6), np.nan)


# Script part for evauating peaks and center field for g-factor measurement for
# a simple loretzian shape EPR spectrum

name = [str()]*len(ListOfFiles)
npts = fulldata.shape[0]
for i in range(len(ListOfFiles)):
    _, _, par = eprload(ListOfFiles[i])
    x = fulldata[0:npts, 4*i]
    data = fulldata[0:npts, 4*i+3]
    name[i] = par['TITL']
    DataForGFactor[i, 0] = par['MWFQ']*1E-9
    DataForGFactor[i, 1] = x[np.argmax(data)]
    DataForGFactor[i, 2] = x[np.argmin(data)]
    center = x[np.argmax(data)] + (x[np.argmin(data)]-x[np.argmax(data)])/2
    DataForGFactor[i, 3] = center
    i1 = int(np.floor(3*npts/8))
    i2 = int(np.floor(5*npts/8))
    x2 = x[i1:i2]
    j = 1
    while len(x2[np.abs(data[i1:i2]) <= 1e-3*j]) == 0:
        j += 1
    value = np.squeeze(x2[np.abs(data[i1:i2]) <= 1e-3*j])
    DataForGFactor[i, 4] = value
    DataForGFactor[i, 5] = np.mean([center, value])
