# -*- coding: utf-8 -*-
"""
Created on Tue Oct 31 16:44:55 2023

@author: tim_t
"""
import matplotlib.pyplot as plt
import numpy as np
from copy import deepcopy
from os import path
from ImportDeerAnalysisParameters import BckgndSubtractionOfRealPart
from ImportDeerAnalysisParameters import DenoisingForOneFile
from ImportDeerAnalysisParameters import ExportOneAscii
from ImportDeerAnalysisParameters import ImportMultipleDEERAsciiFiles
from ImportMultipleFiles import datasmooth
import deerlab as dl
plt.close('all')
# File location in folder
folder = 'C:\\Users\\tim_t\\Python\\EPRImportScripts\\DataFilesDEER'
# Importing global path to files
ListOfFiles = ImportMultipleDEERAsciiFiles(FolderPath=folder, Extension='DSC')
# Initialize dictionnaries
DataDict, ParamDict = {}, {}
DataDict2, ParamDict2 = {}, {}

for j in range(len(ListOfFiles)):
    d, p, d2, p2 = {}, {}, {}, {}
    filename = ListOfFiles[j]
    # Import data and achieve baseline subtraction
    d, p = BckgndSubtractionOfRealPart(
        filename, d, p, Dimensionorder=1, Percent_tmax=19/20,
        mode='polyexp', truncsize=1/2)
    DataDict.update(deepcopy(d))
    ParamDict.update(deepcopy(p))
    ExportOneAscii(filename, d, p, mode='time')
    # Get the data denoised and get a more (precise?) background subtraction
    # on the denoised data
    d2, p2 = DenoisingForOneFile(filename, d, p, Dimensionorder=1,
                                 Percent_tmax=19/20, mode='polyexp',
                                 truncsize=1/2)
    DataDict2.update(deepcopy(d2))
    ParamDict2.update(deepcopy(p2))
    # Export the denoised data in ascii format to be used with the SVD analysis
    # with the https://denoising.cornell.edu/SVDReconstruction software.
    ExportOneAscii(filename, d2, p2, mode='timeden')

# Comparison of datasubtraction before and after denoising
for j in range(len(ListOfFiles)):
    filename = ListOfFiles[j]
    fileID = path.split(filename)
    fulldata1 = DataDict.get(fileID[1])
    Exp_parameter1 = ParamDict.get(fileID[1])
    itmin = Exp_parameter1.get('itmin')
    itmax = Exp_parameter1.get('itmax')
    time1 = fulldata1[:itmax, 0].ravel()
    y1 = fulldata1[:itmax, 5].ravel()
    fulldata2 = DataDict2.get(fileID[1])
    Exp_parameter2 = ParamDict2.get(fileID[1])
    itmin2 = Exp_parameter2.get('itmin')
    itmax2 = Exp_parameter2.get('itmax')
    time2 = fulldata2[:itmax2, 0].ravel()
    y2 = fulldata2[:itmax2, 5].ravel()
    # plt.figure(j)
    fig, axes = plt.subplots(2, 1)
    plt.suptitle(fileID[1])
    data = datasmooth(fulldata1[:, 2].real, window_length=4, method='binom')
    axes[0].plot(fulldata1[:, 0], data, 'k')
    axes[1].plot(time1, y1, 'k', time2, y2, 'r')
