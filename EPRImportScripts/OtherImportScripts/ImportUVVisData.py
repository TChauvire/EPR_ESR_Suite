# -*- coding: utf-8 -*-
"""
Created on Fri Dec  1 15:13:31 2023

Script to import multiple UV-Vis data in once.
The data are export as Fulldata and data-header.
Fulldata are compiled to be a matrix with column vectors.
The first column is the wavelength in nm,
The second column is the uncorrected absorbance
The third column is the baseline corrected absorbance by taking the last
200 points
The fourth is the 2pts binomial smoothed data.
The fifth is the standard deviation of the data

@author: tim_t
"""
from ImportMultipleFiles import ImportMultipleNameFiles, datasmooth
from csv import reader
import numpy as np
from os import path

# plt.close('all')
# File location in folder
folder = 'D:\\Documents\\Recherches\\ResearchAssociatePosition\\'\
    'Data\\Rebecca_Project\\UVVis\\20231201\\'
ListOfFiles = ImportMultipleNameFiles(FolderPath=folder, Extension='.CSV')
# Check for the maximum length datafiles
for file in ListOfFiles:
    list1 = []
    maxlen = 0
    with open(file, 'r', newline='', encoding='utf-16') as f:
        data = reader(f, delimiter=",", dialect='excel')
        for row in data:
            list1.append(row[0])
    if maxlen < len(list1):
        maxlen = len(list1)
# Initialize the variables
fulldata = np.full((maxlen, len(ListOfFiles)*5), np.nan, dtype=float)
header = [None] * len(ListOfFiles)*5
for i in range(len(ListOfFiles)):
    file = ListOfFiles[i]
    filename = path.split(file)[1]
    x, y, std = [], [], []
    with open(file, 'r', newline='', encoding='utf-16') as f:
        data = reader(f, delimiter=",", dialect='excel')
        next(data)
        for row in data:
            x.append(float(row[0]))
            y.append(float(row[1]))
            std.append(float(row[2]))
    npts = len(x)
    fulldata[0:npts, 5*i] = x
    fulldata[0:npts, 5*i+1] = y
    ycorr = y-np.mean(y[-100:], axis=0)
    fulldata[0:npts, 5*i+2] = ycorr
    fulldata[0:npts, 5*i+3] = datasmooth(y=ycorr, window_length=2,
                                         method='binom')
    fulldata[0:npts, 5*i+4] = std
    header[5*i] = 'wavelength [nm]'
    header[5*i+1] = filename
    header[5*i+2] = str(filename + '_corrected')
    header[5*i+3] = str(filename + '_smoothed')
    header[5*i+4] = str(filename + '_std')
