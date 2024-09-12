# -*- coding: utf-8 -*-
"""
Created on Mon Nov  6 12:21:29 2023

@author: tim_t
"""
from ImportDeerAnalysisParameters import ReadDenoisedData
import matplotlib.pyplot as plt
import numpy as np
filename = "011619_AerCheACheW_Bl21_4PDeer_2500ns_150K_db6_denoised.dat"
path = "C:\\Users\\tim_t\\Python\\EPRImportScripts\\DataFilesDEER\\AsciiDenoisedFiles\\"
fullpath = path + filename

y2, x2, header2 = ReadDenoisedData(fullpath)

Header = {}
x, y = [], []
with open(fullpath, 'r') as f:
    lines = f.readlines()
    for i in range(len(lines)):
        if lines[i][0] == '%':
            pairs = lines[i].split('\t')
            Header.update({str(i): pairs})
        else:
            pairs = lines[i].split('\t')
            x.append(float(pairs[0]))
            y.append(float(pairs[1])+1j*float(pairs[2]))
