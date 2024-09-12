# -*- coding: utf-8 -*-
"""
Created on Tue Jan  2 12:10:59 2024

@author: tim_t
"""
from ImportMultipleFiles import ImportMultipleNameFiles, MaxLengthOfFiles, OpenMultipleComplexFiles2
import numpy as np
folder = 'D:\\Documents\\Recherches\\ResearchAssociatePosition\\Data\\'\
    'PulsedEPR\\2024\\240103\\110K\\FSE'

ListOfFiles = ImportMultipleNameFiles(folder, Extension='.DSC')
maxlen = MaxLengthOfFiles(ListOfFiles)
fulldata, header = OpenMultipleComplexFiles2(
    ListOfFiles, Scaling=None, polyorder=1, window_length=50)
