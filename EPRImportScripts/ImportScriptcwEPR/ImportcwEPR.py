# -*- coding: utf-8 -*-
"""
Created on Tue Jan  2 12:10:59 2024

@author: tim_t
"""
from ImportMultipleFiles import ImportMultipleNameFiles, OpenMultipleFiles
import numpy as np
folder = 'D:\\Documents\\Recherches\\ResearchAssociatePosition\\Presentation'\
    '\\2024\\WorkshopFebruary\\DemoWorkshop\\'
ListOfFiles = ImportMultipleNameFiles(folder, Extension='.DSC')
fulldata, header = OpenMultipleFiles(
    ListOfFiles, Scaling=None, polyorder=2, window_length=100)
