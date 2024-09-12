# -*- coding: utf-8 -*-
"""
Created on Tue Oct 24 16:41:33 2023
Test scripts for deer lab error.
@author: tim_t
"""

import numpy as np
import deerlab as dl

t = np.linspace(0, 1.5, 200)
r = np.linspace(2, 7, 200)

decay1 = 1
stretch1 = 3

mean1 = 2
std1 = 0.2

Pmodeltest = dl.dd_gauss
Bmodeltest = dl.bg_strexp
Bmodeltest1 = dl.bg_hom3d(t, conc=400, lam=20)  # this works

# Error is starting appearing here...
Pmodeltest2 = dl.dd_gauss(r, mean=mean1, std=std1)
Bmodeltest2 = dl.bg_strexp(t, decay=decay1, stretch=stretch1)
