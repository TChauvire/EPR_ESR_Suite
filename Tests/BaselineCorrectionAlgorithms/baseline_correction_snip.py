# -*- coding: utf-8 -*-
"""
Created on Tue Jul 21 14:56:29 2020

@author: TC229401
"""

def baseline_correction_snip(raman_spectra,niter):

    assert(isinstance(raman_spectra, pd.DataFrame)), 'Input must be pandas DataFrame'

    spectrum_points = len(raman_spectra.columns)
    raman_spectra_transformed = np.log(np.log(np.sqrt(raman_spectra +1)+1)+1)

    working_spectra = np.zeros(raman_spectra.shape)

    for pp in np.arange(1,niter+1):
        r1 = raman_spectra_transformed.iloc[:,pp:spectrum_points-pp]
        r2 = (np.roll(raman_spectra_transformed,-pp,axis=1)[:,pp:spectrum_points-pp] + np.roll(raman_spectra_transformed,pp,axis=1)[:,pp:spectrum_points-pp])/2
        working_spectra = np.minimum(r1,r2)
        raman_spectra_transformed.iloc[:,pp:spectrum_points-pp] = working_spectra

    baseline = (np.exp(np.exp(raman_spectra_transformed)-1)-1)**2 -1
    return baseline