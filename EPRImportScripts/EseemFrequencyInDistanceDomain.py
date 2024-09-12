# -*- coding: utf-8 -*-
"""
Created on Tue Dec  5 15:32:51 2023
Generate Sinusoidal ESEEEM for 4P-DEER analysis
@author: tim_t
"""
import numpy as np
import deerlab as dl
import matplotlib.pyplot as plt

freq14N_1 = 3
freq14N_2 = 6
freq15N = 4
freq0 = 0.2
freq2H = 7.5
freq = [freq14N_1, freq15N, freq14N_2, freq0, freq2H]

for i in range(len(freq)):
    # Generate sinusoidal for Tikhonov regularization
    # i=3
    Time = np.arange(0, 10, 0.01)
    V = 1+0.2*np.sin(2*np.pi*freq[i]*Time)*np.exp(-Time/1)
    tau1 = 300/1000  # First inter-pulse delay, us
    tau2 = 5000/1000
    # Distance vector
    r = dl.distancerange(Time, Time.shape[0])
    Tnew = dl.correctzerotime(V, Time)
    t = Tnew+tau1
    # Construct the model
    exp = dl.ex_4pdeer(tau1, tau2, pathways=[1])
    Vmodel = dl.dipolarmodel(t, r, Pmodel=None, Bmodel=None, experiment=exp)

    # Fit the model to the data
    results = dl.fit(Vmodel, V, penalties=None, bootstrap=0, noiselvl=None,
                     mask=None, weights=None, regparam='aic', reg=True)
    s = results.model
    P = results.P

    fig, axes = plt.subplots(1, 2)
    fig.suptitle(
        "Test Influence of {0} MHz Eseem frequency on the distance domain".format(freq[i]))
    # https://matplotlib.org/stable/api/axes_api.html
    l1, = axes[0].plot(
        Tnew, V, 'k', label='SimulatedEseemFrequency', linewidth=1)
    l2, = axes[0].plot(Tnew, s, 'r',
                       label='Tikhonov Regularization', linewidth=1)

    axes[0].grid()
    axes[0].set_xlabel("Time Domain [us]")
    axes[0].set_ylabel("S(t)")
    axes[0].legend()

    l1, = axes[1].plot(r, P, 'k', label='Tikhonov Regularization', linewidth=1)
    axes[1].grid()
    axes[1].set_ylabel("P(r)")
    axes[1].set_xlabel("Distance Domain [nm]")
    #axes[1].set_title('Criterium GCV')

    axes[1].legend()
    plt.tight_layout()

    # plt.plot(NewTime, V14N_1)
