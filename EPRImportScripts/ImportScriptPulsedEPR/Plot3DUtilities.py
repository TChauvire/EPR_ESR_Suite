# -*- coding: utf-8 -*-
"""
Function to plot in 3D
1) 3D plot with Mayavi (in time domain/frequency domain)
2) Contour plot with level selection option with matplotlib
(in time domain/frequency domain)
@author: tim_t
"""
from mayavi import mlab
from tvtk.api import tvtk
import matplotlib.pyplot as plt
import numpy as np
# mlab.options.backend = 'envisage'


def mayavi3D(x1=None, x2=None, z=None, mode='time', name=None):
    fig = mlab.figure(bgcolor=(1, 1, 1), fgcolor=(0, 0, 0))
    Z = z/np.max(z)
    fig.scene.interactor.interactor_style = tvtk.InteractorStyleTerrain()
    if mode == 'time':
        ranges = [0, np.round(x1.max(), 0), 0, np.round(x2.max(), 0), -1, 1]
        m = mlab.surf(x1, x2, Z, extent=ranges)
        # m.actor.actor.scale = (1.0, 1.0, np.max(x1)/10)
        ax = mlab.axes(line_width=2, nb_labels=5,
                       z_axis_visibility=True,
                       ranges=ranges)
        ax.axes.label_format = '%.1f'
        mlab.title('Time Domain: '+name, size=1, height=0.9)
        mlab.xlabel('t1 [us]')
        mlab.ylabel('t2 [us]')
        mlab.zlabel('Intensity (a.u.)')
    elif mode == 'freq':
        ranges = [np.round(x1.min(), -1), np.round(x1.max(), -1),
                  np.round(x2.min(), -1), np.round(x2.max(), -1), 0, np.max(z)]
        m = mlab.surf(x1, x2, z, warp_scale=10/(np.max(z)))
        m.actor.actor.scale = (1.0, 1.0, 2.0)
        ax = mlab.axes(line_width=2, nb_labels=5,
                       z_axis_visibility=False,
                       ranges=ranges)
        ax.axes.label_format = '%.0f'
        mlab.title('Frequency Domain: '+name, size=1, height=0.9)
        mlab.xlabel('f1 [MHz]')
        mlab.ylabel('f2 [MHz]')
        mlab.zlabel('Intensity [a.u.]')
    ax.label_text_property.font_family = 'courier'
    ax.label_text_property.font_size = 4
    return  # fig


def Contour3D(x1=None, x2=None, z=None, level=1, mode='time', name=None):
    X1, X2 = np.meshgrid(x1, x2)
    fig, ax = plt.subplots()
    if mode == 'time':
        curves = ax.contour(X1, X2, z, levels=8, colors='k',
                            linestyles='solid', linewidths=1)
        curves2 = ax.contourf(X1, X2, z, levels=50,
                              corner_mask=False, antialiased=False,
                              cmap='jet', alpha=.5)
        fig.suptitle(name, weight='bold', fontsize=18)
        ax.set_xlabel('t1 [us]', weight='bold', fontsize=16)
        ax.set_ylabel('t2 [us]', weight='bold', fontsize=16)
        ax.tick_params(axis='both', which='major', labelsize=14)
        ax.tick_params(axis='both', which='minor', labelsize=8)
        ax.grid(visible=True, which='both', axis='both')
        cbar = plt.colorbar(curves2, ax=ax, label='Intensity [a.u.]',
                            format=lambda x, _: f"{x:.0E}")
        #cbar.ax.set_yticklabels(["{:.1E}".format(i) for i in cbar.get_ticks()])
    elif mode == 'freq':
        curves = ax.contour(X1, X2, z, levels=12, colors='k',
                            linestyles='solid', linewidths=1)
        vmin = np.min(np.log(z))*level
        vmax = np.max(np.log(z))
        levels = np.linspace(vmin, vmax, 50)
        levels2 = np.linspace(vmin, vmax, 10)
        curves2 = ax.contourf(X1, X2, np.log(z), levels, vmin=vmin, vmax=vmax,
                              corner_mask=False, antialiased=False,
                              cmap='jet',  alpha=.5)
        curves2.cmap.set_under(color='w')
        curves2.set_clim(np.min(np.log(z))*level+0.1)
        fig.suptitle(name, weight='bold', fontsize=18)
        ax.set_xlabel('f1 [MHz]', weight='bold', fontsize=16)
        ax.set_ylabel('f2 [MHz]', weight='bold', fontsize=16)
        ax.tick_params(axis='both', which='major', labelsize=14)
        ax.tick_params(axis='both', which='minor', labelsize=8)
        ax.grid(visible=True, which='both', axis='both')
        cbar = plt.colorbar(curves2, ax=ax, ticks=levels2,
                            label='np.log(FFT) [a.u.]')
        cbar.ax.set_yticklabels(["{:.1f}".format(i) for i in cbar.get_ticks()])
    return
