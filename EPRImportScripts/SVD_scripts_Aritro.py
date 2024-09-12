# -*- coding: utf-8 -*-
"""
Created on Fri Oct 27 13:05:09 2023

SVD regularization suite in Python used for the data treatment of Dipolar

This set of packages was initially written under Matlab by Maddhur Srivastava
(ms2736@cornell.edu). It was then adapted by Chris Myer, staff at
CAC at Cornell (http://www.cac.cornell.edu/).
So the original software working with a Bokh graphical user interface is found
at https://github.com/CornellCAC/denoising.
And you may want the access to the orginal more advanced software hosted at
this address : https://denoising.cornell.edu/

For more information about the techniques, check the different paper from
the litterature:
    - M. Srivastava, J. H. Freed, J. Phys. Chem. Lett. 2017, 8, 5648-5655
    (dx.doi.org/10.1021/acs.jpclett.7b02379)
    - Srivastava, M.; Freed, J.H, J. Phys. Chem. A. 2019, 123(1), 359-370
    (dx.doi.org/10.1021/acs.jpca.8b07673)
    - Y-W. Chiang, P. B. Borbat, J. H. Freed, J. Mag. Res. 2005, 172, 279-295
    (dx.doi.org/10.1016/j.jmr.2004.10.012)
    - Y-W. Chiang, P. B. Borbat, J. H. Freed, J. Mag. Res. 177 (2005) 184–196
    (dx.doi.org/10.1016/j.jmr.2005.07.021)

@author: Timothee Chauvire (tsc84@cornell.edu)
(https://github.com/TChauvire/EPR_ESR_Suite)
"""
import numpy as np
import matplotlib.pyplot as plt
import numpy.polynomial.polynomial as pl
from scipy.signal import find_peaks, peak_widths
from scipy.optimize import curve_fit
from os import path
from numpy.polynomial.polynomial import polyfit, polyvander, polyval
from eprload_BrukerBES3T import eprload_BrukerBES3T
from eprload_BrukerESP import eprload_BrukerESP
from automatic_phase import automatic_phase
# import scipy.interpolate
# import scipy.optimize


def process_data(x_x, x_y, K, U, s, V, rmin, rmax):
    # (rmin, rmax) = rminrmaxcalculus(x_x)
    sigma = s
    M = len(x_y)
    N = M
    ttime = x_x.max()
    # rrscale = rscale(x_x)
    # process_breakpoints()  # Find a work around here
    Pr, Pic, sum_Pic = get_Pr_et_al(x_y, U, s, V)
    iregions = [(0, int(M))]
    n_lambda = len(iregions)
    loc = [4, 13]
    seg_range = []
    # eig_cutoff = []
    # loc = []
    # % Singular value cut-off assignment to each segment
    # for i in range(0, n_lambda):
    #    eig_cutoff.append(thresholds[i])  # input('Insert cutoff value for eigen values:- ');
    #    loc.append(np.where(sigma >= eig_cutoff[i])[0][-1])
    # svc_Pr = loc[:]
    # loc = svc_Pr  # find out what's happen here
    # % Create an empty variable
    # iregions = (,)
    # K = get_kernel(M, N, ttime, rmin, rmax)
    # for k in range(N):
    #     S_r[:, k] = K*Pr[k, :].T
    PR = []
    for i in range(0, n_lambda):
        PR.append(Pr[loc[i], iregions[i][0]:iregions[i][1]])
    PR = np.concatenate(tuple(PR))
    Picard = (Pr**2).T
    return x_y, sigma, PR, Pr, Picard, sum_Pic


def subplotSVD(x, y, sigma, Pr, sum_Pic):
    fig, axes = plt.subplots(2, 2)
    fig.suptitle("Test SVD Reconstruction")

    r = rscale(x)
    # https://matplotlib.org/stable/api/axes_api.html
    axes[0, 0].plot(x, y)
    axes[0, 0].set_xlabel("Time [us]")
    axes[0, 0].grid()
    axes[0, 0].set_ylabel("Intensity (a.u.)")
    axes[0, 0].set_title('Time Domain')
    # axes[0].legend()

    axes[0, 1].plot(sigma)
    axes[0, 1].set_yscale('log')
    axes[0, 1].grid()
    axes[0, 1].set_ylabel("Singular Values")
    axes[0, 1].set_xlabel("SVC number")
    axes[0, 1].set_title('Singular Values')

    axes[1, 0].plot(r, Pr.T[:, 20])
    axes[1, 0].grid()
    axes[1, 0].set_ylabel("P(r)")
    axes[1, 0].set_xlabel("Distance r [nm]")
    axes[1, 0].set_title('Distance Domain')

    axes[1, 1].plot(sum_Pic)
    axes[1, 1].set_yscale('log')
    axes[1, 1].grid()
    axes[1, 1].set_ylabel("Picard_summed")
    axes[1, 1].set_xlabel("SVC number")
    axes[1, 1].set_title('Pickard plot')

    plt.tight_layout()
    return


def ukernel(pr, ti):
    # the distance scale pr has to be in nanometer
    # the time ti scale in us
    sz = 500  # len(pr)
    ipr3ti = np.outer(1./(pr**3), ti)
    dTheta = (np.arange(1, sz+1)-0.5) * np.pi/sz
    # line below specify is the kernel for nanometer scales
    usolt = (np.cos(2*np.pi*5.20356e1*ipr3ti[:, :, np.newaxis]*(
        1-3*(np.cos(dTheta))**2))*np.sin(dTheta)) * (dTheta[1]-dTheta[0])
    usol = np.sum(usolt, axis=-1)/2.
    return usol.T


def get_kernel(t, rmin, rmax):
    # M and N are the same size by construction here...
    M = len(t)
    N = len(t)
    # (rmin, rmax) = rminrmaxcalculus(t)
    pr = rscale(t, rmin, rmax)  # distance in nanometer
    ddist = (rmax-rmin)  # distance distri. range in nanometer
    # Initialization.
    h = t.max()/M
    h2 = ddist/N
    # Compute the Kernel K
    ti = (np.arange(1, M+1)-0.5)*h
    K = h2 * ukernel(pr, ti)
    return K


def get_KUsV(t, rmin, rmax):
    K = get_kernel(t, rmin, rmax)
    U, s, Vh = np.linalg.svd(K)
    V = Vh.T
    return K, U, s, V


def get_Pr_et_al(S, U, s, V):
    M = len(S)
    N = M
    P = np.zeros((N, M))
    print(S.shape, U.shape, s.shape, V.shape)
    Pic = np.zeros(N)
    sum_Pic = np.zeros(N)
    Pr = np.zeros((N, M))

    for i in range(N):
        P[i, :] = (np.dot(U[:, i].T, S) / s[i]) * V[:, i]
        Pic[i] = (np.dot(U[:, i].T, S)**2) / (s[i]**2)
        sum_Pic[i] = np.sum(Pic)

    for k in range(N):
        Pr[k, :] = np.sum(P[0:k+1, :], axis=0)

    return Pr, Pic, sum_Pic


def reconstruct_dipolar_signal(K, PR):
    rsignal = np.dot(K, PR)
    return rsignal


def PR_uncertainty(Pr, svc_umins, svc_umaxs, svc_Pr, iregions):
    # I never used this part of the script. I have no idea if it's works
    PRmin = []
    PRmax = []
    for region in range(len(iregions)):
        cur_iregion = iregions[region]
        y0 = Pr[svc_Pr[region], iregions[region][0]:iregions[region][1]]
        ymaxs = Pr[svc_umins[region]:svc_umaxs[region] +
                   1, cur_iregion[0]:cur_iregion[1]].max(axis=0)
        ymins = Pr[svc_umins[region]:svc_umaxs[region] +
                   1, cur_iregion[0]:cur_iregion[1]].min(axis=0)
        # for P(r) >=0, clip min uncertainty at y=0.; otherwise, leave untouched
        ymins = np.where((y0 >= 0.) * (ymins < 0.), 0., ymins)
        PRmin.append(ymins)
        PRmax.append(ymaxs)
    PRmin = np.concatenate(tuple(PRmin))
    PRmax = np.concatenate(tuple(PRmax))
    return PRmin, PRmax


def rminrmaxcalculus(t):
    r""" 
    Empirical distance range given a dipolar EPR experiment time axis

    This function calculates the empirical distance range for a DEER time axis. 
    The distance range is determined by the time step and the Nyquist criterion 
    for the minimum distance, and by the requirement that at least half an 
    oscillation should be observable over the measured time window for the 
    maximum distance. The function allows to specify the length of the output 
    distance axis. If not given, only the minimum and maximum distances are 
    returned.

    Parameters
    ----------
    t : array_like
        Time axis, in microseconds. The time points at which the dipolar signal 
        was measured.
    Returns
    -------
    r : tuple ``(rmin,rmax)``.

    Notes
    -----
    The minimal and maximal distances, ``rmin`` and ``rmax``, are empirical 
    values that determine the minimal and maximal distance for which the given 
    time trace can provide reliable information.
    The minimum distance is determined by the time step :math:`\Delta t` 
    and the Nyquist criterion:

    .. math:: 

        r_\text{min} = \left( \frac{4\Delta t \nu_0}{0.85} \right)^{1/3}

    The maximum distance is determined by the requirement that at least half 
    an oscillation     should be observable over the measured time window from 
    :math:`t_\text{min}` to :math:`t_\text{max}`.

    .. math:: 

        r_\text{max} = 6\left( \frac{t_\text{max}-t_\text{min}}{2} \right)^{1/3}

    where :math:`\nu_0` = 52.04 MHz nm^3.
    See Jeschke et al, Appl. Magn. Reson. 30, 473-498 (2006),
    https://doi.org/10.1007/BF03166213 and 
    https://github.com/JeschkeLab/DeerLab/blob/main/deerlab/distancerange.py

    """
    t = np.atleast_1d(t)
    D = 52.0356  # MHz nm^3
    # Minimum distance is determined by maximum frequency detectable with
    # the given time increment, based on Nyquist
    dt = np.mean(np.diff(t))  # time increment
    nupara_max = 1/2/dt  # maximum parallel dipolar frequency (Nyquist)
    nu_max = nupara_max/2  # maximum perpendicular dipolar frequency
    nu_max = nu_max*0.85  # add a bit of buffer
    rmin = (D/nu_max)**(1/3)  # length in nanometer
    # At least half a period of the oscillation
    # should be observable in the time window.
    trange = np.max(t) - np.min(t)
    Tmax = trange*2  # maximum period length
    rmax = (D*Tmax)**(1/3)  # length in nanometer
    return (rmin, rmax)  # distance in nanometer


def rscale(t, rmin, rmax):
    # (rmin, rmax) = rminrmaxcalculus(t)
    ddist = rmax-rmin
    npts = len(t)
    h2 = ddist/npts
    rrscale = (np.arange(1, npts+1)-0.5)*h2+rmin
    return rrscale  # distance in nanometer
# thresholds = [sigma[t] for t in svc_Pr]
# svc_Pr = [0]


def ExponentialCorr1D_DEER(x=None, y=None, Dimensionorder=3, Percent_tmax=9/10,
                           mode='strexp', truncsize=3/4, *args, **kwargs):
    '''
    Function that achieve a baseline correction by fitting a function
    parameterized by a streched exponential for a supposed homogeneous
    three-dimensional solution (d=3) or a stretched exponential for other
    dimensions as described by multiple publications.
    See by example : (dx.doi.org/10.1039/c9cp06111h )

    .. math::

    B(t) = \exp\left(-\kappa \vert t\vert^{d}\right)

    k is the decay rate constant of the background and d is
    the so-called fractal dimension of the exponential

    The fitting is done on the last 3/4 points of the data.
    Script written by Timothée Chauviré
    (https://github.com/TChauvire/EPR_ESR_Suite/), 10/18/2023

    Parameters
    ----------
    x : abscissa of the data, TYPE : numpy data array, column vector
        DESCRIPTION. The default is None.
    y : data which baseline has to be corrected,
        TYPE : numpy data array, column vector
        It has to have the same size than x
        DESCRIPTION. The default is None.
    Dimensionorder : order of so-called fractal dimension of the exponential
        TYPE : Integer, optional
        DESCRIPTION. The default is 0.

    Returns
    -------
    ynew : baseline data array
        TYPE: numpy data array, same shape as input data y
    (k,d) : coefficient used for the exponential fit
        TYPE : tuple of real values, coefficient of the streched exponential
    perr : error coefficient obtained from the covariance matrix
        (perr = np.sqrt(np.diag(pcov)))
        TYPE : diagonalized 2-D array
    mode='strexp', stretched exponential ackground subtraction
    mode='poly', polynomial fit of the logarithmic data
    '''
    shape = y.shape
    if x.shape[0] != np.ravel(x).shape[0]:
        raise ValueError('x must be a column vector. ExponentialCorr1D_DEER'
                         'function does not work on 2D arrays.')
    else:
        x = np.ravel(x)

    if y.shape[0] != np.ravel(y).shape[0]:
        raise ValueError('y must be a column vector. ExponentialCorr1D_DEER'
                         'function does not work on 2D arrays.')
    else:
        y = np.ravel(y)

    if y.shape[0] != x.shape[0]:
        raise ValueError('x and y must be column vector of the same size.')
    yfit = np.full(y.shape, np.nan)
    npts = x.shape[0]
    npts_new = int(np.floor(npts*truncsize))
    itmax = int(np.floor(Percent_tmax*npts))
    xfitinit = (np.array(x[(npts-npts_new):itmax])).ravel()
    # xfitinit = np.array(x[-npts_new:tmax]).ravel()
    yfit = (y/np.max(y)).ravel().real
    yfitinit = (np.array(yfit[(npts-npts_new):itmax])).ravel()
    # yfitinit = np.array(yfit[-npts_new:tmax]).ravel()

    def strexp(x, ModDepth, decay, stretch):
        a = (1-ModDepth)*(np.exp((-1)*(np.abs(decay*x))) ** (stretch/3))
        return a

    def strexp2(x, ModDepth, decay):
        a = (1-ModDepth)*(np.exp((-1)*(np.abs(decay*x))) **
                          (Dimensionorder/3))
        return a

    # # Add parameters
    p0_1 = [0.3, 0.25, Dimensionorder]
    b_1 = ([0, 0, 2], [1, 200, 6])  # (lowerbound,upperbound)  bounds=b,
    # Add parameters
    p0_2 = [0.3, 0.25]
    b_2 = ([0, 0], [1, 200])  # (lowerbound,upperbound)  bounds=b,

    if mode == 'strexp':
        poptarray, pcov = curve_fit(strexp, x[:itmax], yfit[:itmax], p0=p0_1,
                                    sigma=None, absolute_sigma=False,
                                    check_finite=None, bounds=b_1)
        perr = np.sqrt(np.diag(pcov))
        yfit2 = (strexp(x, poptarray[0],
                 poptarray[1], poptarray[2]))*np.max(y)
        yfit2 = yfit2.reshape(shape)
        return yfit2, poptarray, perr
    if mode == 'strexp_fixed':
        poptarray, pcov = curve_fit(strexp2, x[:itmax], yfit[:itmax], p0=p0_2,
                                    sigma=None, absolute_sigma=False,
                                    check_finite=None, bounds=b_2)
        perr = np.sqrt(np.diag(pcov))
        yfit2 = (strexp2(x, poptarray[0], poptarray[1]))*np.max(y)
        poptarray = np.append(poptarray, Dimensionorder)
        yfit2 = yfit2.reshape(shape)
        return yfit2, poptarray, perr
    if mode == 'poly':
        c, stats = pl.polyfit(xfitinit, yfitinit,
                              deg=Dimensionorder, full=True)
        ypoly = pl.polyval(x, c)*np.max(y)
        error_parameters, _ = error_vandermonde(x, residuals=stats[0], rank=3)
        return ypoly, c, error_parameters
    if mode == 'polyexp':
        c, stats = pl.polyfit(xfitinit, np.log(yfitinit),
                              deg=Dimensionorder, full=True)
        ypoly = np.exp(pl.polyval(x, c))*np.max(y)
        error_parameters, _ = error_vandermonde(x, residuals=stats[0], rank=3)
        return ypoly, c, error_parameters


def BckgndSubtractionOfRealPart(Filename, DataDict, ParamDict, Scaling=None,
                                Dimensionorder=3, Percent_tmax=9/10,
                                mode='strexp', truncsize=3/4, *args, **kwargs):
    '''

    ----------
    ListOfFiles : TYPE
        DESCRIPTION.
    Scaling : TYPE, optional
        DESCRIPTION. The default is None.
    Dimensionorder : TYPE, optional
        DESCRIPTION. The default is 1.
    Percent_tmax : TYPE, optional
        DESCRIPTION. The default is 9/10.
    *args : TYPE
        DESCRIPTION.
    **kwargs : TYPE
        DESCRIPTION.

    Raises
    ------
    ValueError
        DESCRIPTION.

    Returns
    -------
    BckgndSubtractedData : TYPE
        DESCRIPTION.
    Modulation_Depth : TYPE
        DESCRIPTION.
    RelBckgndDecay : TYPE
        DESCRIPTION.
    NoiseLevel : TYPE
        DESCRIPTION.

    '''
    fileID = path.split(Filename)
    data, abscissa, par = eprload(Filename, Scaling)
    if (data.shape[0] != np.ravel(data).shape[0]):
        raise ValueError(
            'The file {0} is\'t a column vector'.format(par['TITL']))
    else:
        npts = abscissa.shape[0]
        new_data = np.full(npts, np.nan, dtype="complex_")
        pivot = int(np.floor(data.shape[0]/2))
        new_data, _ = automatic_phase(vector=data, pivot1=pivot,
                                      funcmodel='minfunc')
        data_real = np.ravel(new_data.real)
        data_imag = np.ravel(new_data.imag)
        abscissa = np.ravel(abscissa)
        # Achieve background correction of the real part :
        # newdata_real = datasmooth(
        #     data_real[0:npts], window_length=10, method='binom')
        # itmin = np.argmax(newdata_real)
        # newx = dl.correctzerotime(data_real, abscissa)
        newx = abscissa
        itmin = np.abs(newx).argmin()
        # itmin = newx([newx==0])
        if itmin > 50:  # The determination of zerotime didn't work, do nothing
            itmin = 0
            newx = abscissa
        tmin = newx[itmin]
        data_bckgnd, p0, perr = ExponentialCorr1D_DEER(x=newx, y=data_real,
                                                       Dimensionorder=Dimensionorder,
                                                       Percent_tmax=Percent_tmax,
                                                       mode=mode, truncsize=truncsize)
        w = int(np.floor(npts/2))
        # Achieve automatic base line correction correction of the imaginary
        # part :
        data_imag_new, _, _, _ = basecorr1D(x=newx, y=data_imag,
                                            polyorder=1, window=w)
        if np.floor(Percent_tmax*npts)-1 <= npts:
            itmax = int(np.floor(Percent_tmax*npts)-1)
        else:
            itmax = npts
        RelBckgndDecay = 1 - data_bckgnd[itmax] / data_bckgnd[itmin]
        FinalData = (data_real - data_bckgnd)/np.max(data_real)
        FinalData = FinalData-FinalData[itmax-20:itmax].mean()
        # BckgndSubtractedData = (data_real - data_bckgnd)/(np.max(data_real - data_bckgnd))
        # Two noises-level are computed :
        # 1) With the full imaginary part "sigma_noise"
        # 2) With half of the imaginary part "sigma_noise_half"
        center = int(np.floor(npts/2))
        sigma_noise = np.std(data_imag_new[itmin:itmax,])/np.max(data_real)
        sigma_noise_half = np.std(data_imag_new[center-int(np.floor(npts/4)):
                                                center+int(np.floor(npts/4))],
                                  )/np.max(data_real)
        NoiseLevel = (sigma_noise, sigma_noise_half)
        # Calculate the Root mean square of error
        RMSE = ComputeRMSE(data_real/np.max(data_real),
                           data_bckgnd/np.max(data_real), p0)
        # Let's create a global dictionnary for storing all the data and the
        # parameters:
        # TITL = str(par['TITL'])
        TITL = fileID[1]
        fulldata = np.full((5*npts, 50), np.nan, dtype="complex_")
        fulldata[0:npts, 0] = newx.ravel()
        fulldata[0:npts, 1] = data.ravel()
        fulldata[0:npts, 2] = new_data.ravel()
        fulldata[0:npts, 3] = new_data.real.ravel()
        fulldata[0:npts, 4] = data_bckgnd.ravel()
        fulldata[0:npts, 5] = FinalData.ravel()

        DataDict.update({TITL: fulldata})
        Header = list(np.zeros((50,)))
        Header[0] = str(par['XNAM'])
        Header[1] = str(TITL+"_rawData")
        Header[2] = str(TITL+"_phased")
        Header[3] = str(TITL+"_phased_real")
        Header[4] = str(TITL+"_background_real")
        Header[5] = str(TITL+"_backgroundsubtracted_real")
        HeaderTITL = str(TITL+'_Header')
        DataDict[HeaderTITL] = Header
        Exp_parameter = {'RelBckgndDecay': RelBckgndDecay, 'tmin':  tmin,
                         'NoiseLevel': NoiseLevel, 'tmax': abscissa[itmax],
                         'itmax': itmax, 'itmin': itmin, 'RMSE': RMSE}
        # Assign the modulation depth in the parameters
        if mode == 'strexp':
            Bckg_type = str(mode+'_'+str(Percent_tmax)+'_'+str(truncsize))
            DEER_SNR = (p0[0]/sigma_noise, p0[0]/sigma_noise_half)
            Exp_parameter.update({'ModDepth': p0[0], 'decay': p0[1],
                                  'stretch': p0[2], 'Bckg_type': Bckg_type,
                                  'DEER_SNR': DEER_SNR})

        if mode == 'strexp_fixed':
            Bckg_type = str(mode+'_'+str(Percent_tmax)+'_'+str(truncsize))
            DEER_SNR = (p0[0]/sigma_noise, p0[0]/sigma_noise_half)
            Exp_parameter.update({'ModDepth': p0[0], 'decay': p0[1],
                                  'stretch': p0[2], 'Bckg_type': Bckg_type,
                                  'DEER_SNR': DEER_SNR})
        elif mode == 'poly':
            Bckg_type = str(mode+str(len(p0)-1)+'_'+str(Percent_tmax)+'_' +
                            str(truncsize))
            Mod_Depth = FinalData[itmin-1:itmin+1].mean()
            DEER_SNR = (Mod_Depth/sigma_noise, Mod_Depth/sigma_noise_half)
            Exp_parameter.update({'ModDepth': Mod_Depth, 'polyparam': p0,
                                  'Bckg_type': Bckg_type,
                                  'DEER_SNR': DEER_SNR})
        elif mode == 'polyexp':
            Bckg_type = str(mode+str(len(p0)-1)+'_'+str(Percent_tmax)+'_' +
                            str(truncsize))
            Mod_Depth = FinalData[itmin-1:itmin+1].mean()
            DEER_SNR = (Mod_Depth/sigma_noise, Mod_Depth/sigma_noise_half)
            Exp_parameter.update({'ModDepth': Mod_Depth, 'polyparam': p0,
                                  'Bckg_type': Bckg_type,
                                  'DEER_SNR': DEER_SNR})
        ParamDict.update({TITL: Exp_parameter})
    return DataDict, ParamDict


def basecorr1D(x=None, y=None, polyorder=0, window=20, *args, **kwargs):
    '''
    Function that achieve a baseline correction by fitting a function a the 
    edge of the data via a window parameter.
    The baseline correction is carried out via the polyfit function:
        (https://numpy.org/doc/stable/reference/generated/numpy.polynomial.polynomial.polyfit.html)
    that Compute least-squares solution to equation V(x) * c = w * y, with V the
    Vandermonde matrix, c the polynomial coefficient, w some weights parameters
    and y the data column vector.

    p(x) = c_0 + c_1 * x + ... + c_n * x^n,

    This script is freely inspired by the easyspin suite from the Stefan Stoll lab
    (https://github.com/StollLab/EasySpin/)
    (https://easyspin.org/easyspin/)

    Script written by Timothée Chauviré (https://github.com/TChauvire/EPR_ESR_Suite/), 09/09/2020

    Parameters
    ----------
    x : abscissa of the data, TYPE : numpy data array, column vector
        DESCRIPTION. The default is None.
    y : data which baseline has to be corrected, 
        TYPE : numpy data array, column vector
        It has to have the same size than x
        DESCRIPTION. The default is None.
    polyorder : order of the polynomial fitted for background subtraction
        TYPE : Integer, optional
        DESCRIPTION. The default is 0.
    window : window used at the start and the end of the data. TYPE: integer, optional
        DESCRIPTION. The default is 20.
        By instance, if window = 20, the fit of the background will be achieved on 
        np.array([y[:20],y[-20:]]).ravel()

    Returns
    -------
    ynew : baseline corrected data. 
        TYPE: numpy data array, same shape as input data y
    c : coefficient used for the polynomial fit as returned by the 
        numpy.polynomial.polynomial.polyfit function
        TYPE : tuple of real values, coefficient of the polynomial from low order to high order rank.
    error_parameters : error associated with the polynomial coefficients, 
        TYPE : tuple of real values, 
        DESCRIPTION: This error is evaluated via the function error_vandermonde() 
        by calculating the covariance matrix and diagonalize it.
    cov : Covariance matrix associated to the Vandermonde matrix of the fitted poynomial. 
        TYPE : Square Matrix 

    '''
    shape = y.shape
    if x.shape[0] != np.ravel(x).shape[0]:
        raise ValueError(
            'x must be a column vector. basecorr1D function does not work on 2D arrays.')
    else:
        x = np.ravel(x)

    if y.shape[0] != np.ravel(y).shape[0]:
        raise ValueError(
            'y must be a column vector. basecorr1D function does not work on 2D arrays.')
    else:
        y = np.ravel(y)

    if y.shape[0] != x.shape[0]:
        raise ValueError('x and y must be column vector of the same size.')

    ynew = np.full(y.shape, np.nan)
    xfit = np.array([x[:window], x[-window:]]).ravel()
    yfit = np.array([y[:window], y[-window:]]).ravel()
    c, stats = polyfit(xfit, yfit, polyorder, full=True)
    ynew = y-polyval(x, c)
    error_parameters, cov = error_vandermonde(
        x, residuals=stats[0], rank=polyorder)
    ynew = ynew.reshape(shape)
    return ynew, c, error_parameters, cov


def error_vandermonde(x, residuals=None, rank=None, *args, **kwargs):
    '''
    Function to generate 1) error estimation on parameters determined by the function
    coef, [residuals, rank, singular_values, rcond] = numpy.polynomial.polynomial.polyfit()
    2) covariance matrix associated.

    The Vandermonde matrix generated by vand = polyvander(x,rank)
    The Covariance matrix cov is obtained via the vandermonde matrix V 
    via this numerical steps:
        1) compute np.dot(V.T,V).inv 
        2) and multiply it by the residual/(nb of data points - nb of coefficients)
    The error parameters are then computed via : error_parameters = np.sqrt(np.diag(cov))

    Script written by Timothée Chauviré (https://github.com/TChauvire/EPR_ESR_Suite/), 09/09/2020

    Parameters
    ----------
    residuals : first value generated by polyfit in the list of the second output
        DESCRIPTION. float number. The default is None.
    vandermonde : Vandermonde matrix generated by vand = polyvander(x,rank)
        DESCRIPTION. The default is None.

    rank : necessary for multidimensional array
        DESCRIPTION. The default is None.

    Raises
    ------
    ValueError if rank is higher than the number of points
        "the number of data points must exceed order "
                                 "to scale the covariance matrix".

    Returns
    -------
    error_parameters : error uncertainties estimated on the parameters ordered 
    from low to high polynomial order.
    By example for a linear model f(x)=a0+a1*x : 
        error_parameters[0] = constant parameters a0, 
        error_parameters[1] = slope a1.

    cov : covariance estimated for the Least-squares fit of a polynomial to data
        return a Matrix(Rank by rank)
    '''

    from numpy.polynomial.polynomial import polyvander

    if len(x) <= rank:
        raise ValueError("the number of data points must exceed order "
                         "to scale the covariance matrix")
        # note, this used to be: fac = resids / (len(x) - order - 2.0)
        # it was deciced that the "- 2" (originally justified by "Bayesian
        # uncertainty analysis") is not was the user expects
        # (see gh-11196 and gh-11197)
    else:
        v = polyvander(x, rank)  # generate the vandermonde matrix
        cov = residuals/(len(x) - rank)*np.linalg.inv(np.dot(v.T, v))
        error_parameters = np.sqrt(np.diag(cov))

    return error_parameters, cov


def ComputeRMSE(y, yfit, p0):
    '''
    Compute the normalized residual sum of square of the residual of a function
    or Root Mean Square of Error (RMSE)
    See by instance :
    https://statisticsbyjim.com/regression/root-mean-square-error-rmse/
    Script written by Timothée Chauviré 10/26/2023

    Parameters
    ----------
    y : experimental data
        TYPE : Numpy data array
    yfit : experimental data
        TYPE : Numpy data array
    p0 : paremeters used for the fit
        TYPE : Numpy data array
    Returns
    -------
    RMSE : normalized residual sum of square or Root mean square of error
        TYPE : real float value
    '''
    NumMeas = y.shape[0]
    NumParams = len(p0)
    resnorm = np.sum((y-yfit)**2)
    RMSE = np.sqrt(resnorm/(NumMeas - NumParams))
    return RMSE


def eprload(FileName=None, Scaling=None, *args, **kwargs):
    if FileName[-4:].upper() in ['.DSC', '.DTA']:
        data, abscissa, par = eprload_BrukerBES3T(FileName, Scaling)
    elif FileName[-4:].lower() in ['.spc', '.par']:
        data, abscissa, par = eprload_BrukerESP(FileName, Scaling)
    else:
        data, abscissa, par = None, None, None
        raise ValueError("Can\'t Open the File {0} ".format(str(FileName)) +
                         "because the extension isn\'t a Bruker extension " +
                         ".DSC, .DTA, .spc, or .par!")
    return data, abscissa, par
