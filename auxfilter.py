from __future__ import print_function, division

from math import pi
import numpy as np
import pandas as pd
from scipy import signal
from scipy.ndimage.filters import gaussian_filter1d


# todo: merge these

def rms_ary(ary):
    """
    Calculate RMS (root mean square) of an N, D dimensional array
    :param ary:
    :return:
    """
    means = np.mean(ary, axis=0)
    errs = ary - means
    return np.mean(errs ** 2, axis=0) ** 0.5

def diff_ary(ary):
    """
    Returns the 1-D derivative of the array along the long axis (axis=0) with the same shape.
    :param ary:
    :return: numpy ndarray
    """
    # data = pd.DataFrame(ary).diff()
    # data = data.fillna(data[1]) # simple backfill
    # data = data.as_matrix()
    data = np.ediff1d(ary, to_begin=ary[1]-ary[0]) # backfill
    return data


def norm_ary(ary):
    """
    Returns the norm of an (N, D) numpy array expressing a list (N items) of D-dimensional vectors.
    :param ary: (N, D)
    :type ary: np.ndarray
    :return: (N, 1)
    """
    squaresum = np.sum(np.asarray(ary)**2, axis=1)
    return squaresum**0.5

def norm_ary_pd(ary):
    """
    Returns the norm of an (N, D) pandas dataframe expressing a list (N items) of D-dimensional vectors.
    :param ary: (N, D) pandas dataframe
    :return: (N, 1) pandas dataframe of the norm
    """
    squaresum = pd.DataFrame.sum(ary**2, axis=1)
    return squaresum**0.5

def instafilt(x, N=1, Wn=0, typ='butter', btype='low', analog=True):
    """
    Janky, doesn't work right now
    :param x: array-like, data to be filtered
    :param typ:
    :param btype:
    :param Wn: Critical frequencies, scalar or (a, b). On the range of (0,1].
    :param analog:
    :return: array-like, filtered data
    """
    raise NotImplementedError('Not ready yet. ')
    # if typ[:3] == 'but':
    #     b,a = signal.butter(N, Wn, btype, analog, output='ba')

    return signal.filtfilt(b, a, x, axis=0)


def butter_lowpass_filter(data, cutoff, fs=1., order=1, axis=0, analog=False):
    # todo: add option to filtfilt or lfilter
    """
    Apply a digital Butterworth low-pass filter.
    :param data: array-like
    :param cutoff: Critical frequency, Hz
    :param fs: Sampling freqency, Hz
    :param order: Order of
    :param axis: ndarray axis, 0='long' axis, 1='row' axis
    :return:
    """
    nyquistFreqInRads = (2*pi*fs)/2
    Wn = 2*pi*cutoff / (nyquistFreqInRads)
    b, a = signal.butter(order, Wn, btype='low', analog=analog)
    y = signal.filtfilt(b, a, data, axis=axis)
    return y

def butterfilt(data, cutoff, fs=1., order=1, btype='low', ftype='filtfilt', axis=0, analog=False):
    """
    Apply a digital Butterworth low-pass filter.
    :param data: array-like
    :type data: np.ndarray pd.DataFrame
    :param cutoff: Critical frequency, Hz
    :param fs: Sampling freqency, Hz
    :param order: Order of
    :param axis: ndarray axis, 0='long' axis, 1='row' axis
    :return:
    """
    nyquistFreqInRads = (2*pi*fs)/2

    if isinstance(cutoff, (tuple, list, np.ndarray)):
        crit = 2*pi*np.array(cutoff) / nyquistFreqInRads
    else:
        crit = 2*pi*cutoff / nyquistFreqInRads
    b, a = signal.butter(order, crit, btype=btype, analog=analog)
    if ftype == 'filtfilt':
        y = signal.filtfilt(b, a, data, axis=axis)
    elif ftype == 'lfilt':
        y = signal.lfilter(b, a, data, axis=axis)
    else:
        raise ValueError('Invalid filter type specified: {}'.format(btype))

    # If data is dataframe, restore the original column and index info
    if isinstance(data, pd.DataFrame):
        y = pd.DataFrame(y, index=data.index, columns=data.columns)
    elif isinstance(data, pd.Series):
        y = pd.Series(y, index=data.index, name=data.name)

    return y

def filt(data, cutoff, fs=1., order=1, rp=10., rs=10., kind='butter', btype='low', ftype='filtfilt', axis=0, analog=False):
    """
    Apply a digital filter.
    :param data:
    :param cutoff:
    :param fs:
    :param order:
    :param rp:
    :param rs:
    :param kind:
    :param btype:
    :param ftype:
    :param axis:
    :param analog:
    :return:
    """
    nyquistFreqInRads = (2*pi*fs)/2
    crit = 2*pi*cutoff / nyquistFreqInRads
    if kind == 'butter':
        b, a = signal.butter(order, crit, btype=btype, analog=analog)
    elif kind == 'bessel':
        b, a = signal.bessel(order, Wn=crit, btype=btype, analog=analog)
    elif kind == 'cheby1':
        b, a = signal.cheby1(order, rp=rp, Wn=crit, btype=btype, analog=analog)
    elif kind == 'cheby2':
        b, a = signal.cheby2(order, rs=rs, Wn=crit, btype=btype, analog=analog)
    elif kind == 'ellip':
        b, a = signal.ellip(order, rp=rp, rs=rs, Wn=crit, btype=btype, analog=analog)
    else:
        raise ValueError('Invalid filter type specified: {}'.format(kind))

    if ftype == 'filtfilt':
        y = signal.filtfilt(b, a, data, axis=axis)
    elif ftype == 'lfilt':
        y = signal.lfilter(b, a, data, axis=axis)
    else:
        raise ValueError('Invalid filter type specified: {}'.format(btype))

    # If data is dataframe, restore the original column and index info
    if isinstance(data, pd.DataFrame):
        y = pd.DataFrame(y, index=data.index, columns=data.columns)
    elif isinstance(data, pd.Series):
        y = pd.Series(y, index=data.index, name=data.name)

    return y

def tukeywin(x, a=0.1):
    tukey = signal.tukey(len(x), a)
    return x * tukey

def smooth_gps(gpsFrame, sigma=10.):
    """
    Takes a Pandas dataframe of lat, lon values and smooths them with gaussian blur to remove stair-step quant error.
    Rule of thumb: Set sigma to the ratio of the other data point sample rate to the GPS sample rate
    :param gpsFrame:
    :return:
    """
    return gaussian_filter1d(gpsFrame, sigma, axis=0)

#
# def minikalman():
#     for filterstep in range(m-1):
#          #Time Update
#            #=============================
#          #Project the state ahead
#         x=A*x
#
#         #Project the error covariance ahead
#         P=A*P*A.T+Q
#
#         #Measurement Update(correction)
#         #===================================
#         #if there is GPS measurement
#         if GPS[filterstep]:
#         #COmpute the Kalman Gain
#         S =(H*P*H).T + R
#         S_inv=S.inv()
#         K=(P*H.T)*S_inv
#
#         #Update the estimate via z
#         Z = measurements[:,filterstep].reshape(H.shape[0],1)
#         y=Z-(H*x)
#         x = x + (K*y)
#
#         #Update the error covariance
#         P=(I-(K*H))*P
#
#
#         # Save states for Plotting
#         x0.append(float(x[0]))
#         x1.append(float(x[1]))
#
#
#         Zx.append(float(Z[0]))
#         Zy.append(float(Z[1]))
#
#         Px.append(float(P[0,0]))
#         Py.append(float(P[1,1]))
#
#
#
#         Kx.append(float(K[0,0]))
#         Ky.append(float(K[1,0]))



