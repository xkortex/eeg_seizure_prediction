import numpy as np
import scipy.signal as signal


def softplus2(x, a=1):
    return np.log(1+np.exp((x+1)*a))/a -1

def envelope(data):
    siga = signal.hilbert(data, axis=0)
    return np.abs(siga)

def softclip(x, a=1, b=1):
    """

    :param x: Data
    :param a: Knee sharpness
    :param b: Slope of linear region
    :return:
    """
    x = x*b
    x_0 = np.log(1+np.exp((x+1)*a))/a -1# softplus(x, a)
    return -np.log(1+np.exp((-x_0+1)*a))/a +1


def norm_softclip(data, sigma=8, zeta=10, global_pow=None, norm_by_chan=False):
    """
    Normalize and soft-clip a signal to the range (-1,1).
    :param data: (N, d) array
    :param sigma: range of std dev the signal is normed to 1, e.g. sigma=6 means the +/-6-sigma line is set to +/-1
    :param zeta: soft-knee coefficient of soft clip. Higher zeta means harder clip, but less distortion
    :param global_pow: (Default: None) Total power to normalize signal by. If None, calculate power using np.std
    :param norm_by_chan: Normalize each channel individually by std
    :return:
    """
    zdata = np.array(data - np.mean(data, axis=0))
    if global_pow is None:
        if norm_by_chan:
            std = np.std(zdata, axis=0)
        else:
            std = np.std(zdata)

    else:
        std = global_pow
    zdata /= (sigma * std)
    return softclip(zdata, zeta)



