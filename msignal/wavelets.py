from __future__ import division, print_function

import numpy as np
from scipy import signal

def abthresh(x, thresh=1.):
    """
    Thresholding function, absolute value of signal less than 'thresh' is set to 0
    :param x:
    :param thresh:
    :return:
    """
    idx = np.abs(x)[:,:] < thresh
    x[idx] = 0
    return x

class CWTlets(object):
    """
    Wavelets for use with the scipy signal.cwt
    """
    @staticmethod
    def rescale(sig):
        """
        Returns a scaled version of the supplied wavelet function, scaled according to window size
        :param sig:
        :return:
        """
        xyz = (1 / np.pi) ** 0.125 # no idea why this works for the ricker wavelet
        return lambda n, a: xyz * sig(n, a) / a**0.5

    @staticmethod
    def haar(n, a):
        a = int(a)
        n = int(n)
        y = np.zeros(n, dtype=float)
        m = n // 2
        y[m - 2 * a:m] = 1
        y[m: m + 2 * a] = -1
        return y

    @staticmethod
    def haaro(n, a):
        """Odd harr wavelet"""
        a = int(a)
        n = int(n)
        m = n // 2
        y = np.zeros(n, dtype=float)
        y[m - a:m + a] = 1
        y[m - 2 * a: m - a] = -1
        y[m + a: m + 2 * a] = -1
        return y

    @staticmethod
    def rect(n, a):
        """Rectangular pulse"""
        a = int(a)
        n = int(n)
        m = n // 2
        y = np.zeros(n, dtype=float)
        y[m - a:m + a] = 1
        y[m - 2 * a: m - a] = 0
        y[m + a: m + 2 * a] = 0
        return y

    @staticmethod
    def ricker_i(points, a):
        return np.imag(signal.hilbert(signal.ricker(points, a)))

    @staticmethod
    def s_rect(n, a, d=0.5):
        """Rectangular pulse with duty cycle"""
        a = int(a)
        n = int(n)
        m = n // 2
        r = 1/d
        y = np.zeros(n, dtype=float)
        y[m - a:m + a] = 1
        y[m - 2 * a: m - a] = 0
        y[m + a: m + 2 * a] = 0
        return y


    @staticmethod
    def s_morlet(n, a, z=2.):
        """
        Morlet with variable shape
        :param n:
        :param a:
        :param z: Shaping parameter, default 2.0. Hat2: z=1/sqrt(e)
        :return:
        """
        n = float(n)
        a = float(a)
        t = np.arange(-n / 2, n / 2) / n
        fc = n / (.5 * np.pi * a) # fc is critical frequency
        b = n ** 2 / a ** 2 / np.pi
        sig = np.exp(-b * t ** 2) * np.exp(1j * z * np.pi * fc * t)
        return sig

    @staticmethod
    def morlet(n, a):
        return CWTlets.s_morlet(n, a)

    @staticmethod
    def gausslet(n, a):
        return CWTlets.s_morlet(n, a, 0.)

    @staticmethod
    def igausslet(n, a):
        # return signal.hilbert(CWTlets.s_morlet(n, a, 0.))
        return np.imag(signal.hilbert(np.real(CWTlets.gausslet(n, a)))) # barf.

    @staticmethod
    def bouncelet(n, a, z=2.):
        n = float(n)
        a = float(a)
        t = np.arange(-n / 2, n / 2) / n
        fc = n / (.5 * np.pi * a)  # fc is critical frequency
        b = n ** 2 / a ** 2 / np.pi
        sig = np.exp(-b * t ** 2) * np.exp(1j * z * np.pi * fc * t)
        sig = np.abs(np.real(sig))
        return sig

    @staticmethod
    def hat2(n, a):
        coef = 1/np.sqrt(np.e)
        return CWTlets.s_morlet(n, a, coef )

    @staticmethod
    def gen_morlet(z):
        return lambda n, a: CWTlets.s_morlet(n, a, z)

    @staticmethod
    def sinclet(n, a):
        n = float(n)
        a = float(a)
        fc = n / (a) # fc is critical frequency
        t = np.arange(-n / 2, n / 2) / n
        fct = fc * t
        real = np.sin(fct) / (fct)
        imag = 1j * (-np.cos(fct) / (fct) + np.sin(fct) / fct ** 2)
        return real + imag

    @staticmethod
    def sqsinc(n, a):
        n = float(n)
        a = float(a)
        fc = n / (a) # fc is critical frequency
        t = np.arange(-n / 2, n / 2) / n
        fct = fc * t
        real = (np.sin(fct) / (fct)) **2
        return real

def lazy_icwt(cw, widths):
    scaling = np.array(widths.reshape(-1,1))
    scaling = 1/ scaling # works with non-rescaling wavelets, e.g. amplitude 1
    cw2 = scaling * cw
    combined = np.mean(cw2, axis=0)
    return combined

def cwtfiltbank(cw, widths, amin, amax):
    idxmin = (np.abs(widths-amin)).argmin()
    idxmax = (np.abs(widths-amax)).argmin() + 1
    return(cw[idxmin:idxmax], widths[idxmin:idxmax])

def wconv(sig, wavelet, width):
    return signal.cwt(sig, wavelet, [width])[0]

def acwt(sig, widths):
    """ Reconstruct an analytic signal using CWT"""
    magical_constant = 16.854880972 # this doesn't actually scale properly
    # Somehow, I'm still seing dependence on the number of widths. 
    cwt_r = signal.cwt(sig, signal.ricker, widths)
    cwt_i = signal.cwt(sig, CWTlets.ricker_i, widths)
    a_sig = lazy_icwt(cwt_r, widths) + 1j * lazy_icwt(cwt_i, widths)
    return a_sig #* magical_constant

def wavediff(sig, width):
    """Wavelet derivative"""
    pass