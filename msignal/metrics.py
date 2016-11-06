"""Derived from code posted publically by Tom Starke on
https://www.quantopian.com/posts/some-code-from-ernie-chans-new-book-implemented-in-python
"""

# import matplotlib.pyplot as py

import numpy as np

def rms(a, axis=None):
    return np.sqrt(np.mean(a**2, axis=axis))

def hurst(p):
    """Computes the Hurst exponent, a metric of periodicity"""
    tau = []; lagvec = []
    #  Step through the different lags
    for lag in range(2,20):
        #  produce price difference with lag
        pp = np.subtract(p[lag:],p[:-lag])
        #  Write the different lags into a vector
        lagvec.append(lag)
        #  Calculate the variance of the differnce vector
        tau.append(np.sqrt(np.std(pp)))
    #  linear fit to double-log graph (gives power)
    m = np.polyfit(np.log10(lagvec),np.log10(tau),1)
    # calculate hurst
    hurst = m[0]*2
    # plot lag vs variance
    #py.plot(lagvec,tau,'o'); show()
    return hurst


def chanstd(data, mode='std'):
    """
    Computes the inter-channel stddev over time, then computes the StdDev/RMS (I think stdev shows better effect
    :param data: array-like, 1st dim is time, 2nd dim is channels
    :param mode: {'std', 'rms'} Summary mode
    :return:
    """
    chanwise = np.std(data, axis=1)
    if mode == 'std':
        result = np.std(chanwise)
    elif mode == 'rms':
        result = rms(chanwise)
    else:
        raise ValueError("Invalid mode: {}".format(mode))
    return result

if __name__=="__main__":
    #  Different types of time series for testing
    #p = log10(cumsum(random.randn(50000)+1)+1000) # trending, hurst ~ 1
    #p = log10((random.randn(50000))+1000)   # mean reverting, hurst ~ 0
    p = np.log10(np.cumsum(np.random.randn(50000))+1000) # random walk, hurst ~ 0.5
    print hurst(p)