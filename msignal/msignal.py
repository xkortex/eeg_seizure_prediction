import numpy as np


def softclip(x, a=1):
    x_0 = np.log(1+np.exp((x+1)*a))/a -1# softplus(x, a)
    return -np.log(1+np.exp((-x_0+1)*a))/a +1