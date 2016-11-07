"""Functions for computing vectors based on mapreduces"""

import numpy as np, pandas as pd
import scipy.stats

from ..msignal import metrics

def metric_ttest(basepath, mapreduce_fn):
    """
    Provided a path of training data, and a mapreduce function which reduces a data frame to a single scalar,
    calculate the p-value of the separation of the parameter by labels
    :param basepath: path of training data
    :param mapreduce_fn: mapreducing function
    :return: p-value
    """


