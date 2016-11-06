from __future__ import print_function, division
import numpy as np
import scipy.io


def validcount(rawdata):
    # rawdata1 = matlabtools.get_matlab_eeg_data(path)['data']
    return np.count_nonzero(rawdata)/ np.size(rawdata)

def get_matlab_eeg_data(path):
    rawdata = scipy.io.loadmat(path)
    ds = rawdata['dataStruct']
    fields = dict(ds.dtype.fields)
    outdata = {}
    for key in fields.keys():
        outdata.update({key: ds[key][0,0]})
    return outdata

def get_matlab_eeg_data_ary(path):
    return get_matlab_eeg_data(path)['data']
