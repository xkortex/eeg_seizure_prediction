import numpy as np
import scipy as sp
import scipy.io

def get_matlab_eeg_data(path):
    rawdata = scipy.io.loadmat(path)
    ds = rawdata['dataStruct']
    fields = dict(ds.dtype.fields)
    outdata = {}
    for key in fields.keys():
        outdata.update({key: ds[key][0,0]})
    return outdata