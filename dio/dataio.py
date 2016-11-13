from __future__ import print_function, division
import os
import numpy as np
import pandas as pd
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

def reload_with_labels(basename):
    data = np.load(basename +'.npy')
    names = pd.read_csv(basename +'_name.csv')
    label = [os.path.basename(name)[:4]+'_'+name[-5] for name in names['path']]
    label_ary = []
    for lab in label:
        if lab[:3] == 'new':
            label_ary.append(-1)
        else:
            label_ary.append(int(lab[-1]))
    label_ary = np.array(label_ary).reshape(-1,1)
    data = np.nan_to_num(np.array(data, float))
    data_vec = data.reshape(data.shape[0],-1)
    assert data_vec.shape[0] == label_ary.shape[0], "Shape mismatch with data and label"
    return (data_vec, label_ary)