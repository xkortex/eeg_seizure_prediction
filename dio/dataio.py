from __future__ import print_function, division
import os
import numpy as np
import pandas as pd
import scipy.io
import json


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


def separate_sets(data_vec, label_ary):
    """
    Separates out numpy-pickled data with corresponding data set.
    :param data_vec:
    :param label_ary:
    :return:
    """
    # df = pd.DataFrame(data_vec)
    # df['label'] = pd.Series(label_ary)
    d0 = data_vec[np.where(label_ary.ravel() == 0)[0], :]
    d1 = data_vec[np.where(label_ary.ravel() == 1)[0], :]
    dt = data_vec[np.where(label_ary.ravel() == -1)[0], :]

    return (d0, d1, dt)

def dump_data(vec_ary, name_ary, meta, filename):
    name_ary = pd.DataFrame(name_ary, columns=['path'])
    np.save(filename, vec_ary)
    name_ary.to_csv(filename + '_name.csv')
    with open(filename + '.json', 'w') as jfile:
        json.dump(meta, jfile)


def subdiv_and_shuffle(data, labels, resample='down', noise=None):
    d0, d1, dt = separate_sets(data, labels)
    if resample == 'down':
        np.random.shuffle(d0)
        d0 = d0[:len(d1)]
    elif resample == 'up':
        ratio = len(d0) / len(d1)
        mult = int(ratio) + 1
        d1 = np.concatenate([d1, ] * mult, axis=0)

    new_set = np.concatenate([d0, d1], axis=0)
    #     print('new set: ', new_set.shape)
    new_labels = np.concatenate([np.zeros(len(d0)), np.ones(len(d1))], axis=0).reshape(-1, 1)
    #     print('new labels: ', new_labels.shape)
    connected_set = np.concatenate([new_set, new_labels], axis=1)
    np.random.shuffle(connected_set)
    X, Y = np.array(connected_set[:, :-1]), np.array(connected_set[:, -1:])
    return X, Y