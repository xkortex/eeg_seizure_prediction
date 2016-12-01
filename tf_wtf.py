
# coding: utf-8
from __future__ import print_function, division
import os, sys
import numpy as np, scipy as sp, pandas as pd
import scipy.io
import scipy.signal as signal, scipy.fftpack as ftpk, scipy.integrate as integrate, scipy.interpolate as interpolate
# import matplotlib.pyplot as plt
# import matplotlib
# import tensorflow as tf
# import tflearn
from sklearn import linear_model as lm, neural_network as nn
from sklearn import preprocessing as preproc

PLOT=False


#
# os.chdir('/home/mike/py/kaggle/')
# print(os.getcwd())
from .dio import dataio



def janky_subdiv(data, start=0, subdiv=64, chanVecSize=1024, nChan=16):
    if data.ndim == 3:
        return data[:,start::subdiv,:].reshape(-1,chanVecSize*nChan//subdiv)
    elif data.ndim == 2:
        # raise NotImplementedError('dont use this')
        return data[:,start::subdiv]
    else:
        raise NotImplementedError("d'oh! input is totally wrong shape")
        
class QuasiRobustScaler(object):
    def __init__(self, X=None, Y=None, normGlobal=True):
        pass
    def fit_transform(X, Y=None, normGlobal=True):
        pass
        

def preprocess_wtf(data_train, data_test, Y, start=0, subdiv=64, renorm=True, random_state=None, verbose=0):
    if verbose >= 2: print('shapes into preproc', data_train.shape, Y.shape)

    ## SOMETHING IN THIS METHOD IS MUTATING Y.
    data_train2 = dataio.respool_electrodes(data_train)
    simple_dtrain = janky_subdiv(data_train2, start=start)
    simple_dtest = janky_subdiv(data_test, start=2)
    if verbose >= 2: print('Shape after jankifier: ', simple_dtrain.shape)


    # homemade scaler
    if renorm:
        normo = dataio.NormOMatic(centerMode='med', normGlobal=True)
        normo.fit(simple_dtrain)
        simple_dtrain = normo.transform(simple_dtrain)
        simple_dtest = normo.transform(simple_dtest)


    if verbose >= 2: print(np.mean(simple_dtrain))
    if verbose >= 2: print('These should match', simple_dtrain.shape, Y.shape)

    dtrain_set = np.concatenate([simple_dtrain, Y], axis=1)
    dframe = pd.DataFrame(dtrain_set)

    d0 = dframe[dframe.iloc[:, -1] == 0]
    d1 = dframe[dframe.iloc[:, -1] == 1]
    if verbose >= 2: print(d0.shape, d1.shape)
    nfalse, nhit = d0.shape[0], d1.shape[0]

    # randomly picks the zero datas... i think. the continuity may be essential
    offset = np.random.randint(0, nfalse - nhit - 1)
    d0b = d0[offset:offset + nhit]
    if verbose >= 2: print(d0b.shape)

    # In[342]:

    d0b_ = d0b.as_matrix()
    d1_ = d1.as_matrix()

    # ### Shuffle and shit
    subdiv_vec = 1
    new_set = np.concatenate([d0b_, d1_], axis=0)
    if verbose >= 2: print('new_set:', new_set.shape)
    np.random.seed(random_state)
    np.random.shuffle(new_set)
    if verbose >= 2: print(np.mean(new_set[:nhit, -1]))
    simple_dtrain = new_set[:, :-2]
    simple_dtrain_lab = new_set[:, -1]
    simple_dtrain = simple_dtrain[:, ::subdiv_vec]
    if verbose >= 2: print(simple_dtrain.shape)
    if verbose >= 2: print(np.mean(simple_dtrain_lab[:nhit]))

    X = simple_dtrain
    Y_new = simple_dtrain_lab

    G = simple_dtest
    return X, Y_new, G

def sketchy_balancer(X, Y, cut=256, sl=1, kf=2, validation_split=0.2, assertion_cutoff = 0.2):
    val_idx = int((1 - validation_split) * len(X))
    X1, Y1 = X[:val_idx:kf, :cut:sl], Y[:val_idx:kf]
    X2, Y2 = X[val_idx::kf, :cut:sl], Y[val_idx::kf]
    balance_y1, balance_y2 = np.mean(Y1, axis=0), np.mean(Y2, axis=0)
    assert 0.5 - assertion_cutoff < balance_y1 < 0.5 + assertion_cutoff, "Labels are unbalanced"
    assert 0.5 - assertion_cutoff < balance_y2 < 0.5 + assertion_cutoff, "Labels are unbalanced"
    return X1, Y1, X2, Y2

def ensemble_classifier(X, Y, start=0, subdiv=64, validation_split=0.20, random_state=None, renorm=True, verbose=0):
    print('shapes into classifier', X.shape, Y.shape)
    X1, Y1, X2, Y2 = sketchy_balancer(X, Y, validation_split=.2)
    # strat = dataio.UnbalancedStratifier(X, Y, 1)
    # dataset = strat.dataset_list()
    # print(type(dataset[0]))
    # (X1, Y1), (X2, Y2) = dataset[0]

    if verbose >= 2: print(X1.shape, Y1.shape, )
    if verbose >= 2: print(np.mean(X,), np.mean(X), np.std(X, ), np.std(X, ), )
    if verbose >= 2: print(np.mean(X1,), np.mean(X2), np.std(X1, ), np.std(X2, ), )

    perc = lm.Perceptron()
    perc.fit(X1, Y1)

    if verbose >= 2: print( perc.score(X1, Y1), np.mean(Y, axis=0))
    if verbose >= 2: print( perc.score(X2, Y2), np.mean(Y, axis=0))

    pr = perc.predict(X2)
    if verbose >= 1: print('Expected 0.5: {:.3f}'.format(pr.mean()))

    if verbose >= 1: print('VALIDATION {:2}: {:.2f} %'.format(start, 100*np.mean(pr == Y2)))
    return perc


def ensembleOMatic(verbose=1, K=64):

    basedir = '/home/mike/data/vectors/'
    data_train = np.load(basedir + 'vec_1478816228.31.npy')
    names_train = pd.read_csv(basedir + 'vec_1478816228.31_name.csv')
    data_test = np.load(basedir + 'vec_1478825795.45.npy')
    names_test = pd.read_csv(basedir + 'vec_1478825795.45_name.csv')
    if verbose >=2: print(data_train.shape, data_test.shape)

    data_train = np.nan_to_num(data_train)
    data_test = np.nan_to_num(data_test)

    names_train['label'] = [int(name[-5]) for name in names_train['path']]
    if verbose >= 2: print(names_train.shape, names_train['label'].mean())
    names_train.head()

    name_mask = names_train['label'] == 0
    Y = np.vstack([name_mask, ~name_mask]).T  # hack to regenerate Y

    # normo = dataio.NormOMatic(centerMode='med', normGlobal=True)
    # normo.fit(simple_dtrain)
    # simple_dtrain = normo.transform(simple_dtrain)
    # simple_dtest = normo.transform
    X_val, Y_val, G_val = preprocess_wtf(data_train, data_test, Y, start=1, subdiv=K, random_state=None, verbose=verbose)

    models = []
    guesses = []
    for i in range(K):
        # Yz = np.copy(Y)
        Y = np.vstack([name_mask, ~name_mask]).T  # hack to regenerate Y

        X, Y2, G = preprocess_wtf(data_train, data_test, Y, start=i, subdiv=K, random_state=None, verbose=verbose)

        # try:
        classer = ensemble_classifier(X, Y2, start=i, subdiv=K, verbose=verbose)
        # except AssertionError:
        #     classer = ensemble_classifier(X, Y2, G, start=i, subdiv=K, verbose=verbose)
        pr_all = classer.predict(X_val)
        guess = classer.predict(G)
        # print('FULL VALID {:2}: {:.2f} %'.format(i, 100*np.mean(pr_all == Y2)))

        # models.append(classer, guesses)
        guesses.append(guess)

    guesses = np.array(guesses)
    print(guesses.shape)
    ensemble_prob = np.mean(guesses, axis=0)
    print(ensemble_prob.shape)
    return ensemble_prob


if __name__ == '__main__':
    probs = ensembleOMatic(1)