from __future__ import print_function, division
import os, sys
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
    if basename.split('.')[-1] == 'npy' or basename.split('.')[-1] == 'npz':
        data = np.load(basename)
        basename = basename[:-4]
    if os.path.exists(basename +'.npy'):
        data = np.load(basename +'.npy')
    elif os.path.exists(basename + '.npz'):
        data = np.load(basename + '.npz')['arr_0']
    else:
        raise IOError("No such file: {}".format(basename+ '.npy or .npz'))

    data = np.nan_to_num(np.array(data, float))
    data_vec = data.reshape(data.shape[0], -1)
    try:
        label_ary = load_label_ary(basename)
    except FileNotFoundError:
        print('WARNING: No label file found!! Returning None for label array')
        return (data_vec, None)

    assert data_vec.shape[0] == label_ary.shape[0], "Shape mismatch with data and label"
    return (data_vec, label_ary)

def load_label_ary(basename):
    names = pd.read_csv(basename + '_name.csv')
    label = [os.path.basename(name)[:4] + '_' + name[-5] for name in names['path']]
    label_ary = []
    for lab in label:
        if lab[:3] == 'new':
            label_ary.append(-1)
        else:
            label_ary.append(int(lab[-1]))
    return np.array(label_ary).reshape(-1, 1)



def separate_sets(data_vec, label_ary):
    """
    Separates out numpy-pickled data with corresponding data set.
    :param data_vec:
    :param label_ary:
    :return:
    """
    # df = pd.DataFrame(data_vec)
    # df['label'] = pd.Series(label_ary)
    """This part currently breaks on 3D arrays"""
    if label_ary.ndim >= 2:
        if label_ary.shape[1] >= 1:
            label_ary = label_ary[:,0]
    d0 = data_vec[np.where(label_ary.ravel() == 0)[0], :]
    d1 = data_vec[np.where(label_ary.ravel() == 1)[0], :]
    dt = data_vec[np.where(label_ary.ravel() == -1)[0], :]

    return (d0, d1, dt)

def dump_data(vec_ary, name_ary, meta, filename, basedir, catchIOError=True):
    def dump(name_ary, meta, filename, basedir):
        name_ary = pd.DataFrame(name_ary, columns=['path', 'valid_percent'])
        np.savez_compressed(basedir + '/' + filename, vec_ary)
        name_ary.to_csv(basedir + '/' + filename + '_name.csv')
        with open(basedir + '/' + filename + '.json', 'w') as jfile:
            json.dump(meta, jfile)

    if catchIOError:
        try:
            dump(name_ary, meta, filename, basedir)
        except IOError as exc:
            exc_type, exc_obj, exc_tb = sys.exc_info()
            fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
            print(exc_type, fname, exc_tb.tb_lineno)
    else:
        dump(name_ary, meta, filename, basedir)


def subdiv_and_shuffle(data, labels, resample='down', noise=None, merge=True, shuffle=True):
    d0, d1, dt = separate_sets(data, labels)
    if resample == 'down':
        np.random.shuffle(d0)
        d0 = d0[:len(d1)]
    elif resample == 'up':
        ratio = len(d0) / len(d1)
        mult = int(ratio) + 1
        d1 = np.concatenate([d1, ] * mult, axis=0)
    else:
        raise ValueError("Invalid resample argument: {}".format(resample))

    new_set = np.concatenate([d0, d1], axis=0)
    # print('new set: ', new_set.shape)
    L0, L1 = np.zeros(len(d0)).reshape(-1, 1), np.ones(len(d1)).reshape(-1, 1)
    # print('label shapes: {} {} total len {}'.format(L0.shape, L1.shape, len(L0) + len(L1)))
    new_labels = np.concatenate([L0, L1], axis=0).reshape(-1, 1)
    #     print('new labels: ', new_labels.shape)
    if not shuffle:
        return new_set, new_labels
    assert len(new_set) == len(new_labels), "Something failed, lengths of X and Y are not the same"
    if merge:
        connected_set = np.concatenate([new_set, new_labels], axis=1)
        np.random.shuffle(connected_set)
        X, Y = np.array(connected_set[:, :-1]), np.array(connected_set[:, -1:])

    else:
        # raise NotImplementedError('Still need to build a sensible shuffler for multi-dim data')
        fullidx = np.arange(new_set.shape[0])
        np.random.shuffle(fullidx)
        shuf_x = []
        shuf_y = [] # inefficient i know but it should be reliable
        for j in fullidx:
            sj = fullidx[j] # shuffled index
            shuf_x.append(new_set[sj])
            shuf_y.append(new_labels[sj])
            X = np.array(shuf_x)
            Y = np.array(shuf_y)

    return X, Y

def subdiv_split_shuffle(data, labels, resample='pare', validation_split=0.5, noise=None, preshuffle=True, shuffle=False,
                         seed=1337):
    """
    Takes a fixed percentage split of the data for validation with equal representation of classes. The remaining
    data is returned as training set
    :param data:
    :param labels:
    :param resample:
    :param validation_split: If you set the validation_split argument in model.fit to e.g. 0.1, then the validation data
    used will be the last 10% of the data. If you set it to 0.25, it will be the last 25% of the data, etc.
    :param noise:
    :param preshuffle: shuffle the incoming data before taking validation split
    :param shuffle:
    :param seed:
    :return:
    """

    d0, d1, dt = separate_sets(data, labels)
    nb_x0, nb_x1, nb_xt = len(d0), len(d1), len(dt)
    y0, y1 = labels[:nb_x0], labels[nb_x0:]
    assert nb_x0 >= nb_x1, "Only deals with imbalanced data sets with excess d0 class data at this point"
    if preshuffle:
        np.random.seed(seed)
        np.random.shuffle(d0)
        np.random.shuffle(d1)
    if resample == 'pare':
        # Create a balanced test set, but hand the rest off to the training set
        val_cut = int(nb_x1 * validation_split) # number to put in validation set
        print('val_cut: ', val_cut)
        # if shuffle:
        #     np.random.shuffle(d0)
        #     np.random.shuffle(d1)
        cut0 = nb_x0-val_cut
        cut1 = nb_x1-val_cut
        d1_train = d1[:cut1]
        y1_train = np.ones((cut1, 1)) # y1[:nb_x1-val_cut]
        d1_test  = d1[cut1:]
        y1_test  = np.ones((val_cut, 1))# y1[val_cut:]

        d0_train = d0[:cut0]
        y0_train = np.zeros((cut0, 1))# y0[:cut0]
        d0_test  = d0[cut0:]
        y0_test  = np.zeros((val_cut, 1))# y0[cut0:]

        x_train = np.concatenate([d0_train, d1_train], axis=0)
        y_train = np.concatenate([y0_train, y1_train], axis=0)
        x_test  = np.concatenate([d0_test, d1_test], axis=0)
        y_test  = np.concatenate([y0_test, y1_test], axis=0)

    elif resample == 'adasyn':
        raise NotImplementedError('not ready yet')


    else:
        raise ValueError("Invalid resample argument: {}".format(resample))

    if shuffle:
        raise NotImplementedError('Untested Feature')
        np.random.seed(seed)
        np.random.shuffle(x_train)
        np.random.seed(seed)
        np.random.shuffle(y_train)
        np.random.seed(seed)
        np.random.shuffle(x_test)
        np.random.seed(seed)
        np.random.shuffle(y_test)
    return (x_train, y_train), (x_test, y_test)

def shuffle_split_with_label(data, labels, resample='down', seed=1337):
    # print("Warning: this method is not validated yet")
    d0, d1, dt = separate_sets(data, labels)
    y0, y1 = labels[:len(d0)], labels[len(d0):]
    # print('d0 {} d1 {} y0 {} y1 {}'.format(d0.shape, d1.shape, y0.shape, y1.shape))

    # assert len(d0) >= len(d1), 'must have more 0 than 1 classes'
    if resample == 'down':
        np.random.seed(seed)
        np.random.shuffle(d0)
        d0 = d0[:len(d1)]
        y0 = y0[:len(d1)]

    else:
        raise ValueError("Invalid resample argument: {}".format(resample))

    # print('d0 {} d1 {} y0 {} y1 {}'.format(d0.shape, d1.shape, y0.shape, y1.shape))
    new_x = np.concatenate([d0, d1], axis=0)
    new_y = np.concatenate([y0, y1], axis=0)
    # print('new shapes', new_x.shape, new_y.shape)
    assert new_x.shape[0] == new_y.shape[0], 'X and Label shapes mismatch, something broke'
    np.random.seed(seed)
    np.random.shuffle(new_x)
    np.random.seed(seed)
    np.random.shuffle(new_y)

    return new_x, new_y

def deinterleave(data, nchan=16):
    """
    Converts 3D array with channel as last dim, to serialized (one channel after another) feature vector
    :param data:
    :param nchan:
    :return:
    """
    vecdim = data.shape[1]
    newvecdim = vecdim*nchan
    newdata = np.zeros((len(data), newvecdim))
    for i in range(0, len(data)):
        for ch in range(0, nchan):
            newdata[i,ch*vecdim:(ch+1)*vecdim] = data[i,:,ch]

    return np.asarray(newdata)


def respool_electrodes(data, nchan=16):
    ndata, ndim = data.shape
    newdata = np.zeros((ndata // nchan, nchan, ndim))
    for i in range(0, len(data), nchan):
        for ch in range(0, nchan):
            #             newframe.append(data[ch])
            k = i // nchan
            newdata[k][ch] = data[i + ch]
            #         newdata.append(np.array(newframe).ravel())

    return np.asarray(newdata).reshape(ndata // nchan, -1)

class NormOMatic(object):
    def __init__(self, centerMode='mean', disperseMode='std', mu=0.0, sigma=1.0, normGlobal=False, verbose=1):
        self._mu = 0  # center point
        self._sigma = 1# stdDev/ dispersion
        self._centerMode = centerMode
        self._disperseMode = disperseMode
        self._normGlobal = normGlobal

    def fit(self, X, Y=None):
        ax = None if self._normGlobal else 0

        if self._centerMode == 'mean':
            mu_fn = np.mean
        elif self._centerMode[:3] == 'med':
            mu_fn = np.median
        else:
            raise ValueError("Invalid mode specifier: {}".format(self._centerMode))
        self._mu = mu_fn(X, axis=ax)

        if self._disperseMode[:3] == 'std':
            sigma_fn = np.std
        elif self._disperseMode[:3] == 'var':
            sigma_fn = np.var
        elif self._disperseMode[:3] == 'mad':
            sigma_fn = NormOMatic.mad
        else:
            raise ValueError("Invalid mode specifier: {}".format(self._disperseMode))
        self._sigma = sigma_fn(X, axis=ax)

    def transform(self, X, Y=None):
        return (X - self.mu) / self.sigma

    @property
    def mu(self): return self._mu

    @property
    def sigma(self):
        return self._sigma

    @staticmethod
    def mad(a, axis=None, dtype=None, out=None, keepdims=False):
        return np.mean(np.abs(a - np.mean(a, axis=axis)), axis=axis)


class UnbalancedStratifier(object):
    def __init__(self, X=None, Y=None, nFolds=2, preshuffle=True, random_state=None, mandateZerosExcess=True):
        self.mandateZerosExcess = mandateZerosExcess
        self.nFolds = nFolds
        self.random_state = random_state
        if X is not None and Y is not None:
            self.feed(X, Y, preshuffle=preshuffle)

    def feed(self, X, Y, preshuffle=True):
        self.X = X # this prolly adds memory bloat, oh well
        self.d0, self.d1, self.dt = separate_sets(X, Y)
        self.zeros_bias = len(self.d1) <= len(self.d0)
        if self.mandateZerosExcess and not self.zeros_bias:
            raise ValueError("Fix this for extensibility later!!!")
        if preshuffle:
            np.random.seed(self.random_state)
            np.random.shuffle(self.d0)
            np.random.shuffle(self.d1)

    def summary(self):
        print('Total Samples:', len(self.d0) + len(self.d1) + len(self.dt))
        print('Shapes', self.d0[0].shape)
        print('Class 0:', len(self.d0))
        print('Class 1:', len(self.d1))
        print('Guess  :', len(self.dt))
        print('Prevalence: {}%'.format(100.*len(self.d1)/len(self.d0)))
        print('Zeros bias? ', self.zeros_bias)
        print('Set mean/std: {:4.4} {:4.4}'.format(np.mean(self.X), np.std(self.X)))
        print('d0  mean/std: {:4.4} {:4.4}'.format(np.mean(self.d0), np.std(self.d0)))
        print('d1  mean/std: {:4.4} {:4.4}'.format(np.mean(self.d1), np.std(self.d1)))

    def dataset_list(self):
        datasets = [d for d in self.gen_dataset()]
        return datasets

    def gen_dataset(self):
        jcut = len(self.d0) // self.nFolds
        kcut = len(self.d1) // self.nFolds
        for i in range(self.nFolds):
            pre = slice(0, i*kcut)
            mid = slice(i*kcut, (i+1)*kcut)
            post = slice((i+1)*kcut, len(self.d1))
            val_d1 = self.d1[mid]
            train_d1 = np.concatenate([self.d1[pre], self.d1[post]])
            valSetSize = len(val_d1)
            trainSetSize = len(train_d1)
            # pre_0 = slice(i*jcut, i*jcut + valSetSize)
            start = self.nFolds * valSetSize
            mid_0 = slice(i*valSetSize, (i+1) * valSetSize)
            post_0 = slice(start + i * trainSetSize, start + (i+1) * trainSetSize)
            val_d0 = self.d0[mid_0]
            train_d0 = self.d0[post_0]
            if len(train_d1) == len(train_d0) and len(val_d1) == len(val_d0):

                train = np.concatenate([train_d0, train_d1], axis=0)
                val = np.concatenate([val_d0, val_d1], axis=0)
                y_train = np.concatenate([np.zeros(len(train_d0)), np.ones(len(train_d1))], axis=0)
                y_val = np.concatenate([np.zeros(len(val_d0)), np.ones(len(val_d1))], axis=0)

                np.random.seed(self.random_state)
                np.random.shuffle(train)
                np.random.shuffle(val)
                np.random.seed(self.random_state)
                np.random.shuffle(y_train)
                np.random.shuffle(y_val)
                # yield (len(train_d1) , len(train_d0) , len(val_d1) , len(val_d0))
                yield ((train, y_train), (val, y_val))
            else:
                yield None

        return
