import numpy as np
import scipy as sp
import scipy.io
import os, pickle
def get_matlab_eeg_data(path):
    rawdata = scipy.io.loadmat(path)
    ds = rawdata['dataStruct']
    return ds['data'][0,0]
    # return outdata['data']
def simple_fft_vectorize(data):
    vector = np.fft.fft(data[:,0])[:6000]
    for i in xrange(1, data.shape[1]):
        vector +=  np.fft.fft(data[:,i])[:6000]
    vector /= data.shape[1]
    return vector
def simple_fft_vectorize_larger(data):
    vector = np.fft.fft(data[:,0])[:12000]
    for i in xrange(1, data.shape[1]):
        vector +=  np.fft.fft(data[:,i])[:12000]
    vector /= data.shape[1]
    return vector
def build_dataset(folder_path, vectorize=simple_fft_vectorize_larger, training=True):
    try:
        X = np.load(vectorize.__name__ + folder_path.replace('/', '_') + '_X.npy')
        if training:
            Y = np.load(vectorize.__name__ + folder_path.replace('/', '_') + '_Y.npy')
            return X, Y*1
        else:
            paths = pickle.load(open(vectorize.__name__ + folder_path.replace('/', '_') + '_paths', 'r'))
            return X, paths
    except:
        pass
    for root, dirs, files in os.walk(folder_path):
        file_paths = sorted([os.path.join(root, path) for path in files],key=lambda x: int(x.split('/')[-1].split('.')[0].split('_')[-1]))
        # file_paths.sort()
        X = None
        Y = None
        count = 0
        for path in file_paths:
            try:
                data = get_matlab_eeg_data(path)
            except:
                continue
            vector = vectorize(data)
            truth_val = [0,1] if path.split('/')[-1].split('.')[0].split('_')[-1]=='1' else [1,0]
            if X is None:
                X = vector
                if training:
                    Y = np.array([truth_val])
            else:
                X = np.vstack((X, vector))
                if training:
                    Y = np.vstack((Y, np.array(truth_val)))
            count+=1
            print 'imported', path, count
        np.save(vectorize.__name__ + folder_path.replace('/', '_') + '_X', X)
        if training:
            np.save(vectorize.__name__ + folder_path.replace('/', '_') + '_Y', Y*1)
            return X, Y*1
        else:
            fs = open(vectorize.__name__ + folder_path.replace('/', '_') + '_paths', 'w')
            pickle.dump(file_paths, fs)
            fs.close()
            return X, file_paths