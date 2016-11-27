"""
Would you like to listen to brain? How it sounds? Are they any
patterns? Any differences between intraictal and preitcal samples?

This script converts some samples on EEG data into sound (a WAV file).
It's ~100x faster (converting sampling rate 400 Hz to 44.1 kHz),
ie. 10 minutes are compressed into ~5 seconds.
The 16 channels are scaled to [-1, 1] range separately and are put
one after other in mono.

It uses scipy wavfile for WAV output. The soundfile library is better
and allows compressed lossless FLAC, but it's not available in Kaggle
Kernels, so try it at home to save ~50% file size.
"""
from __future__ import print_function, division
import os, sys
import tqdm
import numpy as np
from scipy.io import loadmat
from sklearn.preprocessing import MinMaxScaler

from ..dio import dataio

try:
    # Allows other file formats, such as FLAC.
    # But is not available in Kaggle Kernels.
    import soundfile as sf
    def save_audio(output_file, data, sample_rate):
        sf.write(output_file, data, sample_rate)
except ImportError:
    from scipy.io import wavfile
    def save_audio(output_file, data, sample_rate):
        wavfile.write(output_file, sample_rate, np.int16(data * 2 ** 15))


def convert(mat):
    # structure:
    # mat: dict
    # mat['dataStruct']: ndarray (1, 1) of type dtype(a[('data', 'O'),
    # ('iEEGsamplingRate', 'O'), ('nSamplesSegment', 'O'),
    # ('channelIndices', 'O'), ('sequence', 'O')])
    #
    names = mat['dataStruct'].dtype.names
    ndata = {n: mat['dataStruct'][n][0, 0] for n in names}
    return ndata


def auralize(input_file, outpath=None, sample_rate=44100, thresh=.75, overwrite=False, verbose=False):
    prefix, ext = os.path.basename(input_file).split('.')
    outfile = '{}_{}.{}'.format(prefix, sample_rate, 'wav')
    if outpath is not None:
        outfile = '{}/{}'.format(outpath, outfile)
    if os.path.exists(outfile) and not overwrite:
        return 1
    data = convert(loadmat(input_file))['data']
    vc = dataio.validcount(data)
    if vc < thresh:
        if verbose:
            print('Dropout: {} - {}'.format(vc, input_file))
        return 1
    channel_count = data.shape[1]
    # scale to [-1, 1], put all channels after each other
    y = np.vstack([MinMaxScaler(feature_range=(-1, 1)).fit_transform(data[:, i:i+1]) for i in range(channel_count)])

    save_audio(outfile, y, sample_rate)
    return 0

def auralize2(input_file, outpath=None, sample_rate=44100, thresh=.75, overwrite=False, verbose=False):
    """
    Outputs to stereo.
    Based on the Melbourne data, even channel indicies should be paired with even, odd with odd,
    :param input_file:
    :param outpath:
    :param sample_rate:
    :param thresh:
    :param overwrite:
    :param verbose:
    :return:
    """
    prefix, ext = os.path.basename(input_file).split('.')
    outfile = '{}_{}.{}'.format(prefix, sample_rate, 'wav')
    if outpath is not None:
        outfile = '{}/{}'.format(outpath, outfile)
    if os.path.exists(outfile) and not overwrite:
        return 1
    data = convert(loadmat(input_file))['data']
    vc = dataio.validcount(data)
    if vc < thresh:
        if verbose:
            print('Dropout: {} - {}'.format(vc, input_file))
        return 1
    channel_count = data.shape[1]
    # scale to [-1, 1], put all channels after each other
    evens = data[:, 0::2]
    odds = data[:, 1::2]
    y = np.vstack([MinMaxScaler(feature_range=(-1, 1)).fit_transform(data[:, i:i+1]) for i in range(channel_count)])

    save_audio(outfile, y, sample_rate)
    return 0


def auto_process(queue, vector_fn=None, processname='brainsound', checkpoint=None, verbose=False):
    basepath = os.path.dirname(queue[0])
    soundpath = basepath + '/sound/'
    if not (os.path.exists(soundpath)):
        os.mkdir(soundpath)
        print("Made path: {}".format(soundpath))
    # for i, path in enumerate(queue):
    for i in tqdm.tqdm(range(len(queue))):
        path = queue[i]
        # if verbose:
        #     sys.stdout.write('\r{} of {}: {}'.format(i, len(queue), path))
        #     sys.stdout.flush()
        try:
            result = auralize(path, outpath=soundpath, verbose=verbose)

        except Exception as exc:
            if verbose:
                print(exc.message)
    print('\nDone processing')



if __name__ == '__main__':
    input_dir = '../input/train_1/'
    # basedir = '/home/mike/Dow'
    for f in ['1_100_0', '1_100_1', '1_101_0', '1_101_1', '1_102_0', '1_102_1']:
        auralize('%s/%s.mat' % (input_dir, f))