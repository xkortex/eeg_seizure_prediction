import os
import numpy as np
from scipy import fftpack, signal
import matplotlib.pyplot as plt
import matplotlib
from ..dio import dataio


def vectorize_fft(rawdata, ndim=800,  # number of vector dimensions to output
                  cutoff=40,  # hard cutoff frequency
                  fs=400,  # sample frequency of input signal
                  takeLog=True,  # take log of freq spectrum
                  avgChan=False,  # take average across all channels
                  stdChan=False,  # take stdDev across all channels - VERY INTERESTING
                  sepComplex=False,  # prolly don't need this
                  hilbertize=False):  # don't need this right now
    spectrum = fftpack.fft(rawdata, axis=0)
    nsamp0 = rawdata.shape[0]
    t = np.linspace(0, fs, nsamp0)
    cutIndex = cutoff * nsamp0 / fs
    #     plt.plot(t[:cutIndex],np.abs(spectrum[:cutIndex]))
    #     plt.plot(t, spectrum)
    if avgChan:
        spectrum = np.mean(spectrum, axis=1)
    elif stdChan:
        spectrum = np.std(spectrum, axis=1)
    spectrum = np.abs(spectrum[:cutIndex])
    rs_spectrum = signal.resample(spectrum, ndim, axis=0)
    rs_t = np.linspace(0, cutoff, ndim)

    #     print(rs_spectrum.shape)
    #     print(np.angle(spectrum[:cutIndex]).shape)
    #     plt.plot(t[:cutIndex], np.angle(spectrum[:cutIndex, 0]))
    """ Phase information at this point looks really nasty, so will ignore it"""

    if takeLog:
        rs_spectrum = np.log(rs_spectrum)

    # plt.plot(rs_t, rs_spectrum)
    return rs_spectrum


def spectrogram(rawdata, nchunk=256,
                windowStep=4,  # subdivide the chunk size in order to get a rolling window
                absLog=False,
                hardCutoff=100,
                fs=400,
                window='tukey',
                alpha=0.1,
                mode='abslog'
                ):
    nsamp0, nchan = rawdata.shape
    spec_ary = []
    cutIndex = hardCutoff * nchunk / fs

    step = nchunk // windowStep
    if window == 'tukey':
        window_sig = signal.tukey(nchunk, alpha)
    elif window == 'hann':
        window_sig = signal.hann(nchunk)
    else:
        raise ValueError('Invalid window signal type: {}'.format(window))
    window_ary = np.array([window_sig,]*nchan).T
    # window_ary = np.concatenate(, axis=1)
    # window_sig.reshape((nchunk, 1))
    for i in range(0, nsamp0, step):
        block = rawdata[i:i + nchunk]
        if block.shape[0] != window_ary.shape[0]:
            window_ary = window_ary[:block.shape[0]] # yes I know this mangles the window, but I don't know how else
            # to deal with uneven division of samples at the end. Actually seems to cause no issue
        datachunk = np.prod([block,window_ary], axis=0)

        spectrum = fftpack.fft(datachunk, axis=0)[:cutIndex]
        if spectrum.shape[0] == cutIndex:  # discard data that doesn't fit right because it messes up the array
            spec_ary.append(spectrum)
            #             print(spectrum.shape)
    spec_ary = np.array(spec_ary)
    if mode=='abslog':
        spec_ary = np.log(np.abs(spec_ary))
    elif mode=='phase':
        spec_ary = np.angle(spec_ary)
    else:
        spect_ary = np.abs(spec_ary)

    return spec_ary


def path_to_picname(path):
    dirname = os.path.dirname(path)
    filename = os.path.basename(path)[:-4]
    picname = dirname+'/pics/'+filename+'.png'
    # print( picname)
    return picname

def spec_to_fig(spec, filename=None, cutoff=100, length=600):
    data = np.average(spec[:,:], axis=2).T
    plt.imshow(data, origin='lower', extent=[0, length, 0, cutoff])
    if filename is not None:
        plt.savefig(filename)
        plt.close()
        print(filename)


def file_to_fig(path, nchunk=256, windowStep=4, savefile=False, returnspec=False):
    data = dataio.get_matlab_eeg_data_ary(path)
    spec = spectrogram(data, nchunk=nchunk, windowStep=windowStep, absLog=1, alpha=0.5, window='hann')
    picname = path_to_picname(path) if savefile else None
    spec_to_fig(spec, filename=picname)
    if returnspec:
        return spec

def plt_16x(plt, spec, npow=1):
    """
    Plot all 16 channels of the spectrogram
    :param plt:
    :param spec:
    :param npow:
    :return:
    """
    matplotlib.rcParams['figure.figsize'] = (12, 16)
    fig = plt.figure()
    nmax = 16
    if npow != 1:
        spec = spec**npow
    for i in range(nmax):
        fig.add_subplot(nmax // 2, 2, i + 1)
        plt.imshow(spec[:, :, i].T, extent=[0, 600, 0, 200], origin='lower')
        ax = plt.gca()
        ax.get_yaxis().set_visible(False)
        ax.get_xaxis().set_visible(False)