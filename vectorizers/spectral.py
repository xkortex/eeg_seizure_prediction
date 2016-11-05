import numpy as np
from scipy import fftpack, signal


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


def spectrogram(rawdata, nchunk=1024,
                windowStep=4,  # subdivide the chunk size in order to get a rolling window
                absLog=False,
                hardCutoff=100,
                fs=400
                ):
    nsamp0 = rawdata.shape[0]
    spec_ary = []
    cutIndex = hardCutoff * nchunk / fs

    step = nchunk // windowStep
    for i in range(0, nsamp0, step):
        spectrum = fftpack.fft(rawdata[i:i + nchunk], axis=0)[:cutIndex]
        if spectrum.shape[0] == cutIndex:  # discard data that doesn't fit right because it messes up the array
            spec_ary.append(spectrum)
            #             print(spectrum.shape)
    spec_ary = np.array(spec_ary)
    if absLog:
        spec_ary = np.log(np.abs(spec_ary))
    else:
        spect_ary = np.abs(spec_ary)

    return spec_ary


