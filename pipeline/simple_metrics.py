from __future__ import print_function, division
import os, sys
import numpy as np, pandas as pd
import tqdm
import time

from ..dio import dataio
from ..vectorizers import naive
from ..msignal import metrics


def vector_metric(data, verbose=False):
    hurst = metrics.hurst(data)
    chanstd = metrics.chanstd(data)
    ccmean, ccstd = metrics.crosscorr_stat(data)

    return np.array([hurst, chanstd, ccmean, ccstd])

def vector_metric_pow(data, verbose=False):
    hurst = metrics.hurst(data)
    chanstd = metrics.chanstd(data)
    ccmean, ccstd = metrics.crosscorr_stat(data)
    rms_powers = np.std(data, axis=0)
    chanstd_pow = np.std(rms_powers)
    chanmean_pow = np.mean(rms_powers)
    met = np.array([hurst, chanstd, ccmean, ccstd, chanstd_pow, chanmean_pow])

    vec = np.concatenate([rms_powers, met], axis=0)
    return vec


def auto_process(queue, verbose=True):
    metric_ary = []
    namelist = []
    for i in tqdm.tqdm(range(len(queue))):
        path = queue[i]
        try:
            data = dataio.get_matlab_eeg_data_ary(path)
            hurst = metrics.hurst(data)
            chanstd = metrics.chanstd(data)
            ccmean, ccstd = metrics.crosscorr_stat(data)
            # hurst = 1
            # chanstd = 1
            metric_ary.append(np.array([hurst, chanstd, ccmean, ccstd]))
            namelist.append(os.path.basename(path))
            if verbose:
                sys.stdout.write('\r{} of {}'.format(i, len(queue)))
                sys.stdout.flush()
        except Exception as exc:
            print(exc.message)
    print('\nDone processing')
    metric_ary = np.nan_to_num(metric_ary)
    meanval = np.mean(metric_ary, axis=0)


    df = pd.DataFrame(metric_ary, columns=['hurst', 'chanstd', 'ccmean', 'ccstd'])
    df['File'] = pd.Series(namelist)
    guess_h = df['hurst'] - meanval[0] # hurst is below mean for seizure
    guess_s = df['chanstd'] - meanval[1] #chanstd is below mean
    guess_dist = guess_h + guess_s # positive score = less likely to seize
    # guess = np.bitwise_and(guess_h, guess_s)
    guess = guess_dist < 0
    guess = guess*1 # hack it back to numerical
    df['Class'] = guess
    print(df)
    outfilename = os.path.dirname(queue[0]) + '/submit_' +  str(time.time()) + '.csv'
    print(outfilename)
    # df = df[['File', 'Class']]
    print('Sum: {} Perc: {}'.format(np.sum(df['Class']), np.mean(df['Class'])))
    df.to_csv(outfilename)

