from __future__ import print_function, division
import os, sys
import numpy as np, pandas as pd
import tqdm
import time
from subprocess import Popen, PIPE


from ..dio import dataio
from ..vectorizers import spectral

def sampen_chan(channeldata):
    spool = []
    for i in range(len(channeldata)):
        spool.append('{}'.format(channeldata[i]))
    stream = '\n'.join(spool)
    cproc = Popen("/home/mike/src/SampEn/sampen", stdin=PIPE, stdout=PIPE)
    out, err = cproc.communicate(stream)
    print(err)
    return out

def sampen_eeg(data):
    sampen = []
    for chan in range(16):
        out = sampen_chan(data[:,chan])
        sampen.append(out)
    return sampen


def auto_process(queue, verbose=True):
    vec_ary = []
    name_ary = []
    for i in tqdm.tqdm(range(len(queue))):
        path = queue[i]
        try:
            data = dataio.get_matlab_eeg_data_ary(path)
            vec1 = sampen_eeg(data)
            vec_ary.append(vec1)
            name_ary.append(path)
        except Exception as exc:
            print(exc.message)
    print('\nDone processing')
    name_ary = pd.DataFrame(name_ary, columns=['path'])
    filename = 'vec_{}'.format(time.time())
    np.save(filename, vec_ary)
    name_ary.to_csv(filename+'_name.csv')