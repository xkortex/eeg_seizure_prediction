from __future__ import print_function, division
import os, sys
import numpy as np, pandas as pd
import argparse
import tqdm
import time
from io import StringIO
from subprocess import Popen, PIPE
import scipy.signal as signal


from ..dio import dataio
from ..vectorizers import spectral

def sampen_chan(channeldata, max_npts_j=100, maxepoch=2, verbose=False):
    spool = []
    for i in range(len(channeldata)):
        spool.append('{}'.format(channeldata[i]))
    stream = '\n'.join(spool)
    argsend = None#" -j {} -m {}".format(max_npts_j, maxepoch)
    efile = "sampen_j0_1k.o"
    command = "SampEn/{}".format(efile)
    cproc = Popen(command, stdin=PIPE, stdout=PIPE)
    out, err = cproc.communicate(stream)
    if err: print(err)
    if verbose: print('OUT: ', out)
    outdata = np.array(out.split(','))
    # if verbose: print(outdata)

    return outdata

def sampen_eeg(data, down=4, verbose=False):
    sampen = []
    if down != 1:
        data = signal.resample_poly(data, 1, down, axis=0)
    for chan in range(16):
        outdata = sampen_chan(data[:,chan], verbose=verbose)
        sampen.append(outdata)
    return np.array(sampen)



def auto_process(queue, checkpoint=10, verbose=False):
    vec_ary = []
    name_ary = []
    filename = 'vec_sampen_{}'.format(int(time.time()))
    for i in tqdm.tqdm(range(len(queue))):
        path = queue[i]
        try:
            data = dataio.get_matlab_eeg_data_ary(path)
            vec1 = sampen_eeg(data, verbose=verbose)
            vec_ary.append(vec1)
            name_ary.append(path)
        except Exception as exc:
            print(exc.message)
        if i % checkpoint == 0:
            dataio.dump_data(vec_ary, name_ary, filename)

    print('\nDone processing')
    dataio.dump_data(vec_ary, name_ary, filename)


def set_arg_parser():
    parser = argparse.ArgumentParser(description='Process eeg data. See docs/main.txt for more info')
    parser.add_argument('infile', type=str, nargs='?',
                       help='Positional argument: Input file or path. If left blank, then default to the internal test file')
    return parser


if __name__ == '__main__':
    # boilerplate
    parser = set_arg_parser()
    args = parser.parse_args()
    if args.infile is None:
        raise ValueError("no file to process")
    data = dataio.get_matlab_eeg_data_ary(args.infile)
    vec1 = sampen_eeg(data, verbose=False)
    print(vec1)
    print(vec1.shape)

