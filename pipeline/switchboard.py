from __future__ import print_function
import os, sys
from tqdm import tqdm
import time

import simple_metrics
from ..pipeline import brain2sound, general_vectorize, sampen
from ..vectorizers import kludge_mh


def queue_auto_process(queue, vector_fn=None):
    # for i, path in enumerate(queue):
    print('Length of queue: {}'.format(len(queue)))
    for i in tqdm(range(len(queue))):
        # sys.stdout.write('\r{} of {}'.format(i, len(queue)))
        # sys.stdout.flush()
        sys.stdout.write('\r{}'.format(queue[i]))
        sys.stdout.flush()
        # time.sleep(.001)
        # print('\r',queue[i])
    print('\nDone')
    return queue


def run_a_process(processname, queue, verbose=False):
    if processname is None:
        raise ValueError("No valid process selected")
    vector_fn = None
    vec_name = 'foo'
    special_vecs = ['show_queue', 'brainsound']
    process_vecs = {'simple_metric': simple_metrics.vector_metric,
                    'logfourier': general_vectorize.vector_ridiculog,
                    'sampen': sampen.sampen_eeg,
                    'ftfc': kludge_mh.vf_fft_timefreqcorr,
                    'fftsplit': general_vectorize.vector_fftsplit,
                    }
    # todo: homogenize the pipeline process for vectorizing
    if processname in special_vecs:
        if processname == 'show_queue':
            auto_process = queue_auto_process

        elif processname == 'brainsound':
            auto_process = brain2sound.auto_process


    elif processname in process_vecs:
        auto_process = general_vectorize.auto_process
        vector_fn = process_vecs[processname]

    else:
        raise ValueError("Invalid process: {}".format(processname))

    results = auto_process(queue, vector_fn, processname, verbose=verbose)
