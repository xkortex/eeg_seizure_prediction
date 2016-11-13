from __future__ import print_function
import os, sys
from tqdm import tqdm
import time

import simple_metrics
from ..pipeline import brain2sound, general_vectorize, sampen


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


def run_a_process(processname, queue):
    vector_fn = None
    vec_name = 'foo'
    # todo: homogenize the pipeline process for vectorizing
    if processname == 'simple_metric':
        auto_process = general_vectorize.auto_process
        vector_fn = simple_metrics.vector_metric
        vec_name = 'simple'

    elif processname == 'show_queue':
        auto_process = queue_auto_process

    elif processname == 'brainsound':
        auto_process = brain2sound.auto_process

    elif processname == 'generalvec':
        auto_process = general_vectorize.auto_process
        vector_fn = general_vectorize.vector_ridiculog
        vec_name = 'ridiculog'

    elif processname == 'sampen':
        auto_process = general_vectorize.auto_process
        vector_fn = sampen.sampen_eeg
        vec_name = 'sampen'


    else:
        print("No valid process selected")
        raise ValueError("No valid process selected")

    results = auto_process(queue, vector_fn, vec_name)