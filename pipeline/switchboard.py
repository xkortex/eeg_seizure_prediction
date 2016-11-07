from __future__ import print_function
import os, sys
from tqdm import tqdm
import time

import simple_metrics
from ..pipeline import brain2sound


def queue_auto_process(queue):
    # for i, path in enumerate(queue):
    for i in tqdm(range(len(queue))):
        # sys.stdout.write('\r{} of {}'.format(i, len(queue)))
        # sys.stdout.flush()
        time.sleep(.001)
    print('\nDone')
    return queue


def run_a_process(processname, queue):
    if processname == 'simple_metric':
        auto_process = simple_metrics.auto_process


    elif processname == 'show_queue':
        auto_process = queue_auto_process


    elif processname == 'brainsound':
        auto_process = brain2sound.auto_process

    else:
        print("No valid process selected")
        raise ValueError("No valid process selected")

    results = auto_process(queue)