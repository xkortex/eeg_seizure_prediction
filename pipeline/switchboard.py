from __future__ import print_function
import os, sys

import simple_metrics


def queue_auto_process(queue):
    for i, path in enumerate(queue):
        sys.stdout.write('\r{} of {}'.format(i, len(queue)))
        sys.stdout.flush()
    print('\nDone')
    return queue


def run_a_process(processname, queue):
    if processname == 'simple_metric':
        auto_process = simple_metrics.auto_process


    elif processname == 'show_queue':
        auto_process = queue_auto_process

    else:
        print("No valid process selected")
        raise ValueError("No valid process selected")

    results = auto_process(queue)