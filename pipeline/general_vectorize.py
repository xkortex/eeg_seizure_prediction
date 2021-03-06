from __future__ import print_function, division
from builtins import input
import os, sys
import numpy as np, pandas as pd
import json
import tqdm
import time

from ..dio import dataio
from ..vectorizers import spectral

# def auto_process(queue, verbose=True):
#     vec_ary = []
#     name_ary = []
#     for i in tqdm.tqdm(range(len(queue))):
#         path = queue[i]
#         try:
#             data = dataio.get_matlab_eeg_data_ary(path)
#             vec1 = spectral.ridiculous_log_transform(data)
#             # vec2 = spectral.ridiculous_log_transform(data)
#             vec_ary.append(vec1)
#             name_ary.append(path)
#         except Exception as exc:
#             print(exc.message)
#     print('\nDone processing')
#     name_ary = pd.DataFrame(name_ary, columns=['path'])
#     filename = 'vec_{}'.format(time.time())
#     np.save(filename, vec_ary)
#     name_ary.to_csv(filename+'_name.csv')

def vector_ridiculog(data, verbose=False):
    return spectral.ridiculous_log_transform(data)

def vector_split(data, verbose=False):
    return spectral.resamp_and_chunk(data, windowRatio=1)

def vector_fftsplit(data, verbose=False):
    return spectral.resamp_and_chunk(data, chunksize=5120, applyFFT=True, downResult=8)

def null_vector_fn(data, verbose=False):
    return np.array(0)

def auto_process(queue, vector_fn=None, vec_name='foo', checkpoint=10, split=1, verbose=False):
    if vector_fn is None:
        vector_fn = null_vector_fn
    notes = input("Please enter a note: ")

    basedir = os.path.dirname(queue[0]) + '/'
    time_start = int(time.time())
    time_str = str(time_start)[-7:-3] + '_' + str(time_start)[-3:]
    meta = {'notes': notes, 'basedir': basedir, 'time_start': time_start, 'length': len(queue)}
    total_len = len(queue)
    minibatch_len = total_len // split
    for k in range(split):
        print('Block {} of {}'.format(k, split))
        vec_ary = []
        label_ary = []
        name_ary = []


        filename = 'vec_{}_{}_{}'.format(vec_name, time_str, k)

        for i0 in tqdm.tqdm(range(k*minibatch_len, (k+1)*minibatch_len)):
            # i = i0 + k*split
            i = i0
            path = queue[i]
            try:
                try:
                    data = dataio.get_matlab_eeg_data_ary(path)
                    valpercent = dataio.validcount(data)
                    vec1 = vector_fn(data, verbose=verbose)
                    vec_ary.append(vec1)
                    name_ary.append([path, valpercent])
                # except IndexError as exc:
                except Exception as exc:
                    exc_type, exc_obj, exc_tb = sys.exc_info()
                    fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
                    # print(exc. exc.message)
                    print(exc_type, fname, exc_tb.tb_lineno)
                if i % checkpoint == 0:
                    dataio.dump_data(vec_ary, name_ary, meta, filename, basedir)
            except Exception as exc:
                print("REALLY BAD ERROR")
                exc_type, exc_obj, exc_tb = sys.exc_info()
                fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
                # print(exc. exc.message)
                print(exc_type, fname, exc_tb.tb_lineno)
        dataio.dump_data(vec_ary, name_ary, meta, filename, basedir)

        ## end k loop

    time_end = time.time()
    time_total = time_end-time_start
    meta.update({'time_end': time_end, 'runtime': time_total, 'avg_time': time_total/len(queue)})

    print('\nDone processing')
