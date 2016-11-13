from __future__ import print_function, division
import os, sys
import numpy as np, pandas as pd
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

def null_vector_fn(data, verbose=False):
    return np.array(0)

def auto_process(queue, vector_fn=None, vec_name='foo', checkpoint=10, verbose=False):
    if vector_fn is None:
        vector_fn = null_vector_fn
    vec_ary = []
    name_ary = []
    filename = 'vec_{}_{}'.format(vec_name, int(time.time()))
    for i in tqdm.tqdm(range(len(queue))):
        path = queue[i]
        try:
            data = dataio.get_matlab_eeg_data_ary(path)
            vec1 = vector_fn(data, verbose=verbose)
            vec_ary.append(vec1)
            name_ary.append(path)
        except Exception as exc:
            print(exc.message)
        if i % checkpoint == 0:
            dataio.dump_data(vec_ary, name_ary, filename)

    print('\nDone processing')
    dataio.dump_data(vec_ary, name_ary, filename)