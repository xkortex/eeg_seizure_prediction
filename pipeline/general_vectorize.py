from __future__ import print_function, division
import os, sys
import numpy as np, pandas as pd
import tqdm
import time

from ..dio import dataio
from ..vectorizers import spectral

def auto_process(queue, verbose=True):
    vec_ary = []
    name_ary = []
    for i in tqdm.tqdm(range(len(queue))):
        path = queue[i]
        try:
            data = dataio.get_matlab_eeg_data_ary(path)
            vec1 = spectral.ridiculous_log_transform(data)
            # vec2 = spectral.ridiculous_log_transform(data)
            vec_ary.append(vec1)
            name_ary.append(path)
        except Exception as exc:
            print(exc.message)
    print('\nDone processing')
    name_ary = pd.DataFrame(name_ary, columns=['path'])
    filename = 'vec_{}'.format(time.time())
    np.save(filename, vec_ary)
    name_ary.to_csv(filename+'_name.csv')
