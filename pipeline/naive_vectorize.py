import glob
import numpy as np

from ..vectorizers import naive
from ..dio import dataio


def auto_process(basepath):
    filenames = glob.glob(basepath + '*.mat');
