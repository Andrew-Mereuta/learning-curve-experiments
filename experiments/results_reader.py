from os.path import exists
import matplotlib.pyplot as plt

import pandas as pd

dir = '../'

file = dir + '11_extrapolations.gz'
if exists(file):
    results = pd.read_pickle(file)
    results
print('done')
