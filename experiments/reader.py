from os.path import exists

import pandas as pd

dir = '../data/experiment/'
openmlid = 3

file = dir + str(openmlid) + '_results.gz'
if exists(file):
    results = pd.read_pickle(file)
    results.to_csv('3-accuracy.csv', index=False)
    for v in results['score_valid']:
        if v >= 1:
            print(v)
print('done')
