from os.path import exists

import pandas as pd
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

dir = '../data/experiment/'
openmlid = 11

lda = LinearDiscriminantAnalysis()
class_name = lda.__class__.__module__ + '.' + lda.__class__.__name__
print(class_name)

file = dir + str(openmlid) + '_results.gz'
if exists(file):
    results = pd.read_pickle(file)
    for v in results['test_mean_squared_error']:
        if v >= 1:
            print(v)
print('done')
