import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from os.path import exists
import os

base_dir = '../lcdb-orig/'
a_postfix = '_anchors_scores_example.gz'
e_postfix = '_extrapolations.gz'
m_postfix = '_metrics_example.gz'
pca_percentages = [0.5, 0.7, 0.9]

openmlid_dirs = os.listdir(base_dir)
for openmlid in openmlid_dirs:
    openmlid_dir = base_dir + openmlid + '/'
    learner_dirs = os.listdir(openmlid_dir)
    print(openmlid)

    for learner in learner_dirs:
        learner_dir = openmlid_dir + learner + '/'

        for pca in pca_percentages:
            a_file = learner_dir + str(pca) + a_postfix
            e_file = learner_dir + str(pca) + e_postfix
            m_file = learner_dir + str(pca) + m_postfix
            line = pd.read_pickle(a_file)
            rows = pd.read_pickle(e_file)
            m_rows = pd.read_pickle(m_file)

            m_rows = m_rows.rename(columns={"max anchor seen": "max_anchor_seen"})

            new_rows = rows.join(m_rows.set_index(['max_anchor_seen', 'curve_model']), on=['max_anchor_seen', 'curve_model'], how='left')
            # print(new_rows.shape)
            anchor_prediction_array = np.tile(line['anchor_prediction'].values[0], (len(rows), 1))
            new_rows['anchor_prediction'] = anchor_prediction_array.tolist()

            score_array = np.tile(line['score'].values[0], (len(new_rows), 1))
            new_rows['score'] = score_array.tolist()

            new_rows.to_pickle(learner_dir + '_' + str(pca) + '_plot_summary.gz')
