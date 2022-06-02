"""This script will choose 20% of random indexes for testing.
20% after the PPT and smote/over/under and 20% before the smote singleouts
beacause in smote the original samples are deleted.
"""

# %%
from os import walk
import random
import pandas as pd
import numpy as np
# %%
# path to input data
input_folder = '../original/'

_, _, input_files = next(walk(f'{input_folder}'))
# %%
idx_dict = {'ds': [], 'indexes':[]}
for i, file in enumerate(input_files):
    dt = pd.read_csv(f'{input_folder}/{file}')
    random_idx = random.sample(list(dt.index), k=int(0.2*len(dt)))
    idx_dict['ds'].append(file.split('.csv')[0])
    idx_dict['indexes'].append(random_idx)
# %%
np.save('../indexes.npy', idx_dict, allow_pickle=True)
# %%
