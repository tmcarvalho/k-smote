"""Find singleouts to synthetise
This script will create a new varible indicating which observations
correspond to a unique signature.
Such a variable is similar to a binary target variable, where 1 is a singleout
and 0 otherwise.
"""

# %%
from os import sep, getcwd
import pandas as pd
import numpy as np

# %%
data = pd.read_csv(f'{getcwd()}{sep}dataset.csv')
# %%
keyVars = ['age', 'det_ind_code', 'vet_benefits', 'year']
# %%
data_copy = data.copy()
# %%
k = data.groupby(keyVars)[keyVars[0]].transform(len)
# %%
data_copy['single_out'] = None
data_copy['single_out'] = np.where(k == 1, 1, 0)
