"""Apply sinthetisation with SMOTE
This script will apply SMOTE technique in the single out cases.
"""
# %%
from os import sep, getcwd
import random
from imblearn.over_sampling import SMOTE
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from kanon import single_outs

# %%
data = pd.read_csv(f'{getcwd()}{sep}dataset.csv')
data, key_vars = single_outs(data)
# %%
# check the target variable that is single out and not single out
data['single_out'].value_counts()
# 0 -> not single out
# 1 -> single out
# %% percentage of single outs
percentage = data['single_out'].value_counts()[1]*100 / data.shape[0]
print(percentage)
# %%
g = sns.countplot(data['single_out'])
g.set_xticklabels(['Not single out','Single out'])
plt.show()
# %%
data = data.apply(LabelEncoder().fit_transform)
# %%
n = pd.Series(np.arange(2*data['single_out'].value_counts()[1],
                int(data['single_out'].value_counts()[0]/2), 1))
# %%
smote = SMOTE(random_state=42,
            k_neighbors=3,
            sampling_strategy={1: random.choice(n)})
# fit predictor and target variable
X = data[data.columns[:-1]]
y = data.iloc[:, -1]
x_smote, y_smote = smote.fit_resample(X, y)
# %%
print('Original dataset shape', data['single_out'].value_counts())
print('Resample dataset shape', y_smote.value_counts())
# %%
print(f'{y_smote.value_counts()[1]-data["single_out"].value_counts()[1]} new observations included')
# %%
original_singleouts_idx = data[data['single_out']==1].index
# %% add single out to merge the sinthetised data with original
x_smote['single_out'] = y_smote
# %% remove original single outs from oversample
oversample = x_smote.copy()
oversample = oversample.drop(original_singleouts_idx).reset_index(drop=True)

# %%
oversample.to_csv(f'{getcwd()}{sep}oversample.csv', index=False)
