"""Apply sinthetisation with SMOTE
This script will apply SMOTE technique in the single out cases.
"""
# %%
from os import sep, getcwd
import random
from imblearn.over_sampling import SMOTE
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from collections import ChainMap
from kanon import single_outs
from record_linkage import record_linkage
from modeling import evaluate_model

# %%
data = pd.read_csv(f'{getcwd()}{sep}dataset.csv')
data, key_vars = single_outs(data)
# %% apply LabelEncoder beacause of smote
data = data.apply(LabelEncoder().fit_transform)
# %% create list of possiblilities to interpolation
n = pd.Series(np.arange(2*data['single_out'].value_counts()[1],
                int(data['single_out'].value_counts()[0]/2), 1))
# %%
knn = [1, 2, 3, 4, 5]
validation = {}
test = {}
for nn in knn:
    smote = SMOTE(random_state=42,
                k_neighbors=nn,
                sampling_strategy={1: random.choice(n)})
    # fit predictor and target variable
    X = data[data.columns[:-1]]
    y = data.iloc[:, -1]
    x_smote, y_smote = smote.fit_resample(X, y)

    # add single out to apply record linkage
    x_smote['single_out'] = y_smote
    # remove original single outs from oversample
    oversample = x_smote.copy()
    oversample = oversample.drop(data[data['single_out']==1].index).reset_index(drop=True)
    # save oversampled data
    oversample.to_csv(f'{getcwd()}{sep}output{sep}oversampled_data{sep}oversample_knn{nn}.csv',
        index=False)
    
    # apply record linkage only to single outs!!
    oversample_singleouts = oversample[oversample['single_out']==1]
    original_singleouts = data[data['single_out']==1]
    potential_matches = record_linkage(oversample_singleouts, original_singleouts, key_vars)
    # save record linkage results
    potential_matches.to_csv(
        f'{getcwd()}{sep}output{sep}record_linkage{sep}potential_matches_knn{nn}.csv',
        index=False)

    # prepare data to modeling
    X, y = data.iloc[:, :-2], data.iloc[:, -2]
    # predictive performance
    validation_result, test_result = evaluate_model(X, y, validation, test, nn)
    print(len(validation_result))
    validation = ChainMap(validation,validation_result)
    print(len(validation))
    test = ChainMap(test,test_result)

# %%
# save validation and test results
np.save(
    f'{getcwd()}{sep}output{sep}modeling{sep}oversampled{sep}validation{sep}validation.npy',
    validation)
np.save(
    f'{getcwd()}{sep}output{sep}modeling{sep}oversampled{sep}test{sep}test.npy',
    test)