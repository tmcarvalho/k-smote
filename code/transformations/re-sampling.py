""" Apply smote, under and over
This script interpolate data using smote, under and over techniques.
"""
# %%
from os import walk
import pandas as pd
import numpy as np
import re
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import SMOTE, BorderlineSMOTE


def keep_numbers(data):
    """fix data types according to the data"""
    data_types = data.copy()
    for col in data.columns:
        # transform numerical strings to digits
        if isinstance(data[col].iloc[0], str) and data[col].iloc[0].isdigit():
            data[col] = data[col].astype(float)
        # remove trailing zeros
        if isinstance(data[col].iloc[0], (int, float)):
            if int(data[col].iloc[0]) == float(data[col].iloc[0]):
                data[col] = data[col].astype(int)
            else: data[col] = data_types[col].astype(float)
    return data, data_types

# %%
def interpolation(original_folder, file):
    """Generate several interpolated data sets.

    Args:
        original_folder (string): path of original folder
        file (string): name of file
    """
    output_interpolation_folder = '../../output/oversampled/re-sampling'

    data = pd.read_csv(f'{original_folder}/{file}')
    # get 80% of data to synthesise
    indexes = np.load('../../indexes.npy', allow_pickle=True).item()
    indexes = pd.DataFrame.from_dict(indexes)

    f = list(map(int, re.findall(r'\d+', file.split('_')[0])))
    index = indexes.loc[indexes['ds']==str(f[0]), 'indexes'].values[0]
    data_idx = list(set(list(data.index)) - set(index))
    data = data.iloc[data_idx, :]

    # encode string with numbers to numeric and remove trailing zeros
    data, _ = keep_numbers(data)
    
    knn = [1, 3, 5]
    # percentage of majority and minority class
    ratios_smote = [0.5, 0.75, 1]
    ratios_under = [0.25, 0.5, 0.75, 1]
    for nn in knn:
        for smote in ratios_smote:
            try:
                smote_samp = SMOTE(random_state=42,
                            k_neighbors=nn,
                            sampling_strategy=smote)
                border_smote = BorderlineSMOTE(random_state=42,
                            k_neighbors=nn,
                            sampling_strategy=smote)
                # fit predictor and target variable
                X = data[data.columns[:-1]]
                y = data.iloc[:, -1]
                x_smote, y_smote = smote_samp.fit_resample(X, y)
                x_bordersmote, y_bordersmote = border_smote.fit_resample(X, y)
                
                # add target
                x_smote[data.columns[-1]] = y_smote
                x_bordersmote[data.columns[-1]] = y_bordersmote

                x_smote.to_csv(f'{output_interpolation_folder}/ds{file.split(".csv")[0]}_smote_knn{nn}_per{smote}.csv', index=False)
                x_bordersmote.to_csv(f'{output_interpolation_folder}/ds{file.split(".csv")[0]}_bordersmote_knn{nn}_per{smote}.csv', index=False)

            except: pass         
    
    for under in ratios_under:
        try:
            under_samp = RandomUnderSampler(random_state=42,
                        sampling_strategy=under)
            # fit predictor and target variable
            X = data[data.columns[:-1]]
            y = data.iloc[:, -1]
            x_under, y_under = under_samp.fit_resample(X, y)
            
            # add target
            x_under[data.columns[-1]] = y_under

            x_under.to_csv(f'{output_interpolation_folder}/ds{file.split(".csv")[0]}_under_per{under}.csv', index=False)
        except: pass    

# %%
import os
original_folder = '../../original'
# _, _, input_files = next(walk(f'{original_folder}'))
input_files = [f for f in os.listdir(original_folder) if f.endswith('.csv')]

not_considered_files = [0,1,3,13,23,28,34,36,40,48,54,66,87]
for idx,file in enumerate(input_files):
    if int(file.split(".csv")[0]) not in not_considered_files:
        print(idx)
        interpolation(original_folder, file)

# %%
