# %%
from os import sep, walk
import re
import pandas as pd
import numpy as np
from dpart.engines import DPsynthpop, PrivBayes, Independent

# %%

def keep_numbers(data):
    """mantain correct data types according to the data"""
    for col in data.columns:
        # transform strings to digits
        if isinstance(data[col].iloc[0], str) and data[col].iloc[0].isdigit():
            data[col] = data[col].astype(float)
        # remove trailing zeros
        if isinstance(data[col].iloc[0], (int, float)):
            if int(data[col].iloc[0]) == float(data[col].iloc[0]):
                data[col] = data[col].astype(int)
    return data


def synt_dpart(original_folder, file):
    output_interpolation_folder = 'output'
    data = pd.read_csv(f'{original_folder}/{file}')

    # get 80% of data to synthesise
    indexes = np.load('indexes.npy', allow_pickle=True).item()
    indexes = pd.DataFrame.from_dict(indexes)

    f = list(map(int, re.findall(r'\d+', file.split('_')[0])))
    index = indexes.loc[indexes['ds']==str(f[0]), 'indexes'].values[0]
    data_idx = list(set(list(data.index)) - set(index))
    data = data.iloc[data_idx, :]
    data = keep_numbers(data)

    # transform target to string because integer targets are not well synthesised
    data[data.columns[-1]] = data[data.columns[-1]].astype(str)
    col_ = data.columns
    X_bounds = {}
    for col in col_:
        if data[col].dtype == np.object_:
            if data[col].str.contains("/").any() or data[col].str.contains("").any():
                data[col] = data[col].apply(lambda x: x.replace("/", "-"))
            col_stats = data[col].unique().tolist()
            col_stats_dict = {'categories': col_stats}
            
        else:
            col_stats_dict = {'min': data[col].min(),
                              'max': data[col].max()}
        X_bounds.update({col: col_stats_dict})

    epsilon = [0.01, 0.1, 0.2, 0.5, 1.0, 5.0, 10.0]
    for ep in epsilon:
        dpart_dpsp = Independent(epsilon=ep, bounds=X_bounds)
        dpart_dpsp.fit(data)
        # print(len(data))
        synth_df = dpart_dpsp.sample(len(data))
        # save synthetic data
        synth_df.to_csv(
            f'{output_interpolation_folder}{sep}ds{file.split(".csv")[0]}_dpart_ep{ep}.csv',
            index=False) 
    print(data.shape)


# %%
original_folder = 'original'
_, _, input_files = next(walk(f'{original_folder}'))

not_considered_files = [0,1,3,13,23,28,34,36,40,48,54,66,87]
for idx,file in enumerate(input_files):
    if int(file.split(".csv")[0]) not in not_considered_files:
        print(idx)
        print(file)
        synt_dpart(original_folder, file)

