# %%
#!/usr/bin/env python
import argparse
from os import sep
import re
import pandas as pd
import numpy as np
from synthcity.plugins import Plugins

parser = argparse.ArgumentParser(description='Master Example')
# parser.add_argument('--input_folder', type=str, help='Input folder', default="./input")
parser.add_argument('--input_file', type=str, default="none")
args = parser.parse_args()


def keep_numbers(data):
    """Fix data types according to the data"""
    for col in data.columns:
        # Transform numerical strings to digits
        if isinstance(data[col].iloc[0], str) and data[col].iloc[0].isdigit():
            data[col] = data[col].astype(float)
        # Remove trailing zeros
        if isinstance(data[col].iloc[0], (int, float)):
            if int(data[col].iloc[0]) == float(data[col].iloc[0]):
                data[col] = data[col].astype(int)
    return data


def synth_city(msg):
    """Synthesize data using a deep learning model

    Args:
        original_folder (str): Path to the original data folder.
        file (str): Name of the original data file.
        technique (str): Deep learning technique to use. Valid options are 'TVAE', 'CTGAN', and 'CopulaGAN'.

    Returns:
        None
    """
    print(msg)
    output_interpolation_folder = 'output/oversampled/city_data'

    f = list(map(int, re.findall(r'\d+', msg.split('_')[0])))
    data = pd.read_csv(f'original/{str(f[0])}.csv')

    # get 80% of data to synthesise
    indexes = np.load('indexes.npy', allow_pickle=True).item()
    indexes = pd.DataFrame.from_dict(indexes)

    index = indexes.loc[indexes['ds']==str(f[0]), 'indexes'].values[0]
    data_idx = list(set(list(data.index)) - set(index))
    data = data.iloc[data_idx, :]

    # encode string with numbers to numeric and remove trailing zeros
    data = keep_numbers(data)

    # transform target to string because integer targets are not well synthesised
    data[data.columns[-1]] = data[data.columns[-1]].astype(str)
    
    technique = msg.split('_')[1]

    if technique == 'dpgan':
        epo = list(map(int, re.findall(r'\d+', msg.split('_')[2])))[0]
        bs = list(map(int, re.findall(r'\d+', msg.split('_')[3])))[0]
        epi = list(map(float, re.findall(r'\d+\.\d+', msg.split('_')[4])))[0]
        model = Plugins().get("dpgan", n_iter=epo, batch_size=bs, epsilon=epi)

    if technique == 'pategan':
        epo = list(map(int, re.findall(r'\d+', msg.split('_')[2])))[0]
        bs = list(map(int, re.findall(r'\d+', msg.split('_')[3])))[0]
        epi = list(map(float, re.findall(r'\d+\.\d+', msg.split('_')[4])))[0]
        model = Plugins().get("pategan", n_iter=epo, batch_size=bs, epsilon=epi)
    
    if technique == 'privbayes':
        epi = list(map(float, re.findall(r'\d+\.\d+', msg.split('_')[2])))[0]
        model = Plugins().get("privbayes", epsilon=epi)

    # Fit the model to the data
    model.fit(data)
    # Generate synthetic data
    new_data = model.generate(count=len(data))
    new_data = new_data.dataframe()

    # Save the synthetic data
    new_data.to_csv(
        f'{output_interpolation_folder}{sep}{msg}.csv',
        index=False)


synth_city(args.input_file)