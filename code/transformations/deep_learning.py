# %%
import argparse
from os import sep
import re
import pandas as pd
import numpy as np
from sdv.metadata import SingleTableMetadata
from sdv.single_table import CTGANSynthesizer, TVAESynthesizer, CopulaGANSynthesizer

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


def synth(msg):
    """Synthesize data using a deep learning model

    Args:
        msg (str): name of the file, technique and parameters.
        
    Returns:
        None
    """
    print(msg)
    output_interpolation_folder = 'output/oversampled/deep_learning'
    f = list(map(int, re.findall(r'\d+', msg.split('_')[0])))
    print(str(f[0]))
    data = pd.read_csv(f'original/{str(f[0])}.csv')
    print(data.shape)
    # get 80% of data to synthesise
    indexes = np.load('indexes.npy', allow_pickle=True).item()
    indexes = pd.DataFrame.from_dict(indexes)

    index = indexes.loc[indexes['ds']==str(f[0]), 'indexes'].values[0]
    data_idx = list(set(list(data.index)) - set(index))
    data = data.iloc[data_idx, :]
    print(data.shape)
    # encode string with numbers to numeric and remove trailing zeros
    data = keep_numbers(data)

    # transform target to string because integer targets are not well synthesised
    data[data.columns[-1]] = data[data.columns[-1]].astype(str)
    
    technique = msg.split('_')[1]
    print(technique)
    ep = list(map(int, re.findall(r'\d+', msg.split('_')[2])))[0]
    bs = list(map(int, re.findall(r'\d+', msg.split('_')[3])))[0]
    
    metadata = SingleTableMetadata()
    metadata.detect_from_dataframe(data=data)

    print("epochs: ", ep)
    print("batch_size: ", bs)
    if technique=='CTGAN':
        model = CTGANSynthesizer(metadata, epochs=ep, batch_size=bs)
    elif technique=='TVAE':
        model = TVAESynthesizer(metadata, epochs=ep, batch_size=bs)
    else:
        model = CopulaGANSynthesizer(metadata, epochs=ep, batch_size=bs)

    # Generate synthetic data 
    model.fit(data)
    # Generate synthetic data
    new_data = model.sample(num_rows=len(data))             

    # Save the synthetic data
    new_data.to_csv(
        f'{output_interpolation_folder}{sep}{msg}.csv',
        index=False)
        
synth(args.input_file)
