# %%
import argparse
from os import sep, walk
import re
import pandas as pd
import numpy as np
from sdv.metadata import SingleTableMetadata
from sdv.single_table import CTGANSynthesizer, TVAESynthesizer, CopulaGANSynthesizer

parser = argparse.ArgumentParser(description='Master Example')
# parser.add_argument('--input_folder', type=str, help='Input folder', default="./input")
parser.add_argument('--input_file', type=str, default="none")
args = parser.parse_args()

# %%
epochs=[100, 200]
batch_size=[50, 100]
embedding_dim=[32, 64] 

# %%

def synth(msg):
    """Synthesize data using a deep learning model

    Args:
        msg (str): name of the file, technique and parameters.
        
    Returns:
        None
    """
    print(msg)
    output_interpolation_folder = 'output/oversampled/deep_learning'
    if msg.split('_')[0] not in ['ds100', 'ds43']:
        f = list(map(int, re.findall(r'\d+', msg.split('_')[0])))
        print(str(f[0]))
        data = pd.read_csv(f'original/{str(f[0])}.csv')

        # get 80% of data to synthesise
        indexes = np.load('indexes.npy', allow_pickle=True).item()
        indexes = pd.DataFrame.from_dict(indexes)

        index = indexes.loc[indexes['ds']==str(f[0]), 'indexes'].values[0]
        data_idx = list(set(list(data.index)) - set(index))
        data = data.iloc[data_idx, :]

        # transform target to string because integer targets are not well synthesised
        data[data.columns[-1]] = data[data.columns[-1]].astype(str)
        
        technique = msg.split('_')[1]
        print(technique)
        ep = list(map(int, re.findall(r'\d+', msg.split('_')[3])))[0]
        bs = list(map(int, re.findall(r'\d+', msg.split('_')[4])))[0]
        
        metadata = SingleTableMetadata()
        metadata.detect_from_dataframe(data=data)

        print("epochs: ", epochs[0])
        print("batch_size: ", batch_size[0])
        if technique=='CTGAN':
            model = CTGANSynthesizer(metadata, epochs=epochs[0], batch_size=batch_size[0], verbose=True)
        elif technique=='TVAE':
            model = TVAESynthesizer(metadata, epochs=epochs[0], batch_size=batch_size[0])
        else:
            model = CopulaGANSynthesizer(metadata, epochs=epochs[0], batch_size=batch_size[0], verbose=True)

        # Generate synthetic data 
        model.fit(data)
        # Generate synthetic data
        new_data = model.sample(num_rows=len(data))             
        new_data_ = pd.concat([new_data, data])

        # Save the synthetic data
        new_data_.to_csv(
            f'{output_interpolation_folder}{sep}{msg}.csv',
            index=False)
        
synth(args.input)
