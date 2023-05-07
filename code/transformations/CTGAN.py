# %%
from os import sep, walk
import re
import pandas as pd
import numpy as np
from sdv.tabular import CTGAN

# %%
epochs=[100, 200]
batch_size=[50, 100]
embedding_dim=[32, 64]

def synt_ctgan(original_folder, file):
    output_interpolation_folder = '../output/oversampled/deep_learning/'
    data = pd.read_csv(f'{original_folder}/{file}')

    # get 80% of data to synthesise
    indexes = np.load('../indexes.npy', allow_pickle=True).item()
    indexes = pd.DataFrame.from_dict(indexes)

    f = list(map(int, re.findall(r'\d+', file.split('_')[0])))
    index = indexes.loc[indexes['ds']==str(f[0]), 'indexes'].values[0]
    data_idx = list(set(list(data.index)) - set(index))
    data = data.iloc[data_idx, :]

    # transform target to string because integer targets are not well synthesised
    data[data.columns[-1]] = data[data.columns[-1]].astype(str)

    for ep in epochs:
        for bs in batch_size:
            for ed in embedding_dim:
                print("epochs: ", ep)
                print("batch_size: ", bs)
                print("embedding: ", ed)
                model = CTGAN(epochs=ep, batch_size=bs, embedding_dim=ed)
                model.fit(data)
                new_data = model.sample(num_rows=len(data))

                # save synthetic data
                new_data.to_csv(
                    f'{output_interpolation_folder}{sep}ds{file.split(".csv")[0]}_CTGAN_ep{ep}_bs{bs}_ed{ed}.csv',
                    index=False)    
# %%
original_folder = '../original'
_, _, input_files = next(walk(f'{original_folder}'))

not_considered_files = [0,1,3,13,23,28,34,36,40,48,54,66,87]
for idx,file in enumerate(input_files):
    if int(file.split(".csv")[0]) not in not_considered_files:
        print(idx)
        print(file)
        synt_ctgan(original_folder, file)
# %%
