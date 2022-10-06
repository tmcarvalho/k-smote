# %%
from os import sep, walk
import pandas as pd
from sdv.tabular import TVAE

# %%
epochs=[100, 200]
batch_size=[50, 100]
embedding_dim=[12, 64]

def synt_TVAE(original_folder, file):
    output_interpolation_folder = '../output/oversampled/TVAE/'
    data = pd.read_csv(f'{original_folder}/{file}')

    # transform target to string because integer targets are not well synthesised
    data[data.columns[-1]] = data[data.columns[-1]].astype(str)

    for ep in epochs:
        for bs in batch_size:
            for ed in embedding_dim:
                model = TVAE(epochs=ep, batch_size=bs, embedding_dim=ed)
                model.fit(data)
                new_data = model.sample(num_rows=len(data))

                # save synthetic data
                new_data.to_csv(
                    f'{output_interpolation_folder}{sep}ds{file.split(".csv")[0]}_TVAE_ep{ep}_bs{bs}_ed{ed}.csv',
                    index=False)   

# %%
original_folder = '../original'
_, _, input_files = next(walk(f'{original_folder}'))

not_considered_files = [0,1,3,13,23,28,34,36,40,48,54,66,87]
for idx,file in enumerate(input_files):
    if int(file.split(".csv")[0]) not in not_considered_files:
        print(idx)
        print(file)
        synt_TVAE(original_folder, file)
# %%
