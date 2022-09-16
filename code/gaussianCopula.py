# %%
from os import sep, walk
import pandas as pd
from collections import defaultdict
from sklearn.preprocessing import LabelEncoder
from sdv.tabular import GaussianCopula

# %%
def synt_gaussianCopula(original_folder, file):
    output_interpolation_folder = '../output/oversampled/gaussianCopula/'
    data = pd.read_csv(f'{original_folder}/{file}')

    # transform target to string because integer targets are not well synthesised
    data[data.columns[-1]] = data[data.columns[-1]].astype(str)

    model = GaussianCopula()
    model.fit(data)
    new_data = model.sample(num_rows=len(data))

    # save synthetic data
    new_data.to_csv(
        f'{output_interpolation_folder}{sep}ds{file.split(".csv")[0]}_gaussianCopula.csv',
        index=False)    


# %%
original_folder = '../original'
_, _, input_files = next(walk(f'{original_folder}'))

not_considered_files = [0,1,3,13,23,28,34,36,40,48,54,66,87]
for idx,file in enumerate(input_files):
    if int(file.split(".csv")[0]) not in not_considered_files:
        print(idx)
        print(file)
        synt_gaussianCopula(original_folder, file)
# %%
