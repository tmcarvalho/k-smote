# %%
from os import walk
from sdv.tabular import GaussianCopula

# %%
def synt_gaussianCopula(data):
    model = GaussianCopula()
    model.fit(data)
    new_data = model.sample(num_rows=len(data))

    return new_data

# %%
original_folder = '../original'
_, _, input_files = next(walk(f'{original_folder}'))

not_considered_files = [0,1,3,13,23,28,34,36,40,48,54,66,87]
for idx,file in enumerate(input_files):
    if int(file.split(".csv")[0]) not in not_considered_files:
        print(idx)
        print(file)
        synt_gaussianCopula(original_folder, file)