# %%
from os import sep, walk
import pandas as pd
import ast

# %%
output_folder = '../PPT_ARX/Cleaned/'
transformed_folder = '../PPT_ARX'

_, _, input_files = next(walk(f'{transformed_folder}'))

# %%
for file in input_files:
    df = pd.read_csv(f'{transformed_folder}/{file}')

    for i, col in enumerate(df.select_dtypes(include=object).columns):
        # remove columns with all values '*'
        try:
            if df[col].nunique()==1:
                df.drop(col, axis=1, inplace=True)
            
        except: pass
        try:
            # select lower limit in intervals
            if '[' in df[col][0]:
                df[col] = df[col].apply(lambda x: x[:-1] + x[-1].replace('[', ']'))
                df[col] = df[col].apply(lambda x: ast.literal_eval(x)[0] if x!='*' else x)

                try: 
                    df[col] = pd.to_numeric(df[col])
                except: continue

        except: continue

    df.to_csv(f'{output_folder}{sep}{file}', index=False)
# %%
