"""Data de-identification
This script will de-identify the data for 5 set of quasi-identifiers.
"""
# %%
import pandas as pd
import numpy as np
from decimal import Decimal
import transformations
from os import sep, walk
from kanon import single_outs_sets

def change_cols_types(df):
    cols = df.select_dtypes(include=np.number).columns.values
    for col in cols:
        df[col] = df[col].apply(Decimal).astype(str)
        if any('.' in s for s in df[col]):
            df[col] = df[col].astype(float)
            if all([x.is_integer() for x in df[col]]):
                df[col] = df[col].astype(int)
        else:
            df[col] = df[col].astype(int)

        if (df[col].dtype == np.float64) and (df[col].max() >= 1000):
            df[col] = df[col].apply(lambda x: int(x))

    return df

def aux_singleouts(key_vars, dt):
    k = dt.groupby(key_vars)[key_vars[0]].transform(len)
    dt['single_out'] = k
    # dt['single_out'] = np.where(k == 1, 1, 0)
    # dt = dt[dt['single_out']==1]
    return dt

def apply_transformations(obj, key_vars, tech_comb, parameters, result):
    """Apply transformations

    Args:
        obj (pd.Dataframe): dataframe for de-identification
        key_vars (list): list of quasi-identifiers
        tech_comb (list): combination of techniques
        parameters (list): parameters of techniques
        result (dictionary): dictionary to store results

    Returns:
        tuple: last transformed variant, dictionary of results
    """

    if 'sup' in tech_comb:
        param = parameters[[tech_comb.index(l) for l in tech_comb if 'sup'==l][0]] if len(tech_comb)>1 else parameters
        obj = transformations.suppression(obj, key_vars, uniq_per=param)    
    if 'topbot' in tech_comb:
        param = parameters[[tech_comb.index(l) for l in tech_comb if 'topbot'==l][0]] if len(tech_comb)>1 else parameters
        obj = transformations.topBottomCoding(obj, key_vars, outlier=param)
    if 'round' in tech_comb:
        param = parameters[[tech_comb.index(l) for l in tech_comb if 'round'==l][0]] if len(tech_comb)>1 else parameters
        obj = transformations.rounding(obj, key_vars, base=param)
    if 'globalrec' in tech_comb:
        param = parameters[[tech_comb.index(l) for l in tech_comb if 'globalrec'==l][0]] if len(tech_comb)>1 else parameters
        obj = transformations.globalRecoding(obj, key_vars, std_magnitude=param)
    
    obj = aux_singleouts(key_vars, obj)
    # if transformed variant is different from others
    if (len(result['combination'])==0) or (not(any(x.equals(obj) for x in result['transformedDS'])) and len(result['combination'])!=0):
        result['combination'].append(tech_comb)  
        result['parameters'].append(parameters) 
        result['key_vars'].append(key_vars)
        result['transformedDS'].append(obj)

    return obj, result


def process_transformations(df, key_vars):
    """Find combinations, respective parameterisation and apply transformations.

    Args:
        df (pd.Dataframe): dataframe for de-identification
        key_vars (list): list of quasi-identifiers

    Returns:
        dictionary: set transformed variants
    """

    df_val = df.copy()

    # create combinations adequate to the data set
    comb_name, param_comb = transformations.parameters(df_val, key_vars)

    result = {'combination': [], 'parameters': [], 'key_vars': [], 'transformedDS': []}

    # transform data
    df_transf = df_val.copy()
    if len(param_comb) > 1:
        for comb in param_comb:
            # apply transformations
            df_transf, result = apply_transformations(df_transf, key_vars, comb_name, comb, result)
    else:
        # apply transformations
        for i in param_comb[0]:
            df_transf, result = apply_transformations(df_transf, key_vars, comb_name, i, result)            

    return result


# %% save best transformed variant for each combined technique
# path to input data
input_folder = '../original/'
transformed_folder = '../PPT'

_, _, input_files = next(walk(f'{input_folder}'))

not_considered_files = [0,1,3,13,23,28,34,36,40,48,54,66,87]
set_qis = {'ds':[], 'set_key_vars':[]}

for idx, file in enumerate(input_files):
    
    if int(file.split(".csv")[0]) not in not_considered_files:
        df =  pd.read_csv(f'{input_folder}/{file}')
        df = change_cols_types(df)
        # get index
        file_idx = int(file.split('.')[0])
        _, set_key_vars = single_outs_sets(df)
        
        if len(set_key_vars) == 5:
            print(f'file: {file}')
            set_qis['ds'].append(file_idx)
            set_qis['set_key_vars'].append(set_key_vars)
            
            for j, key_vars in enumerate(set_key_vars):
                # apply de-identification to the set of key vars    
                result = process_transformations(df, key_vars)
                # transform dict to dataframe
                res_df = pd.DataFrame(result)

                for i in range(len(res_df)):
                    per = (len(res_df['transformedDS'][i].loc[res_df['transformedDS'][i].single_out==1]) * 100) / len(res_df['transformedDS'][i]) if len(res_df['transformedDS'][i][res_df['transformedDS'][i].single_out==1])!=0 else 0
                    if per <= 35:
                        res_df['transformedDS'][i].to_csv(f'{transformed_folder}{sep}ds{str(file_idx)}_transf{str(i)}_qi{j}.csv', index=False)

# %%
res_df.to_csv(f'{transformed_folder}{sep}all_transfs_info.csv', index=False)
# %%
set_qis_df = pd.DataFrame(set_qis)
set_qis_df.to_csv(f'{transformed_folder}{sep}list_key_vars.csv', index=False)

# %%
transformed_folder = '../PPT'
_, _, input_files = next(walk(f'{transformed_folder}'))


# %%

ff = []
for f in input_files:
    ff.append(f'{f.split("_")[0]}_{f.split("_")[1]}')

from collections import Counter
c = Counter(ff)
# %%
final_file = []
for key, value in c.items():
    if value>=0:
        final_file.append(key)
# %%
final_ff = []
for f in final_file:
    final_ff.append(f.split('_')[0])
# %%
len(set(final_ff))
# %%
