"""_summary_
"""
# %%
from os import walk
import pandas as pd
import numpy as np
from record_linkage import threshold_record_linkage
import re
import gc
import ast

# %%
def privacy_risk_privatesmote_and_ppts(transf_file, orig_data, args, list_key_vars):
    dict_per = {'privacy_risk_50': [], 'privacy_risk_70': [], 'privacy_risk_90':[], 'privacy_risk_100': [], 'ds': []}

    int_transf_files = list(map(int, re.findall(r'\d+', transf_file.split('_')[0])))
    int_transf_qi = list(map(int, re.findall(r'\d+', transf_file.split('_')[2])))
    set_key_vars = list_key_vars.loc[list_key_vars['ds']==int_transf_files[0], 'set_key_vars'].values[0]
    print(transf_file) 
    print(int_transf_files)

    transf_data = pd.read_csv(f'{args.input_folder}/{transf_file}')
    
    key_vars = ast.literal_eval(set_key_vars)[int_transf_qi[0]]
    
    if args.type == 'smote_singleouts':
        # re-calculate single outs after the oversampling and select them
        # transf_data = aux_singleouts(key_vars, transf_data)
        # select single outs in the original data according the set of key vars
        orig_data = aux_singleouts(key_vars, orig_data)
    
    # remove suppressed variables in PPT
    try: key_vars = [k for k in key_vars if transf_data[k].values[0]!='*']
    except: pass
    
    try:
        # remove suppressed rows in PPT
        if transf_data[key_vars[0]].iloc[-1] == '*':
            transf_data = transf_data[transf_data[key_vars[0]].map(lambda x: x!='*')]
    except: pass  
    
    percentages = threshold_record_linkage(
        transf_data,
        orig_data,
        key_vars)
     
    dict_per['privacy_risk_50'].append(percentages[0])
    dict_per['privacy_risk_70'].append(percentages[1])
    dict_per['privacy_risk_90'].append(percentages[2])
    dict_per['privacy_risk_100'].append(percentages[3])
    dict_per['ds'].append(transf_file.split('.csv')[0])
    
    del percentages
    gc.collect()
    
    return dict_per


def aux_singleouts(key_vars, dt):
    k = dt.groupby(key_vars)[key_vars[0]].transform(len)
    dt['single_out'] = None
    dt['single_out'] = np.where(k == 1, 1, 0)
    dt = dt[dt['single_out']==1]
    return dt


def privacy_risk_resampling_and_gans(transf_file, orig_data, args, key_vars, i):
    dict_per = {'privacy_risk_50': [], 'privacy_risk_70': [], 'privacy_risk_90':[], 'privacy_risk_100': [], 'ds': []}
    
    transf_data = pd.read_csv(f'{args.input_folder}/{transf_file}')

    print(transf_file)

    percentages = threshold_record_linkage(
        transf_data,
        orig_data,
        key_vars)
    
    dict_per['privacy_risk_50'].append(percentages[0])
    dict_per['privacy_risk_70'].append(percentages[1])
    dict_per['privacy_risk_90'].append(percentages[2])
    dict_per['privacy_risk_100'].append(percentages[3])
    dict_per['ds'].append(transf_file.split('.csv')[0])
    dict_per['qi'] = i
    
    del percentages
    gc.collect()

    return dict_per  


# %% 
def apply_in_privatesmote_and_ppts(transf_file, args):
    original_folder = 'original'
    _, _, input_files = next(walk(f'{original_folder}'))

    list_key_vars = pd.read_csv('list_key_vars.csv')
    
    int_transf_files = list(map(int, re.findall(r'\d+', transf_file.split('_')[0])))
    orig_file = [file for file in input_files if int(file.split(".csv")[0]) == int_transf_files[0]]
    print(orig_file)
    orig_data = pd.read_csv(f'{original_folder}/{orig_file[0]}')

    # apply for the 80% of original data 
    indexes = np.load('indexes.npy', allow_pickle=True).item()
    indexes = pd.DataFrame.from_dict(indexes)
    f = list(map(int, re.findall(r'\d+', transf_file.split('_')[0])))
    index = indexes.loc[indexes['ds']==str(f[0]), 'indexes'].values[0]

    # split data 80/20
    idx = list(set(list(orig_data.index)) - set(index))
    orig_data = orig_data.iloc[idx, :].reset_index(drop=True)

    try: 
        risk = privacy_risk_privatesmote_and_ppts(transf_file, orig_data, args, list_key_vars)
        total_risk = pd.DataFrame.from_dict(risk)
        del risk
        gc.collect()
        total_risk.to_csv(f'{args.output_folder}/{transf_file.split(".csv")[0]}_per.csv', index=False) 
    except: pass


def apply_in_resampling_and_gans(transf_file, args):
    original_folder = 'original'
    _, _, input_files = next(walk(f'{original_folder}'))

    int_transf_files = list(map(int, re.findall(r'\d+', transf_file.split('_')[0])))

    list_key_vars = pd.read_csv('list_key_vars.csv')
    set_key_vars = ast.literal_eval(
        list_key_vars.loc[list_key_vars['ds']==int_transf_files[0], 'set_key_vars'].values[0])

    orig_file = [file for file in input_files if int(file.split(".csv")[0]) == int_transf_files[0]]
    print(int_transf_files)
    orig_data = pd.read_csv(f'{original_folder}/{orig_file[0]}')
    print(orig_file)
    # apply for the 80% of original data
    indexes = np.load('indexes.npy', allow_pickle=True).item()
    indexes = pd.DataFrame.from_dict(indexes)
    f = list(map(int, re.findall(r'\d+', transf_file.split('_')[0])))
    index = indexes.loc[indexes['ds']==str(f[0]), 'indexes'].values[0]

    # split data 80/20
    idx = list(set(list(orig_data.index)) - set(index))
    orig_data = orig_data.iloc[idx, :].reset_index(drop=True)

    for i in range(len(set_key_vars)):
        risk = privacy_risk_resampling_and_gans(transf_file, orig_data, args, set_key_vars[i], i)
        total_risk = pd.DataFrame.from_dict(risk)
        total_risk.to_csv(f'{args.output_folder}/{transf_file.split(".csv")[0]}_qi{i}_per.csv', index=False) 
        del risk
        gc.collect()