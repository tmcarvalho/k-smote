"""NOT USED!!! 
NOTE: Train with synthetic data and evaluate the prediction of target with real data.
Problem: this approach analyses how closer the synthetic data are from real data. But
it is expected that synthetic data keeps the real data characteristics! So we believe that
such an approach evaluates the utility of synthetic data instead of privacy!!
"""
# %%
from os import walk
import pandas as pd
from sdv.metrics.tabular import NumericalLR
from sklearn.preprocessing import LabelEncoder
import re
import ast

# %%
def privacy_metric_ppt(original_folder, transf_folder, orig_file, transf_file, list_key_vars):
    #dict_per = {'privacy_risk_50': [], 'privacy_risk_75': [], 'privacy_risk_100': [], 'ds': []}

    int_transf_files = list(map(int, re.findall(r'\d+', transf_file.split('_')[0])))
    int_transf_qi = list(map(int, re.findall(r'\d+', transf_file.split('_')[2])))
    set_key_vars = list_key_vars.loc[list_key_vars['ds']==int_transf_files[0], 'set_key_vars'].values[0]
    print(int_transf_files)
    
    # if (int(orig_file.split(".csv")[0]) in [2,4,5,8,10,14,16,32]) and (int(orig_file.split(".csv")[0]) in int_transf_files):
    if int(orig_file.split(".csv")[0]) in int_transf_files:
        orig_data = pd.read_csv(f'{original_folder}/{orig_file}')
        transf_data = pd.read_csv(f'{transf_folder}/{transf_file}')
        
        risk = different_qis(orig_data, transf_data, int_transf_qi, set_key_vars)
        
        #matches.to_csv(f'{output_rl_folder}/{tf.split(".csv")[0]}_rl.csv', index=False) 
        #dict_per['privacy_risk_50'].append(percentages[0])
        #dict_per['privacy_risk_75'].append(percentages[1])
        #dict_per['privacy_risk_100'].append(percentages[2])
        # gc.collect()
    
        return risk


def privacy_metric_smote_under_over(original_folder, transf_folder, orig_file, transf_file, list_key_vars):
    #dict_per = {'privacy_risk_50': [], 'privacy_risk_75': [], 'privacy_risk_100': [], 'ds': []}

    int_transf_files = list(map(int, re.findall(r'\d+', transf_file.split('_')[0])))
    print(int_transf_files)
    set_key_vars = list_key_vars.loc[list_key_vars['ds']==int_transf_files[0], 'set_key_vars'].values[0]

    # if (int(orig_file.split(".csv")[0]) in [2,4,5,8,10,14,16,32]) and (int(orig_file.split(".csv")[0]) in int_transf_files):
    if int(orig_file.split(".csv")[0]) in int_transf_files:
        orig_data = pd.read_csv(f'{original_folder}/{orig_file}')
        transf_data = pd.read_csv(f'{transf_folder}/{transf_file}')

        # apply LabelEncoder for modeling
        orig_data = orig_data.apply(LabelEncoder().fit_transform)
        transf_data = transf_data.apply(LabelEncoder().fit_transform)

        risk = NumericalLR.compute(
            orig_data,
            transf_data,
            key_fields=ast.literal_eval(set_key_vars)[0],
            sensitive_fields=[orig_data.columns[-1]]
            )

        return risk  


def privacy_metric_smote(original_folder, transf_folder, orig_file, transf_file, list_key_vars):
    int_transf_files = list(map(int, re.findall(r'\d+', transf_file.split('_')[0])))
    int_transf_qi = list(map(int, re.findall(r'\d+', transf_file.split('_')[2])))
    
    if int_transf_files[0]!=34:
        set_key_vars = list_key_vars.loc[list_key_vars['ds']==int_transf_files[0], 'set_key_vars'].values[0]
        print(int_transf_files)
        
        if int(orig_file.split(".csv")[0]) in int_transf_files:
            orig_data = pd.read_csv(f'{original_folder}/{orig_file}')
            transf_data = pd.read_csv(f'{transf_folder}/{transf_file}')
            transf_data = transf_data.loc[:, transf_data.columns[:-1]]

            risk = different_qis(orig_data, transf_data, int_transf_qi, set_key_vars)

            return risk


def different_qis(orig_data, transf_data, int_transf_qi, set_key_vars):
    # apply LabelEncoder for modeling
    orig_data = orig_data.apply(LabelEncoder().fit_transform)
    transf_data = transf_data.apply(LabelEncoder().fit_transform)

    if int_transf_qi[0] == 0:
        risk = NumericalLR.compute(
            orig_data,
            transf_data,
            key_fields=ast.literal_eval(set_key_vars)[0],
            sensitive_fields=[orig_data.columns[-1]]
            )

    if int_transf_qi[0] == 1:
        risk = NumericalLR.compute(
            orig_data,
            transf_data,
            key_fields=ast.literal_eval(set_key_vars)[1],
            sensitive_fields=[orig_data.columns[-1]]
            )
    
    if int_transf_qi[0] == 2:
        risk = NumericalLR.compute(
            orig_data,
            transf_data,
            key_fields=ast.literal_eval(set_key_vars)[2],
            sensitive_fields=[orig_data.columns[-1]]
            )
    
    if int_transf_qi[0] == 3:
        risk = NumericalLR.compute(
            orig_data,
            transf_data,
            key_fields=ast.literal_eval(set_key_vars)[3],
            sensitive_fields=[orig_data.columns[-1]]
            )
    
    if int_transf_qi[0] == 4:
        risk = NumericalLR.compute(
            orig_data,
            transf_data,
            key_fields=ast.literal_eval(set_key_vars)[4],
            sensitive_fields=[orig_data.columns[-1]]
            )
    
    return risk

# %% 
def apply_in_ppt():
    output_rl_folder = '../output/record_linkage/PPT'
    original_folder = '../original'
    transf_folder = '../PPT'
    _, _, input_files = next(walk(f'{original_folder}'))
    _, _, transf_files = next(walk(f'{transf_folder}'))

    not_considered_files = [0,1,3,13,23,28,34,36,40,48,54,66,87]
    dict_per = {'ds': [], 'privacy_risk': []}
    list_key_vars = pd.read_csv('../PPT/list_key_vars.csv')

    for idx, file in enumerate(input_files):
        print(file)
        for i, tf in enumerate(transf_files):
            print(tf)
            if (int(file.split(".csv")[0]) not in not_considered_files) and (tf != 'list_key_vars.csv'):
                risk = privacy_metric_ppt(original_folder, transf_folder, file, tf, list_key_vars)
                dict_per['privacy_risk'].append(risk)
                dict_per['ds'].append(tf.split('.csv')[0])

    total_risk = pd.DataFrame(dict_per) 
    total_risk = total_risk.dropna()

    total_risk.to_csv(f'{output_rl_folder}/total_risk.csv', index=False)


def apply_in_smote_under_over():
    output_rl_folder = '../output/record_linkage/smote_under_over'
    original_folder = '../original'
    transf_folder = '../output/oversampled/smote_under_over'
    _, _, input_files = next(walk(f'{original_folder}'))
    _, _, transf_files = next(walk(f'{transf_folder}'))
    
    list_key_vars = pd.read_csv('../PPT/list_key_vars.csv')

    not_considered_files = [0,1,3,13,23,28,34,36,40,48,54,66,87]
    dict_per = {'ds': [], 'privacy_risk': []}

    for idx, file in enumerate(input_files):
        print(file)
        for i, tf in enumerate(transf_files):
            print(tf)
            if (int(file.split(".csv")[0]) not in not_considered_files) and (tf != 'list_key_vars.csv'):
                risk = privacy_metric_smote_under_over(original_folder, transf_folder, file, tf, list_key_vars)
                dict_per['privacy_risk'].append(risk)
                dict_per['ds'].append(tf.split('.csv')[0])

    total_risk = pd.DataFrame(dict_per) 
    total_risk = total_risk.dropna()

    total_risk.to_csv(f'{output_rl_folder}/total_risk.csv', index=False)


def apply_in_smote_singleouts(transf_files, output_rl_folder):
    original_folder = '../original' 
    _, _, input_files = next(walk(f'{original_folder}'))
    list_key_vars = pd.read_csv('../PPT/list_key_vars.csv')

    not_considered_files = [0,1,3,13,23,28,34,36,40,48,54,66,87]
    dict_per = {'ds': [], 'privacy_risk': []}

    for idx, file in enumerate(input_files):
        print(file)
        for i, tf in enumerate(transf_files):
            print(tf)
            if (int(file.split(".csv")[0]) not in not_considered_files) and (tf != 'list_key_vars.csv'):
                risk = privacy_metric_smote(original_folder, transf_folder, file, tf, list_key_vars)
                dict_per['privacy_risk'].append(risk)
                dict_per['ds'].append(tf.split('.csv')[0])

    total_risk = pd.DataFrame(dict_per) 
    total_risk = total_risk.dropna()

    total_risk.to_csv(f'{output_rl_folder}/total_risk.csv', index=False)


# %%
apply_in_ppt()
# %%
apply_in_smote_under_over()
# %%
output_rl_folder = '../output/record_linkage/smote_singleouts'
transf_folder = '../output/oversampled/smote_singleouts'
_, _, transf_files = next(walk(f'{transf_folder}'))
apply_in_smote_singleouts(transf_files, output_rl_folder)

# %%
output_rl_scratch_folder = '../output/record_linkage/smote_singleouts_scratch'
transf_scratch_folder = '../output/oversampled/smote_singleouts_scratch'
_, _, transf_scratch_files = next(walk(f'{transf_folder}'))
apply_in_smote_singleouts(transf_scratch_files, output_rl_scratch_folder)
# %%
