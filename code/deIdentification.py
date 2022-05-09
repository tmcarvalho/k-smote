"""Data de-identification
This script will de-identify the data and 
calculate the privacy risk for each transformed variant.
"""
# %%
import pandas as pd
import transformations
from os import sep, walk
from sklearn.preprocessing import LabelEncoder
from kanon import single_outs_sets
import record_linkage
import gc

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
    # data set without target variable to apply transformation techniques
    df_val = df[df.columns[:-1]].copy()

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
risk_folder = '../output/record_linkage/PPT'

_, _, input_files = next(walk(f'{input_folder}'))

for idx, file in enumerate(input_files):
    print(idx)
    # idx = int(input_files[0].split('.')[0])
    df =  pd.read_csv(f'{input_folder}/{file}')
    # get index
    idx = int(file.split('.')[0])
    data = df.apply(LabelEncoder().fit_transform)
    _, set_key_vars = single_outs_sets(data)
    for j, key_vars in enumerate(set_key_vars):
        # apply de-identification to the set of key vars    
        result = process_transformations(data, key_vars)
        # transform dict to dataframe
        res_df = pd.DataFrame(result)

        for i in range(len(res_df)):
            res_df['transformedDS'][i].to_csv(f'{transformed_folder}{sep}ds{str(idx)}_transf{str(i)}_qi{j}.csv')
            potential_matches, risk = record_linkage.apply_record_linkage(res_df['transformedDS'][i], data, key_vars)
            res_df['privacy_risk_50'] = risk[0]
            res_df['privacy_risk_75'] = risk[1]
            res_df['privacy_risk_100'] = risk[2]
            potential_matches.to_csv(f'{risk_folder}{sep}ds{str(idx)}_transf{str(i)}_rl_qi{j}.csv')
            gc.collect() 
# %%
