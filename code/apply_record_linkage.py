"""_summary_
"""
# %%
from os import walk
import pandas as pd
from record_linkage import threshold_record_linkage
from kanon import single_outs_sets
from sdv.metrics.tabular import NumericalLR
from sklearn.preprocessing import LabelEncoder
import re
import gc
import ast
# %%
original_folder = '../original'
output_rl_folder = '../output/record_linkage/PPT'
transf_folder = '../PPT'
_, _, input_files = next(walk(f'{original_folder}'))
_, _, transf_files = next(walk(f'{transf_folder}'))


not_considered_files = [0,1,3,13,23,28,32,36,40,48,54,66,87]
#dict_per = {'privacy_risk_50': [], 'privacy_risk_75': [], 'privacy_risk_100': [], 'ds': []}
dict_per = {'ds': [], 'privacy_risk': []}

list_key_vars = pd.read_csv('../PPT/list_key_vars.csv')

for idx,file in enumerate(input_files):
    for tf in transf_files:
        if int(file.split(".csv")[0]) not in not_considered_files:
            int_transf_files = list(map(int, re.findall(r'\d+', tf.split('_')[0])))
            int_transf_qi = list(map(int, re.findall(r'\d+', tf.split('_')[2])))
            set_key_vars = list_key_vars.loc[list_key_vars['ds']==int_transf_files[0], 'set_key_vars'].values[0]

            # if (int(file.split(".csv")[0]) in [2,4,5,8,10,14,16,32]) and (int(file.split(".csv")[0]) in int_transf_files):
            if int(file.split(".csv")[0]) in int_transf_files:
                orig_data = pd.read_csv(f'{original_folder}/{file}')
                # apply LabelEncoder to interpolated data (smote)
                # orig_data = orig_data.apply(LabelEncoder().fit_transform)
                transf_data = pd.read_csv(f'{transf_folder}/{tf}')
                # TODO: check index col in all cases before running

                if int_transf_qi[0] == 0:
                    # matches, percentages = threshold_record_linkage(
                    #     transf_data,
                    #     orig_data,
                    #     ast.literal_eval(set_key_vars)[0])
                    risk = NumericalLR.compute(
                        orig_data,
                        transf_data,
                        key_fields=ast.literal_eval(set_key_vars)[0],
                        sensitive_fields=orig_data.columns[-1]
                        )
                if int_transf_qi[0] == 1:
                    # matches, percentages = threshold_record_linkage(
                    #     transf_data,
                    #     orig_data,
                    #     ast.literal_eval(set_key_vars)[1])
                    risk = NumericalLR.compute(
                        orig_data,
                        transf_data,
                        key_fields=ast.literal_eval(set_key_vars)[1],
                        sensitive_fields=orig_data.columns[-1]
                        )
                
                if int_transf_qi[0] == 2:
                    # matches, percentages = threshold_record_linkage(
                    #     transf_data,
                    #     orig_data,
                    #     ast.literal_eval(set_key_vars)[2])
                    risk = NumericalLR.compute(
                        orig_data,
                        transf_data,
                        key_fields=ast.literal_eval(set_key_vars)[2],
                        sensitive_fields=orig_data.columns[-1]
                        )
                
                if int_transf_qi[0] == 3:
                    # matches, percentages = threshold_record_linkage(
                    #     transf_data,
                    #     orig_data,
                    #     ast.literal_eval(set_key_vars)[3])
                    risk = NumericalLR.compute(
                        orig_data,
                        transf_data,
                        key_fields=ast.literal_eval(set_key_vars)[3],
                        sensitive_fields=orig_data.columns[-1]
                        )
                
                if int_transf_qi[0] == 4:
                    # matches, percentages = threshold_record_linkage(
                    #     transf_data,
                    #     orig_data,
                    #     ast.literal_eval(set_key_vars)[4])
                    risk = NumericalLR.compute(
                        orig_data,
                        transf_data,
                        key_fields=ast.literal_eval(set_key_vars)[0],
                        sensitive_fields=orig_data.columns[-4]
                        )

                #matches.to_csv(f'{output_rl_folder}/{tf.split(".csv")[0]}_rl.csv', index=False) 
                #dict_per['privacy_risk_50'].append(percentages[0])
                #dict_per['privacy_risk_75'].append(percentages[1])
                #dict_per['privacy_risk_100'].append(percentages[2])
                dict_per['privacy_risk'].append(risk)
                dict_per['ds'].append(tf.split('.csv')[0])
                gc.collect()

risk = pd.DataFrame(dict_per, index=[0])  
risk.to_csv(f'{output_rl_folder}/total_risk.csv', index=False)

# %% 
