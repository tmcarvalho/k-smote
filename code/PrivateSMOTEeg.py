# %%
import re
import ast
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from collections import Counter
from matplotlib.colors import to_rgba
# %%
original = pd.read_csv('../original/2.csv')
knn1 = pd.read_csv('../output/oversampled/smote_singleouts_scratch_org/ds2_smote_QI4_knn1_per1.csv')
knn3 = pd.read_csv('../output/oversampled/smote_singleouts_scratch_org/ds2_smote_QI4_knn3_per1.csv')
knn5 = pd.read_csv('../output/oversampled/smote_singleouts_scratch_org/ds2_smote_QI4_knn5_per1.csv')
# %%
def aux_singleouts(key_vars, dt):
    k = dt.groupby(key_vars)[key_vars[0]].transform(len)
    dt['single_out'] = None
    dt['single_out'] = np.where(k == 1, 'Y', 'N')
    #dt = dt[dt['single_out']==1]
    return dt

# %% get key vars
list_key_vars = pd.read_csv('../list_key_vars.csv')
transf_file = 'ds2_smote_QI4_knn1_per1'
int_transf_files = list(map(int, re.findall(r'\d+', transf_file.split('_')[0])))
int_transf_qi = list(map(int, re.findall(r'\d+', transf_file.split('_')[2])))
set_key_vars = list_key_vars.loc[list_key_vars['ds']==int_transf_files[0], 'set_key_vars'].values[0]
key_vars = ast.literal_eval(set_key_vars)[int_transf_qi[0]]
    
# %% remove test indexes in original data
indexes = np.load('../indexes.npy', allow_pickle=True).item()
indexes = pd.DataFrame.from_dict(indexes)
f = list(map(int, re.findall(r'\d+', transf_file.split('_')[0])))
index = indexes.loc[indexes['ds']==str(f[0]), 'indexes'].values[0]

# split data 80/20
idx = list(set(list(original.index)) - set(index))
orig_data = original.iloc[idx, :].reset_index(drop=True)

# %% atualize single outs
orig_data = aux_singleouts(key_vars, orig_data)
knn1 = aux_singleouts(key_vars, knn1)
knn3 = aux_singleouts(key_vars, knn3)
knn5 = aux_singleouts(key_vars, knn5)

# %%
orig_data['type'] = 'Original'
knn1['type'] = '1-NN'
knn3['type'] = '3-NN'
knn5['type'] = '5-NN'
knn1['synt'] = np.where(knn1['single_out'] == 'Y', 'Synthetic', 'Non-single out')
knn3['synt'] = np.where(knn1['single_out'] == 'Y', 'Synthetic', 'Non-single out')
knn5['synt'] = np.where(knn1['single_out'] == 'Y', 'Synthetic', 'Non-single out')
orig_data['synt'] = np.where(orig_data['single_out'] == 'Y', 'Single out', 'Non-single out')
all_data = pd.concat([orig_data, knn1, knn3, knn5])
# %%
# plt.figure(figsize=(20,6))
sns.set(font_scale=1.5)
# sns.set_style('darkgrid', {'legend.frameon':True})
color_dict = {
    'Non-single out': to_rgba('cornflowerblue', 1),
    'Single out': to_rgba('salmon', 0.3),
    'Synthetic': to_rgba('darkseagreen', 0.45)
              }
g = sns.relplot(
    data=all_data, x="k", y="j",
    col="type", col_wrap=4, hue=orig_data.columns[-1],
    palette=color_dict,
    linewidth=0.1,
    kind="scatter"
)
g.set_titles(col_template="{col_name}")

g.set_ylabels("")
g.set_xlabels("")
sns.move_legend(g, loc='upper center', bbox_to_anchor=(0.5,1.15), title='Type', 
ncol=3, borderaxespad=0, frameon=False, markerscale=1.3)
#plt.savefig(f'../output/plots/PrivateSMOTEeg6.jpeg', bbox_inches='tight')

# %%
