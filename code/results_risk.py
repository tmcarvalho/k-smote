# %%
from io import StringIO
import os
from os import walk
import re
import zipfile
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
import gc

# %%
def concat_each_rl(folder, technique):
    _, _, input_files = next(walk(f'{folder}'))
    concat_results = pd.DataFrame()

    for file in input_files:
        if 'per' in file:
            f = list(map(int, re.findall(r'\d+', file.split('_')[0])))[0]
            if f in [2,4,5,8,10,14,16,32]:
                if technique == 'ppt':
                    if 'csv' in file:
                        risk = pd.read_csv(f'{folder}/{file}')
                    else:
                        risk = np.load(f'{folder}/{file}', allow_pickle=True)
                        risk = pd.DataFrame(risk.tolist())
                
                    concat_results = pd.concat([concat_results, risk])

                if technique == 'smote_under_over':
                    if file != 'total_risk.csv':
                        if 'rl' not in file:
                            risk = pd.read_csv(f'{folder}/{file}')    
                            concat_results = pd.concat([concat_results, risk])

                if technique == 'smote_singleouts':
                    if file != 'total_risk.csv':
                        if 'per' in file.split('_')[5]:     
                            risk = pd.read_csv(f'{folder}/{file}')    
                            concat_results = pd.concat([concat_results, risk])
            
        gc.collect()

    return concat_results


# %%
risk_ppt = concat_each_rl('../output/record_linkage/PPT', 'ppt')
# %%
risk_smote_under_over= concat_each_rl('../output/record_linkage/smote_under_over', 'smote_under_over')
# %%
risk_smote_one = concat_each_rl('../output/record_linkage/smote_singleouts', 'smote_singleouts')
# %% 
risk_smote_two = concat_each_rl('../output/record_linkage/smote_singleouts_scratch', 'smote_singleouts')
# %%
risk_ppt = risk_ppt.reset_index(drop=True)
risk_smote_under_over = risk_smote_under_over.reset_index(drop=True)
risk_smote_one = risk_smote_one.reset_index(drop=True)
risk_smote_two = risk_smote_two.reset_index(drop=True)
# %%

results = []
risk_ppt['technique'] = 'PPT'
risk_ppt['ds_qi'] = None

risk_smote_under_over['technique'] = None
for i in range(len(risk_smote_under_over)):
    technique = risk_smote_under_over['ds'][i].split('_')[1]
    risk_smote_under_over['technique'][i] = technique.title()

risk_smote_one['technique'] = 'Synthetisation \n one class' 
risk_smote_two['technique'] = 'Synthetisation \n two classes'   

results = pd.concat([risk_ppt, risk_smote_under_over, risk_smote_one, risk_smote_two])
results = results.reset_index(drop=True)

# %%
results['dsn'] = results['ds'].apply(lambda x: x.split('_')[0])
# %%
results_max = results.groupby(['dsn', 'technique'], as_index=False)['privacy_risk_50', 'privacy_risk_70', 'privacy_risk_90', 'privacy_risk_100'].min()

# %%
results_melted = results_max.melt(id_vars=['dsn', 'technique'], value_vars=['privacy_risk_50', 'privacy_risk_70', 'privacy_risk_90', 'privacy_risk_100'])
# %%
order = ['PPT', 'Over', 'Under', 'Smote', 'Interpolation one class', 'Interpolation two classes']
# %%
g = sns.FacetGrid(results_melted, col='variable', col_wrap=1, height=4.5, aspect=1.5, margin_titles=True)
g.map(sns.boxplot, 'technique', 'value', palette='muted', order=order)
g.set(yscale='log')
g.set_axis_labels(x_var="", y_var="Re-identification Risk")
for axes in g.axes.flat:
    _ = axes.set_xticklabels(axes.get_xticklabels(), rotation=35)
titles = ['Threshold at 50%','Threshold at 70%', 'Threshold at 90%', 'Threshold at 100%']
for ax,title in zip(g.axes.flatten(),titles):
    ax.set_title(title )
sns.set(font_scale=1)
# plt.tight_layout()
# g.savefig(f'{os.path.dirname(os.getcwd())}/output/plots/8results_allthr_technique.png', bbox_inches='tight')

# %%
results_melted['variable'] = np.where(results_melted['variable']=='privacy_risk_50', 'Threshold at 50%', results_melted['variable'])
results_melted['variable'] = np.where(results_melted['variable']=='privacy_risk_70', 'Threshold at 70%', results_melted['variable'])
results_melted['variable'] = np.where(results_melted['variable']=='privacy_risk_90', 'Threshold at 90%', results_melted['variable'])
results_melted['variable'] = np.where(results_melted['variable']=='privacy_risk_100', 'Threshold at 100%', results_melted['variable'])


# %%
plt.figure(figsize=(20,10))
ax = sns.boxplot(data=results_melted.loc[results_melted['variable']!='Threshold at 100%'], x='technique', y='value', hue='variable', order=order)
# ax.set(ylim=(0, 30))
ax.legend(loc='lower center', bbox_to_anchor=(0.5, 1), ncol=3, title='')
ax.set_yscale("log")
sns.set(font_scale=2)
plt.xticks(rotation=30)
plt.xlabel("")
plt.ylabel("Re-identification Risk")
# plt.show()
figure = ax.get_figure()
figure.savefig(f'{os.path.dirname(os.getcwd())}/output/plots/althr_except100.pdf', bbox_inches='tight')

# %%
plt.figure(figsize=(15,10))
ax = sns.boxplot(data=results_melted.loc[results_melted['variable']=='Threshold at 100%'], x='technique', y='value', palette='muted', order=order)
#ax.set(ylim=(-10, 60))
#ax.set_yscale("log")
sns.set(font_scale=2)
plt.xticks(rotation=30)
plt.xlabel("")
plt.ylabel("Threshold at 100%")
plt.show()
figure = ax.get_figure()
figure.savefig(f'{os.path.dirname(os.getcwd())}/output/plots/thr_100.pdf', bbox_inches='tight')



# %% TERMINAR!!!
folder = '../output/record_linkage/smote_under_over'
_, _, input_files = next(walk(f'{folder}'))
# %%
len(input_files)
# %%
input_files = [f for f in input_files if 'total_risk' not in f]
# %%
for i, f in enumerate(input_files):
    if i >= 971:
        if 'rl' in f:
            df = pd.read_csv(f'{folder}/{input_files[i]}')
            df = df[df['Score'] >= \
            0.5*df['Score'].max()]
            df.to_csv(f'{folder}/{input_files[i]}', index=False)
            gc.collect()


# %%
folder = '../output/record_linkage/smote_under_over'
_, _, input_files = next(walk(f'{folder}'))
input_files = [f for f in input_files if 'total_risk' not in f]
print(len(input_files))
with zipfile.ZipFile(f'{folder}/potential_matches.zip', "a", zipfile.ZIP_DEFLATED) as zip_file:
    for i, f in enumerate(input_files):
        print(i)
        if 'rl' in f and i >=524:
            df = pd.read_csv(f'{folder}/{input_files[i]}')
            s = StringIO()
            df.to_csv(s, index=False) 
            zip_file.writestr(f'{input_files[i]}', s.getvalue())
            os.remove(f'{folder}/{input_files[i]}')
# %%
