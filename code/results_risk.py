# %%
from cmath import nan
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
            if f not in [0,1,3,13,23,28,34,36,40,48,54,66,87]:
                if technique == 'ppt':
                    if 'csv' in file:
                        risk = pd.read_csv(f'{folder}/{file}')
                    else:
                        risk = np.load(f'{folder}/{file}', allow_pickle=True)
                        risk = pd.DataFrame(risk.tolist())
                    risk['ds_complet']=file
                    concat_results = pd.concat([concat_results, risk])

                if technique == 'smote_under_over':
                    if file != 'total_risk.csv':
                        if 'rl' not in file:
                            risk = pd.read_csv(f'{folder}/{file}')    
                            risk['ds_complet']=file
                            concat_results = pd.concat([concat_results, risk])

                if technique == 'smote_singleouts':
                    if file != 'total_risk.csv':
                        if 'per' in file.split('_')[5]:     
                            risk = pd.read_csv(f'{folder}/{file}')    
                            risk['ds_complet']=file
                            concat_results = pd.concat([concat_results, risk])
    
        gc.collect()

    return concat_results


# %%
# risk_ppt = concat_each_rl('../output/record_linkage/PPT', 'ppt')
# %%
risk_ppt = concat_each_rl('../output/record_linkage/PPT_ARX', 'ppt')
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

# %%
risk_smote_one['technique'] = 'privateSMOTE' 
risk_smote_two['technique'] = 'privateSMOTE \n regardless of \n the class'   

# %%
results = pd.concat([risk_ppt, risk_smote_under_over, risk_smote_one, risk_smote_two])
results = results.reset_index(drop=True)

# %%
results['dsn'] = results['ds'].apply(lambda x: x.split('_')[0])

# %%
# results.to_csv('../output/rl_results.csv', index=False)
# results = pd.read_csv('../output/rl_results.csv')

# %%
predictive_results = pd.read_csv('../output/predictiveresults.csv')

# %%
predictive_results_max = predictive_results.loc[predictive_results.groupby(['ds', 'technique'])['mean_test_f1_weighted_perdif'].idxmax()].reset_index(drop=True)
# %%
results = results.reset_index(drop=True)
results['flag'] = None
for i in range(len(predictive_results_max)):
    for j in range(len(results)):
        if (predictive_results_max['ds_complete'][i].split('.csv')[0] in results['ds'][j]) and (results['technique'][j] == predictive_results_max['technique'][i]):
                results['flag'][j] = 1

# %%
results = results.loc[results['flag']==1]
# %%
results_max = results.groupby(['dsn', 'technique'], as_index=False)['privacy_risk_50', 'privacy_risk_70', 'privacy_risk_90', 'privacy_risk_100'].min()

# %%
results_melted = results_max.melt(id_vars=['dsn', 'technique'], value_vars=['privacy_risk_50', 'privacy_risk_70', 'privacy_risk_90', 'privacy_risk_100'])
# %%
order = ['PPT', 'RUS', 'SMOTE', 'privateSMOTE', 'privateSMOTE \n regardless of \n the class']
# %%
results_melted = results_melted.loc[results_melted['technique']!='Over']
results_melted.loc[results_melted['technique']=='Under', 'technique'] = 'RUS'
results_melted.loc[results_melted['technique']=='Smote', 'technique'] = 'SMOTE'

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
#plt.tight_layout()
#g.savefig(f'{os.path.dirname(os.getcwd())}/output/plots/allthr_technique.png', bbox_inches='tight')

# %%
results_melted['variable'] = np.where(results_melted['variable']=='privacy_risk_50', 'Threshold at 50%', results_melted['variable'])
results_melted['variable'] = np.where(results_melted['variable']=='privacy_risk_70', 'Threshold at 70%', results_melted['variable'])
results_melted['variable'] = np.where(results_melted['variable']=='privacy_risk_90', 'Threshold at 90%', results_melted['variable'])
results_melted['variable'] = np.where(results_melted['variable']=='privacy_risk_100', 'Threshold at 100%', results_melted['variable'])
# %%
results_melted = results_melted.loc[~results_melted.value.isnull()]
# %%
# results_melted.loc[results_melted.value==0, 'value'] = 0.01
# %%
plt.figure(figsize=(20,10))
ax = sns.boxplot(data=results_melted.loc[results_melted['variable']!='Threshold at 100%'], x='technique', y='value', hue='variable', order=order,  palette='muted')
# ax.set(ylim=(0, 30))
ax.legend(loc='lower center', bbox_to_anchor=(0.5, 1), ncol=3, title='', borderaxespad=0., frameon=False)
ax.set_yscale("log")
sns.set(font_scale=2)
plt.xticks(rotation=30)
plt.xlabel("")
plt.ylabel("Re-identification Risk")
plt.show()
#figure = ax.get_figure()
#figure.savefig(f'{os.path.dirname(os.getcwd())}/output/plots/allthr_except100_arx.pdf', bbox_inches='tight')

# %%
plt.figure(figsize=(15,10))
ax = sns.boxplot(data=results_melted.loc[results_melted['variable']=='Threshold at 100%'], x='technique', y='value', palette='Spectral_r', order=order)
ax.set_yscale("log")
ax.set(ylim=(0, 10**2))
sns.set(font_scale=2)
plt.xticks(rotation=30)
plt.xlabel("")
plt.ylabel("Threshold at 100%")
plt.show()
figure = ax.get_figure()
figure.savefig(f'{os.path.dirname(os.getcwd())}/output/plots/thr_100_arx.pdf', bbox_inches='tight')



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
#####################################################
priv = pd.read_csv('../output/rl_results.csv')
max_priv = priv.loc[priv.groupby(['dsn', 'technique'])['privacy_risk_100'].idxmin()].reset_index(drop=True)
# %%
order = ['PPT', 'RUS', 'SMOTE', 'privateSMOTE', 'privateSMOTE \n regardless of \n the class']
priv_melted = max_priv.melt(id_vars=['dsn', 'technique'], value_vars=['privacy_risk_100'])
priv_melted = priv_melted.loc[priv_melted['technique']!='Over']
priv_melted.loc[priv_melted['technique']=='Under', 'technique'] = 'RUS'
priv_melted.loc[priv_melted['technique']=='Smote', 'technique'] = 'SMOTE'

priv_melted['variable'] = np.where(priv_melted['variable']=='privacy_risk_100', 'Threshold at 100%', priv_melted['variable'])

priv_melted = priv_melted.loc[~priv_melted.value.isnull()]

#priv_melted_max = priv_melted.groupby(['dsn', 'technique'], as_index=False)['value'].min()
# %%
plt.figure(figsize=(15,10))
ax = sns.boxplot(data=priv_melted, x='technique', y='value', palette='Spectral_r', order=order)
ax.set_yscale("log")
ax.set(ylim=(0, 10**2))
sns.set(font_scale=2)
plt.xticks(rotation=30)
plt.xlabel("")
plt.ylabel("Threshold at 100%")
plt.show()
figure = ax.get_figure()
figure.savefig(f'{os.path.dirname(os.getcwd())}/output/plots/thr_100_riskfirst_arx.pdf', bbox_inches='tight')

# %%

predictive_results['flag'] = None
for i in range(len(predictive_results)):
    for j in range(len(max_priv)):
        if (predictive_results['ds_complete'][i].split('.csv')[0] in max_priv['ds_complet'][j]) and (max_priv['technique'][j] == predictive_results['technique'][i]):
                predictive_results['flag'][i] = 1
# %%
predictive_results_max = predictive_results.loc[predictive_results['flag']==1]
# %%
predictive_results_max = predictive_results_max.loc[predictive_results_max.groupby(['ds', 'technique'])['mean_test_f1_weighted_perdif'].idxmax()].reset_index(drop=True)

# %%
predictive_results_max = predictive_results_max.loc[predictive_results_max['technique']!='Over']
predictive_results_max.loc[predictive_results_max['technique']=='Under', 'technique'] = 'RUS'
predictive_results_max.loc[predictive_results_max['technique']=='Smote', 'technique'] = 'SMOTE'

# %%
sns.set_style("darkgrid")
plt.figure(figsize=(11,8))
ax = sns.boxplot(data=predictive_results_max, x='technique', y='mean_test_f1_weighted_perdif', palette='Spectral_r', order=order)
# ax.set(ylim=(0, 30))
sns.set(font_scale=1.5)
plt.xticks(rotation=30)
plt.xlabel("")
plt.ylabel("Percentage difference of predictive performance (F-score)")
plt.show()
figure = ax.get_figure()
figure.savefig(f'{os.path.dirname(os.getcwd())}/output/plots/allresults_technique_fscore_riskfirst_arx.pdf', bbox_inches='tight')

# %%
sns.set_style("darkgrid")
fig, axes = plt.subplots(1, 2, figsize=(16,6.5))
sns.boxplot(ax=axes[0], data=priv_melted, x='technique', y='value', palette='Spectral_r', order=order)
sns.boxplot(ax=axes[1], data=predictive_results_max, x='technique', y='mean_test_f1_weighted_perdif', palette='Spectral_r', order=order)
sns.set(font_scale=1.5)
axes[0].set_yscale("log")
axes[0].set_ylabel("Threshold at 100%")
axes[0].set_xlabel("")
axes[0].set_xticklabels(ax.get_xticklabels(), rotation=30)
axes[1].set_ylabel("Percentage difference of predictive performance")
axes[1].set_xlabel("")
axes[1].set_xticklabels(ax.get_xticklabels(), rotation=30)
# figure = axes.get_figure()
plt.savefig(f'{os.path.dirname(os.getcwd())}/output/plots/riskfirst_withfscore_arx.pdf', bbox_inches='tight')

# %%
