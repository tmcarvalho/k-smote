# %%
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
risk_bordersmote= concat_each_rl('../output/record_linkage/borderlineSmote', 'smote_under_over')
# %%
risk_gaussianCopula = concat_each_rl('../output/record_linkage/gaussianCopula', 'smote_under_over')
# %%
risk_tvae= concat_each_rl('../output/record_linkage/TVAE', 'smote_under_over')
# %%
risk_ctgan= concat_each_rl('../output/record_linkage/CTGAN', 'smote_under_over')
# %%
risk_copulagan= concat_each_rl('../output/record_linkage/copulaGAN', 'smote_under_over')
# %%
risk_smote_one = concat_each_rl('../output/record_linkage/smote_singleouts', 'smote_singleouts')
# %% 
risk_smote_two = concat_each_rl('../output/record_linkage/smote_singleouts_scratch', 'smote_singleouts')
# %%
# risk_bordersmote_one = concat_each_rl('../output/record_linkage/borderlineSmote_singleouts', 'smote_singleouts')
# %%
risk_ppt = risk_ppt.reset_index(drop=True)
risk_smote_under_over = risk_smote_under_over.reset_index(drop=True)
risk_bordersmote = risk_bordersmote.reset_index(drop=True)
risk_gaussianCopula = risk_gaussianCopula.reset_index(drop=True)
risk_tvae = risk_tvae.reset_index(drop=True)
risk_ctgan = risk_ctgan.reset_index(drop=True)
risk_copulagan = risk_copulagan.reset_index(drop=True)
risk_smote_one = risk_smote_one.reset_index(drop=True)
# risk_bordersmote_one = risk_bordersmote_one.reset_index(drop=True)
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
risk_bordersmote['technique'] = 'BorderlineSMOTE' 
risk_gaussianCopula['technique'] = 'Gaussian Copula'
risk_tvae['technique'] = 'TVAE'
risk_ctgan['technique'] = 'CTGAN'
risk_copulagan['technique'] = 'Copula GAN'
risk_smote_one['technique'] = 'privateSMOTE' 
risk_smote_two['technique'] = 'privateSMOTE \n regardless of \n the class'
# risk_bordersmote_one['technique'] = 'privateBorderlineSMOTE'

# %%
results = pd.concat([risk_ppt, risk_smote_under_over, risk_bordersmote, risk_gaussianCopula, risk_copulagan, risk_tvae, risk_ctgan, risk_smote_one, risk_smote_two])
results = results.reset_index(drop=True)

# %%
results['dsn'] = results['ds'].apply(lambda x: x.split('_')[0])

# %%
# results.to_csv('../output/rl_results.csv', index=False)
results = pd.read_csv('../output/rl_results.csv')

priv=results.copy()
# %%
predictive_results = pd.read_csv('../output/bayesianTest_baseline_org_auc.csv')

# %%
predictive_results_max = predictive_results.loc[predictive_results.groupby(['ds', 'technique'])['test_roc_auc_perdif'].idxmax()].reset_index(drop=True)
# %%
results = results.reset_index(drop=True)
results['flag'] = None
for i in range(len(predictive_results_max)):
    for j in range(len(results)):
        if (predictive_results_max['ds_complete'][i].split('.npy')[0] in results['ds'][j]) and (results['technique'][j] == predictive_results_max['technique'][i]):
            results['flag'][j] = 1

# %%
results = results.loc[results['flag']==1]
# %%
results_max = results.groupby(['dsn', 'technique'], as_index=False)['privacy_risk_50', 'privacy_risk_70', 'privacy_risk_90', 'privacy_risk_100'].min()

# %%
results_melted = results_max.melt(id_vars=['dsn', 'technique'], value_vars=['privacy_risk_50', 'privacy_risk_70', 'privacy_risk_90', 'privacy_risk_100'])
# %%
results_melted = results_melted.loc[results_melted['technique']!='Over']
results_melted.loc[results_melted['technique']=='Under', 'technique'] = 'RUS'
results_melted.loc[results_melted['technique']=='Smote', 'technique'] = 'SMOTE'
results_melted.loc[results_melted['technique']=='privateSMOTE', 'technique'] = 'privateSMOTE A'
results_melted.loc[results_melted['technique']=='privateSMOTE \n regardless of \n the class', 'technique'] = 'privateSMOTE B'

# %%
order = ['PPT', 'RUS', 'SMOTE', 'BorderlineSMOTE', 'Copula GAN', 'TVAE', 'CTGAN', 'privateSMOTE A', 'privateSMOTE B']
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
results_melted_privateSMOTE = results_melted.loc[results_melted.technique!='privateSMOTE A']
results_melted_privateSMOTE.loc[results_melted_privateSMOTE['technique']=='privateSMOTE B', 'technique'] = 'privateSMOTE'
order = ['PPT', 'RUS', 'SMOTE', 'BorderlineSMOTE', 'Copula GAN', 'TVAE', 'CTGAN', 'privateSMOTE']
plt.figure(figsize=(20,10))
ax = sns.boxplot(data=results_melted_privateSMOTE.loc[results_melted_privateSMOTE['variable']!='Threshold at 100%'],
    x='technique', y='value', hue='variable', order=order,  palette='muted')
# ax.set(ylim=(0, 30))
sns.set(font_scale=2.3)
ax.legend(loc='lower center', bbox_to_anchor=(0.5, 1), ncol=3, title='', borderaxespad=0., frameon=False)
ax.set_yscale("log")
ax.margins(x=0)
#ax.margins(y=0.08)
#ax.use_sticky_edges = False
#ax.autoscale_view(scalex=True)
plt.xticks(rotation=45)
plt.xlabel("")
plt.ylabel("Re-identification Risk (%)")
plt.show()
# figure = ax.get_figure()
# figure.savefig(f'{os.path.dirname(os.getcwd())}/output/plots/allthr_except100_arx_outofsample_auc.pdf', bbox_inches='tight')

# %%
# y_values = results_melted_privateSMOTE.loc[results_melted_privateSMOTE['variable']=='Threshold at 100%']["value"].values
plt.figure(figsize=(15,11))
ax = sns.boxplot(data=results_melted_privateSMOTE.loc[results_melted_privateSMOTE['variable']=='Threshold at 100%'], x='technique', y='value',
    palette='Spectral_r', order=order)
ax.set_yscale("symlog")
#ax.set(ylim=(-0.2,np.max(y_values)))
sns.set(font_scale=2)
ax.margins(y=0.02)
ax.margins(x=0.03)
ax.use_sticky_edges = False
ax.autoscale_view(scaley=True)
plt.xticks(rotation=45)
plt.xlabel("")
plt.ylabel("Re-identification Risk (threshold at 100%)")
# plt.autoscale(True)
plt.show()
# figure = ax.get_figure()
# figure.savefig(f'{os.path.dirname(os.getcwd())}/output/plots/alltechniques_performanceFirst_thr100_outofsample_auc.pdf', bbox_inches='tight')


# %%
#####################################################
priv = pd.read_csv('../output/rl_results.csv')
max_priv = priv.loc[priv.groupby(['dsn', 'technique'])['privacy_risk_100'].idxmin()].reset_index(drop=True)
# %%
priv_melted = max_priv.melt(id_vars=['dsn', 'technique'], value_vars=['privacy_risk_100'])
priv_melted = priv_melted.loc[priv_melted['technique']!='Over']
priv_melted.loc[priv_melted['technique']=='Under', 'technique'] = 'RUS'
priv_melted.loc[priv_melted['technique']=='Smote', 'technique'] = 'SMOTE'
priv_melted.loc[priv_melted['technique']=='privateSMOTE', 'technique'] = 'privateSMOTE A'
priv_melted.loc[priv_melted['technique']=='privateSMOTE \n regardless of \n the class', 'technique'] = 'privateSMOTE B'

order = ['PPT', 'RUS', 'SMOTE', 'BorderlineSMOTE', 'Copula GAN', 'TVAE', 'CTGAN', 'privateSMOTE A', 'privateSMOTE B']

priv_melted['variable'] = np.where(priv_melted['variable']=='privacy_risk_100', 'Threshold at 100%', priv_melted['variable'])

priv_melted = priv_melted.loc[~priv_melted.value.isnull()]

# priv_melted_max = priv_melted.groupby(['dsn', 'technique'], as_index=False)['value'].min()
# %%
plt.figure(figsize=(12,8))
ax = sns.boxplot(data=priv_melted, x='technique', y='value', palette='Spectral_r', order=order)
ax.set_yscale("log")
# ax.set(ylim=(0, 10**2))
sns.set(font_scale=1.5)
ax.margins(y=0.02)
ax.margins(x=0.03)
ax.use_sticky_edges = False
ax.autoscale_view(scaley=True)
plt.xticks(rotation=30)
plt.xlabel("")
plt.ylabel("Re-identification Risk (threshold at 100%)")
plt.show()
#figure = ax.get_figure()
#figure.savefig(f'{os.path.dirname(os.getcwd())}/output/plots/thr100_riskFirst.pdf', bbox_inches='tight')

# %%
order_priv = ['PPT', 'RUS', 'SMOTE', 'BorderlineSMOTE', 'Copula GAN', 'TVAE', 'CTGAN', 'privateSMOTE']
priv_melted_privateSMOTE = priv_melted.loc[priv_melted['technique']!='privateSMOTE A']
priv_melted_privateSMOTE.loc[priv_melted_privateSMOTE['technique']=='privateSMOTE B', 'technique'] = 'privateSMOTE'
plt.figure(figsize=(12,8))
ax = sns.boxplot(data=priv_melted_privateSMOTE, x='technique', y='value', palette='Spectral_r', order=order_priv)
ax.set_yscale("symlog")
# ax.set(ylim=(0, 10**2))
ax.margins(y=0.02)
ax.margins(x=0.03)
ax.use_sticky_edges = False
ax.autoscale_view(scaley=True)
sns.set(font_scale=1.5)
plt.xticks(rotation=30)
plt.xlabel("")
plt.ylabel("Re-identification Risk (threshold at 100%)")
plt.show()
# figure = ax.get_figure()
# figure.savefig(f'{os.path.dirname(os.getcwd())}/output/plots/thr100_riskFirst_auc.pdf', bbox_inches='tight')

# %%
predictive_results['flag'] = None
for i in range(len(predictive_results)):
    for j in range(len(max_priv)):
        if (predictive_results['ds_complete'][i].split('.npy')[0] in max_priv['ds_complet'][j]) and (max_priv['technique'][j] == predictive_results['technique'][i]):
                predictive_results['flag'][i] = 1
# %%
predictive_results_max = predictive_results.loc[predictive_results['flag']==1].reset_index(drop=True)
# %% find nan
predictive_results_max[predictive_results_max.test_roc_auc_perdif.isna()]
# %% drop nan
predictive_results_max = predictive_results_max[predictive_results_max['test_roc_auc_perdif'].notna()]

# %%
predictive_results_max = predictive_results_max.loc[predictive_results_max.groupby(['ds', 'technique'])['test_roc_auc_perdif'].idxmax()].reset_index(drop=True)

# %%
predictive_results_max = predictive_results_max.loc[predictive_results_max['technique']!='Over']
predictive_results_max.loc[predictive_results_max['technique']=='Under', 'technique'] = 'RUS'
predictive_results_max.loc[predictive_results_max['technique']=='Smote', 'technique'] = 'SMOTE'
predictive_results_max.loc[predictive_results_max['technique']=='privateSMOTE', 'technique'] = 'privateSMOTE A'
predictive_results_max.loc[predictive_results_max['technique']=='privateSMOTE \n regardless of \n the class', 'technique'] = 'privateSMOTE B'

# %%
sns.set_style("darkgrid")
plt.figure(figsize=(12,8))
ax = sns.boxplot(data=predictive_results_max, x='technique', y='test_roc_auc_perdif', palette='Spectral_r', order=order)
# ax.set(ylim=(0, 30))
ax.set_yscale("symlog")
sns.set(font_scale=1.5)
plt.xticks(rotation=45)
plt.xlabel("")
plt.ylabel("Percentage difference of predictive performance (F-score)")
plt.show()
#figure = ax.get_figure()
#figure.savefig(f'{os.path.dirname(os.getcwd())}/output/plots/allresults_technique_fscore_riskfirst_arx.pdf', bbox_inches='tight')

# %%
order_priv = ['PPT', 'RUS', 'SMOTE', 'BorderlineSMOTE', 'Copula GAN', 'TVAE', 'CTGAN', 'privateSMOTE']
priv_melted_privateSMOTE = priv_melted.loc[(priv_melted['technique']!='privateSMOTE A') & (priv_melted['technique']!='Gaussian Copula')]
priv_melted_privateSMOTE.loc[priv_melted_privateSMOTE['technique']=='privateSMOTE B', 'technique'] = 'privateSMOTE'
predictive_results_max_privateSMOTE = predictive_results_max.loc[predictive_results_max['technique']!='privateSMOTE A']
predictive_results_max_privateSMOTE.loc[predictive_results_max_privateSMOTE['technique']=='privateSMOTE B', 'technique'] = 'privateSMOTE'
y_values_priv = priv_melted_privateSMOTE["value"].values
y_values_pred = predictive_results_max_privateSMOTE["test_roc_auc_perdif"].values
# %%
sns.set_style("darkgrid")
fig, axes = plt.subplots(1, 2, figsize=(24,8))
sns.boxplot(ax=axes[0], data=priv_melted_privateSMOTE,
    x='technique', y='value', palette='Spectral_r', order=order_priv)
sns.boxplot(ax=axes[1], data=predictive_results_max_privateSMOTE,
    x='technique', y='test_roc_auc_perdif', palette='Spectral_r', order=order_priv)
sns.set(font_scale=1.5)
axes[0].set_yscale("symlog")
axes[0].set_ylabel("Re-identification Risk (threshold at 100%)")
axes[0].set_xlabel("")
axes[0].set_xticklabels(axes[0].get_xticklabels(), rotation=45)
axes[1].set_yscale("symlog")
axes[1].set_ylabel("Percentage difference of predictive performance")
axes[1].set_xlabel("")
axes[1].set_xticklabels(axes[1].get_xticklabels(), rotation=45)
axes[0].set(ylim=(-0.2,np.max(y_values_priv)))
# axes[1].set(ylim=(-0.2,np.max(y_values_pred)))
sns.set(font_scale=1.8)
axes[0].margins(y=0.2)
#axes[1].margins(y=0.2)
axes[0].use_sticky_edges = False
axes[1].use_sticky_edges = False
axes[0].autoscale_view(scaley=True)
axes[1].autoscale_view(scaley=True)
plt.savefig(f'{os.path.dirname(os.getcwd())}/output/plots/riskfirst_auc_pair_outofsample.pdf', bbox_inches='tight')

# %%
