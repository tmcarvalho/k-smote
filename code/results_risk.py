# %%
import os
from os import walk
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

# %%
def concat_each_rl(folder, technique):
    _, _, input_files = next(walk(f'{folder}'))
    concat_results = pd.DataFrame()

    for file in input_files:
        if technique == 'PPT':
            risk = pd.read_csv(f'{folder}/{file}')
            risk['ds_complete']=file
            concat_results = pd.concat([concat_results, risk])

        if (technique == 'resampling_gans') and (file != 'total_risk.csv') and ('rl' not in file):
            risk = pd.read_csv(f'{folder}/{file}')    
            risk['ds_complete']=file
            concat_results = pd.concat([concat_results, risk])

        if (technique == 'privateSMOTE') and (file != 'total_risk.csv') and ('per' in file.split('_')[5]):
            risk = pd.read_csv(f'{folder}/{file}')    
            risk['ds_complete']=file
            concat_results = pd.concat([concat_results, risk])

    return concat_results

# %%
risk_ppt = concat_each_rl('../output/record_linkage/PPT_ARX', 'PPT')
# %%
risk_resampling= concat_each_rl('../output/record_linkage/re-sampling', 'resampling_gans')
# %%
risk_deeplearning= concat_each_rl('../output/record_linkage/deep_learning', 'resampling_gans')

# %%
# risk_privateSMOTEA = concat_each_rl('../output/record_linkage/smote_singleouts', 'privateSMOTE')
# %% 
risk_privateSMOTEB = concat_each_rl('../output/record_linkage/smote_singleouts_scratch', 'privateSMOTE')

# %%
risk_ppt = risk_ppt.reset_index(drop=True)
risk_resampling = risk_resampling.reset_index(drop=True)
risk_deeplearning = risk_deeplearning.reset_index(drop=True)
#risk_privateSMOTEA = risk_privateSMOTEA.reset_index(drop=True)
risk_privateSMOTEB = risk_privateSMOTEB.reset_index(drop=True)
# %% find best configurations to apply in the test set
# risk_ppt_best = risk_ppt.copy()
# risk_ppt_best['qi'] = risk_ppt_best['ds'].apply(lambda x: x.split('_')[2])
# risk_ppt_best['dsn'] = risk_ppt_best['ds'].apply(lambda x: x.split('_')[0])
# risk_ppt_best = risk_ppt_best.loc[risk_ppt_best.groupby(['dsn', 'qi'])['privacy_risk_100'].idxmax()].reset_index(drop=True)
# %%
# risk_ppt_best.to_csv('../output/best_config_PPT.csv', index=False)

# %%
risk_ppt['technique'] = 'PPT'
risk_resampling['technique'] = risk_resampling['ds'].apply(lambda x: x.split('_')[1].title())
risk_deeplearning['technique'] = risk_deeplearning['ds'].apply(lambda x: x.split('_')[1])
# risk_privateSMOTEA['technique'] = 'privateSMOTE A'
risk_privateSMOTEB['technique'] = 'privateSMOTE'
# %%
results = []
results = pd.concat([risk_ppt, risk_resampling, risk_deeplearning, risk_privateSMOTEB])
results = results.reset_index(drop=True)

# %%
results['dsn'] = results['ds'].apply(lambda x: x.split('_')[0])

# %%
results.to_csv('../output/rl_results.csv', index=False)
#results = pd.read_csv('../output/rl_results.csv')

# %%
priv=results.copy()
# %%
predictive_results = pd.read_csv('../output/test_cv_roc_auc.csv')

# %%
predictive_results_max = predictive_results.loc[predictive_results.groupby(['ds', 'technique'])['test_roc_auc_perdif'].idxmax()].reset_index(drop=True)
# %%
results = results.reset_index(drop=True)
results['flag'] = None
for i in range(len(predictive_results_max)):
    for j in range(len(results)):
        file = results['ds_complete'][j].split('.csv')[0]
        if (predictive_results_max['ds_complete'][i].split('.csv')[0] in file[:-4]) and (results['technique'][j] == predictive_results_max['technique'][i]):
            results['flag'][j] = 1

# %%
results = results.loc[results['flag']==1]
# %%
results_max = results.loc[results.groupby(['dsn', 'technique'])['privacy_risk_100'].idxmin()].reset_index(drop=True)

# %%
results_melted = results_max.melt(id_vars=['dsn', 'technique'], value_vars=['privacy_risk_50', 'privacy_risk_70', 'privacy_risk_90', 'privacy_risk_100'])
# %%
results_melted = results_melted.loc[results_melted['technique']!='Over']
results_melted.loc[results_melted['technique']=='Under', 'technique'] = 'RUS'
results_melted.loc[results_melted['technique']=='Bordersmote', 'technique'] = 'BorderlineSMOTE'
results_melted.loc[results_melted['technique']=='Smote', 'technique'] = 'SMOTE'
results_melted.loc[results_melted['technique']=='copulaGAN', 'technique'] = 'Copula GAN'

# %%
order = ['PPT', 'RUS', 'SMOTE', 'BorderlineSMOTE', 'Copula GAN', 'TVAE', 'CTGAN', 'privateSMOTE']
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
#g.savefig(f'{os.path.dirname(os.getcwd())}/output/plots/allthr_technique.pdf', bbox_inches='tight')

# %%
results_melted['variable'] = np.where(results_melted['variable']=='privacy_risk_50', 'Threshold at 50%', results_melted['variable'])
results_melted['variable'] = np.where(results_melted['variable']=='privacy_risk_70', 'Threshold at 70%', results_melted['variable'])
results_melted['variable'] = np.where(results_melted['variable']=='privacy_risk_90', 'Threshold at 90%', results_melted['variable'])
results_melted['variable'] = np.where(results_melted['variable']=='privacy_risk_100', 'Threshold at 100%', results_melted['variable'])
# %%
results_melted = results_melted.loc[~results_melted.value.isnull()]

# %%
plt.figure(figsize=(20,10))
ax = sns.boxplot(data=results_melted.loc[results_melted['variable']!='Threshold at 100%'],
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
#figure = ax.get_figure()
#figure.savefig(f'{os.path.dirname(os.getcwd())}/output/plots/allthr_except100.pdf', bbox_inches='tight')

# %%
# y_values = results_melted_privateSMOTE.loc[results_melted_privateSMOTE['variable']=='Threshold at 100%']["value"].values
plt.figure(figsize=(15,11))
ax = sns.boxplot(data=results_melted.loc[results_melted['variable']=='Threshold at 100%'], x='technique', y='value',
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
plt.show()
#figure = ax.get_figure()
#figure.savefig(f'{os.path.dirname(os.getcwd())}/output/plots/performanceFirst_thr100.pdf', bbox_inches='tight')


# %%
#####################################################
priv = pd.read_csv('../output/rl_results.csv')

# %%
priv = priv.fillna(0)
# %%
max_priv = priv.loc[priv.groupby(['dsn', 'technique'])['privacy_risk_100'].idxmin()].reset_index(drop=True)
# %%
priv_melted = max_priv.melt(id_vars=['dsn', 'technique'], value_vars=['privacy_risk_100'])
priv_melted.loc[priv_melted['technique']=='Under', 'technique'] = 'RUS'
priv_melted.loc[priv_melted['technique']=='Bordersmote', 'technique'] = 'BorderlineSMOTE'
priv_melted.loc[priv_melted['technique']=='Smote', 'technique'] = 'SMOTE'
priv_melted.loc[priv_melted['technique']=='copulaGAN', 'technique'] = 'Copula GAN'

order = ['PPT', 'RUS', 'SMOTE', 'BorderlineSMOTE', 'Copula GAN', 'TVAE', 'CTGAN', 'privateSMOTE']

priv_melted['variable'] = np.where(priv_melted['variable']=='privacy_risk_100', 'Threshold at 100%', priv_melted['variable'])

priv_melted = priv_melted.loc[~priv_melted.value.isnull()]

# priv_melted_max = priv_melted.groupby(['dsn', 'technique'], as_index=False)['value'].min()
# %%
plt.figure(figsize=(12,8))
ax = sns.boxplot(data=priv_melted, x='technique', y='value', palette='Spectral_r', order=order)
ax.set_yscale("symlog")
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
predictive_results['flag'] = None
for i in range(len(predictive_results)):
    for j in range(len(max_priv)):
        if (predictive_results['ds_complete'][i].split('.csv')[0] in max_priv['ds_complete'][j]) and (max_priv['technique'][j] == predictive_results['technique'][i]):
                predictive_results['flag'][i] = 1
# %%
predictive_results_max = predictive_results.loc[predictive_results['flag']==1].reset_index(drop=True)

# %%
predictive_results_max = predictive_results_max.loc[predictive_results_max.groupby(['ds', 'technique'])['test_roc_auc_perdif'].idxmax()].reset_index(drop=True)

# %%
predictive_results_max.loc[predictive_results_max['technique']=='Under', 'technique'] = 'RUS'
predictive_results_max.loc[predictive_results_max['technique']=='Bordersmote', 'technique'] = 'BorderlineSMOTE'
predictive_results_max.loc[predictive_results_max['technique']=='Smote', 'technique'] = 'SMOTE'
predictive_results_max.loc[predictive_results_max['technique']=='copulaGAN', 'technique'] = 'Copula GAN'

# %%
sns.set_style("darkgrid")
plt.figure(figsize=(12,8))
ax = sns.boxplot(data=predictive_results_max, x='technique', y='test_roc_auc_perdif', palette='Spectral_r', order=order)
# ax.set(ylim=(0, 30))
# ax.set_yscale("symlog")
sns.set(font_scale=1.5)
plt.xticks(rotation=45)
plt.xlabel("")
plt.ylabel("Percentage difference of predictive performance (AUC)")
plt.show()
#figure = ax.get_figure()
#figure.savefig(f'{os.path.dirname(os.getcwd())}/output/plots/allresults_technique_fscore_riskfirst_arx.pdf', bbox_inches='tight')

# %%
order_priv = ['PPT', 'RUS', 'SMOTE', 'BorderlineSMOTE', 'Copula GAN', 'TVAE', 'CTGAN', 'privateSMOTE']
y_values_priv = priv_melted["value"].values
y_values_pred = predictive_results_max["test_roc_auc_perdif"].values
# %%
sns.set_style("darkgrid")
fig, axes = plt.subplots(1, 2, figsize=(24,8))
sns.boxplot(ax=axes[0], data=priv_melted,
    x='technique', y='value', palette='Spectral_r', order=order_priv)
sns.boxplot(ax=axes[1], data=predictive_results_max,
    x='technique', y='test_roc_auc_perdif', palette='Spectral_r', order=order_priv)
sns.set(font_scale=1.5)
axes[0].set_yscale("symlog")
axes[0].set_ylabel("Re-identification Risk (threshold at 100%)")
axes[0].set_xlabel("")
axes[0].set_xticklabels(axes[0].get_xticklabels(), rotation=45)
#axes[1].set_yscale("symlog")
axes[1].set_ylabel("Percentage difference of predictive performance")
axes[1].set_xlabel("")
axes[1].set_xticklabels(axes[1].get_xticklabels(), rotation=45)
axes[0].set(ylim=(-0.2,np.max(y_values_priv)+15))
# axes[1].set(ylim=(-0.2,np.max(y_values_pred)))
sns.set(font_scale=1.8)
axes[0].margins(y=0.2)
#axes[1].margins(y=0.2)
axes[0].use_sticky_edges = False
axes[1].use_sticky_edges = False
axes[0].autoscale_view(scaley=True)
axes[1].autoscale_view(scaley=True)
# plt.savefig(f'{os.path.dirname(os.getcwd())}/output/plots/riskfirst_pair.pdf', bbox_inches='tight')

# %%
