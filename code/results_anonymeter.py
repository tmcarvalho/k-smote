# %%
import os
from os import walk
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

# %%
def concat_each_file(folder):
    _, _, input_files = next(walk(f'{folder}'))
    concat_results = pd.DataFrame()

    for file in input_files:
        risk = pd.read_csv(f'{folder}/{file}')
        risk['ds_complete']=file
        concat_results = pd.concat([concat_results, risk])

    return concat_results

# %%
risk_ppt = concat_each_file('../output/anonymeter/PPT_ARX')
# %%
risk_resampling= concat_each_file('../output/anonymeter/re-sampling')
# %%
risk_deeplearning= concat_each_file('../output/anonymeter/deep_learning')
# %%
risk_privateSMOTE = concat_each_file('../output/anonymeter/PrivateSMOTE')
# %%
risk_privateSMOTE_force = concat_each_file('../output/anonymeter/PrivateSMOTE_force')
# %%
risk_privateSMOTE_laplace = concat_each_file('../output/anonymeter/PrivateSMOTE_laplace')
# %%
risk_privateSMOTE_force_laplace = concat_each_file('../output/anonymeter/PrivateSMOTE_force_laplace')
# %%
risk_ppt = risk_ppt.reset_index(drop=True)
risk_resampling = risk_resampling.reset_index(drop=True)
risk_deeplearning = risk_deeplearning.reset_index(drop=True)
risk_privateSMOTE = risk_privateSMOTE.reset_index(drop=True)
risk_privateSMOTE_force = risk_privateSMOTE_force.reset_index(drop=True)
risk_privateSMOTE_laplace = risk_privateSMOTE_laplace.reset_index(drop=True)
risk_privateSMOTE_force_laplace = risk_privateSMOTE_force_laplace.reset_index(drop=True)
# %%
risk_ppt['technique'] = 'PPT'
risk_resampling['technique'] = risk_resampling['ds_complete'].apply(lambda x: x.split('_')[1].title())
risk_deeplearning['technique'] = risk_deeplearning['ds_complete'].apply(lambda x: x.split('_')[1])
risk_privateSMOTE['technique'] = 'privateSMOTE'
risk_privateSMOTE_force['technique'] = 'privateSMOTE_force'
risk_privateSMOTE_laplace['technique'] = 'privateSMOTE_laplace'
risk_privateSMOTE_force_laplace['technique'] = 'privateSMOTE_force_laplace'
# %%
results = []
results = pd.concat([risk_ppt, risk_resampling, risk_deeplearning, risk_privateSMOTE, risk_privateSMOTE_force, risk_privateSMOTE_laplace, risk_privateSMOTE_force_laplace])
results = results.reset_index(drop=True)
# %%
results['dsn'] = results['ds_complete'].apply(lambda x: x.split('_')[0])
# %%
results_risk_max = results.copy()
results_risk_max = results.loc[results.groupby(['dsn', 'technique'])['value'].idxmin()].reset_index(drop=True)

# %%
results_risk_max = results_risk_max.loc[results_risk_max['technique']!='Over']
results_risk_max.loc[results_risk_max['technique']=='Under', 'technique'] = 'RUS'
results_risk_max.loc[results_risk_max['technique']=='Bordersmote', 'technique'] = 'BorderlineSMOTE'
results_risk_max.loc[results_risk_max['technique']=='Smote', 'technique'] = 'SMOTE'
results_risk_max.loc[results_risk_max['technique']=='copulaGAN', 'technique'] = 'Copula GAN'
results_risk_max.loc[results_risk_max['technique']=='privateSMOTE', 'technique'] = 'PrivateSMOTE'
results_risk_max.loc[results_risk_max['technique']=='privateSMOTE_force', 'technique'] = 'PrivateSMOTE Force'
results_risk_max.loc[results_risk_max['technique']=='privateSMOTE_laplace', 'technique'] = r'$\epsilon$-PrivateSMOTE'
results_risk_max.loc[results_risk_max['technique']=='privateSMOTE_force_laplace', 'technique'] = r'$\epsilon$-PrivateSMOTE Force'
# %%
order = ['PPT', 'RUS', 'SMOTE', 'BorderlineSMOTE', 'Copula GAN', 'TVAE', 'CTGAN', 'PrivateSMOTE', 'PrivateSMOTE Force', r'$\epsilon$-PrivateSMOTE', r'$\epsilon$-PrivateSMOTE Force']
# %%
sns.set_style("darkgrid")
plt.figure(figsize=(20,10))
ax = sns.boxplot(data=results_risk_max,
    x='technique', y='value', order=order, color='c')
# ax.set(ylim=(0, 30))
sns.set(font_scale=2.5)
ax.legend(loc='lower center', bbox_to_anchor=(0.5, 1), ncol=3, title='', borderaxespad=0., frameon=False)
#ax.set_yscale("symlog")
#ax.set_ylim(-0.2,150)
plt.xticks(rotation=45)
plt.xlabel("")
plt.ylabel("Re-identification Risk")
plt.show()
# figure = ax.get_figure()
# figure.savefig(f'{os.path.dirname(os.getcwd())}/output/plots/anonymeter_k5.pdf', bbox_inches='tight')

# %%
predictive_results = pd.read_csv('../output/test_cv_roc_auc_newprivatesmote.csv')

# %%
predictive_results_max = predictive_results.loc[predictive_results.groupby(['ds', 'technique'])['test_roc_auc_perdif'].idxmax()].reset_index(drop=True)
# %%
all_results = results.reset_index(drop=True)
# %%
all_results['flag'] = None
for i in range(len(predictive_results_max)):
    for j in range(len(all_results)):
        file = all_results['ds_complete'][j].split('.csv')[0]
        #print(file.split('_')[:6])
        if (all_results['technique'][j] == 'PPT') or ('privateSMOTE' in all_results['technique'][j]):
            file = all_results['ds_complete'][j].split('.csv')[0]
            if (predictive_results_max['ds_complete'][i].split('.csv')[0] == file) and (all_results['technique'][j] == predictive_results_max['technique'][i]):
                all_results['flag'][j] = 1
                # print(predictive_results_max['ds_complete'][i])
                # print(all_results['ds_complete'][j])
        else:
            if (predictive_results_max['ds_complete'][i].split('.csv')[0] in file[:-4]) and (all_results['technique'][j] == predictive_results_max['technique'][i]):
                all_results['flag'][j] = 1

# %%
all_results = all_results.loc[all_results['flag']==1]
# %%
results_max = all_results.loc[all_results.groupby(['dsn', 'technique'])['value'].idxmin()].reset_index(drop=True)
# %%
results_max = results_max.loc[results_max['technique']!='Over']
results_max.loc[results_max['technique']=='Under', 'technique'] = 'RUS'
results_max.loc[results_max['technique']=='Bordersmote', 'technique'] = 'BorderlineSMOTE'
results_max.loc[results_max['technique']=='Smote', 'technique'] = 'SMOTE'
results_max.loc[results_max['technique']=='copulaGAN', 'technique'] = 'Copula GAN'
results_max.loc[results_max['technique']=='privateSMOTE', 'technique'] = 'PrivateSMOTE'
results_max.loc[results_max['technique']=='privateSMOTE_force', 'technique'] = 'PrivateSMOTE Force'
results_max.loc[results_max['technique']=='privateSMOTE_laplace', 'technique'] = r'$\epsilon$-PrivateSMOTE'
results_max.loc[results_max['technique']=='privateSMOTE_force_laplace', 'technique'] = r'$\epsilon$-PrivateSMOTE Force'
# %%
order = ['PPT', 'RUS', 'SMOTE', 'BorderlineSMOTE', 'Copula GAN', 'TVAE', 'CTGAN', 'PrivateSMOTE', 'PrivateSMOTE Force', r'$\epsilon$-PrivateSMOTE', r'$\epsilon$-PrivateSMOTE Force']
# %%
plt.figure(figsize=(11,6))
ax = sns.boxplot(data=results_max, x='technique', y='value',
    palette='Spectral_r', order=order)
sns.set(font_scale=1.6)
ax.margins(y=0.02)
ax.margins(x=0.03)
ax.use_sticky_edges = False
ax.autoscale_view(scaley=True)
plt.xticks(rotation=45)
plt.xlabel("")
plt.ylabel("Re-identification Risk")
plt.show()
# figure = ax.get_figure()
# figure.savefig(f'{os.path.dirname(os.getcwd())}/output/plots/anonymeter_k5_predictiveperformance.pdf', bbox_inches='tight')

# %%
PROPS = {
    'boxprops':{'facecolor':'lightsteelblue', 'edgecolor':'steelblue'},
    'medianprops':{'color':'darkcyan'},
    'whiskerprops':{'color':'steelblue'},
    'capprops':{'color':'steelblue'}
}
privsmote = results_max.loc[results_max.technique.str.contains('PrivateSMOTE')]
sns.set_style("darkgrid")
plt.figure(figsize=(8,6))
ax = sns.boxplot(data=privsmote, x='technique', y='value',order=order, **PROPS)
# ax.set(ylim=(-60, 30))
#ax.set_yscale("log")
sns.set(font_scale=1.1)
plt.xticks(rotation=45)
plt.xlabel("")
plt.ylabel("Privacy Risk")
#plt.yscale('symlog')
plt.autoscale(True)
plt.show()
#figure = ax.get_figure()
#figure.savefig(f'{os.path.dirname(os.getcwd())}/output/plots/performance_privateSMOTE.pdf', bbox_inches='tight')

# %%
predictive_results_max = predictive_results_max.loc[predictive_results_max['technique']!='Over']
predictive_results_max.loc[predictive_results_max['technique']=='Under', 'technique'] = 'RUS'
predictive_results_max.loc[predictive_results_max['technique']=='Bordersmote', 'technique'] = 'BorderlineSMOTE'
predictive_results_max.loc[predictive_results_max['technique']=='Smote', 'technique'] = 'SMOTE'
predictive_results_max.loc[predictive_results_max['technique']=='copulaGAN', 'technique'] = 'Copula GAN'
predictive_results_max.loc[predictive_results_max['technique']=='privateSMOTE', 'technique'] = 'PrivateSMOTE'
predictive_results_max.loc[predictive_results_max['technique']=='privateSMOTE_force', 'technique'] = 'PrivateSMOTE Force'
predictive_results_max.loc[predictive_results_max['technique']=='privateSMOTE_laplace', 'technique'] = r'$\epsilon$-PrivateSMOTE'
predictive_results_max.loc[predictive_results_max['technique']=='privateSMOTE_force_laplace', 'technique'] = r'$\epsilon$-PrivateSMOTE Force'
# %%
performance_privsmote = predictive_results_max.loc[predictive_results_max.technique.str.contains('PrivateSMOTE')]
sns.set_style("darkgrid")
fig, axes = plt.subplots(1, 2, figsize=(15,8))
sns.boxplot(ax=axes[0], data=performance_privsmote,
    x='technique', y='test_roc_auc_perdif', order=order, palette='Spectral')
sns.boxplot(ax=axes[1], data=privsmote,
    x='technique', y='value', order=order, palette='Spectral')
sns.set(font_scale=1)
axes[0].set_ylabel("Percentage difference of predictive performance (AUC)")
axes[0].set_xlabel("")
axes[0].set_xticklabels(axes[0].get_xticklabels(), rotation=45)
axes[1].set_ylabel("Privacy Risk (linkability)")
axes[1].set_xlabel("")
axes[1].set_xticklabels(axes[1].get_xticklabels(), rotation=45)
sns.set(font_scale=1.4)
axes[0].margins(y=0.2)
axes[0].use_sticky_edges = False
axes[1].use_sticky_edges = False
axes[0].autoscale_view(scaley=True)
axes[1].autoscale_view(scaley=True)
# plt.savefig(f'{os.path.dirname(os.getcwd())}/output/plots/performance_risk_privateSMOTE.pdf', bbox_inches='tight')

# %%
sns.set_style("darkgrid")
fig, axes = plt.subplots(1, 2, figsize=(24,9))
sns.boxplot(ax=axes[0], data=predictive_results_max,
    x='technique', y='test_roc_auc_perdif', order=order, palette='Spectral')
sns.boxplot(ax=axes[1], data=results_max,
    x='technique', y='value', order=order, palette='Spectral')
sns.set(font_scale=1.6)
axes[0].set_ylabel("Percentage difference of predictive performance (AUC)")
axes[0].set_xlabel("")
axes[0].set_xticklabels(axes[0].get_xticklabels(), rotation=45)
axes[1].set_ylabel("Privacy Risk (linkability)")
axes[1].set_xlabel("")
axes[1].set_xticklabels(axes[1].get_xticklabels(), rotation=45)
sns.set(font_scale=1.6)
axes[0].margins(y=0.2)
axes[0].use_sticky_edges = False
axes[1].use_sticky_edges = False
axes[0].autoscale_view(scaley=True)
axes[1].autoscale_view(scaley=True)
# plt.savefig(f'{os.path.dirname(os.getcwd())}/output/plots/performance_risk.pdf', bbox_inches='tight')

# %%
