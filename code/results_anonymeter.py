# %%
from os import walk
import os
import pandas as pd
import numpy as np
import seaborn as sns
import re
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
risk_city = concat_each_file('../output/anonymeter/city')
# %%
risk_privateSMOTE = concat_each_file('../output/anonymeter/PrivateSMOTE')
# TODO: run all QIs
# %%
risk_ppt['technique'] = 'PPT'
risk_resampling['technique'] = risk_resampling['ds_complete'].apply(lambda x: x.split('_')[1].title())
risk_deeplearning['technique'] = risk_deeplearning['ds_complete'].apply(lambda x: x.split('_')[1])
risk_city['technique'] = risk_city['ds_complete'].apply(lambda x: x.split('_')[1].upper())
risk_privateSMOTE['technique'] = r'$\epsilon$-PrivateSMOTE'

# %%
results = pd.concat([risk_ppt, risk_resampling, risk_deeplearning, risk_city, risk_privateSMOTE
                    ]).reset_index(drop=True)
# %%
results['dsn'] = results['ds_complete'].apply(lambda x: x.split('_')[0])
# %%
results.loc[results['technique']=='Under', 'technique'] = 'RUS'
results.loc[results['technique']=='Bordersmote', 'technique'] = 'BorderlineSMOTE'
results.loc[results['technique']=='Smote', 'technique'] = 'SMOTE'
results.loc[results['technique']=='CopulaGAN', 'technique'] = 'Copula GAN'

# %%
results.to_csv('../output_analysis/anonymeter.csv', index=False)
# %%
order = ['PPT', 'RUS', 'SMOTE', 'BorderlineSMOTE', 'Copula GAN', 'TVAE', 'CTGAN', 'DPGAN', 'PATEGAN', r'$\epsilon$-PrivateSMOTE']
# %%
sns.set_style("darkgrid")
plt.figure(figsize=(20,10))
ax = sns.boxplot(data=results,
    x='technique', y='value', order=order, color='c')
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

# %%  BETTER IN PRIVACY
results_risk_min = results.loc[results.groupby(['dsn', 'technique'])['value'].idxmin()].reset_index(drop=True)
sns.set_style("darkgrid")
plt.figure(figsize=(20,10))
ax = sns.boxplot(data=results_risk_min,
    x='technique', y='value', order=order, color='c')
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
#####################################
#         PERFORMANCE FIRST         #
#####################################
priv_results = pd.read_csv('../output_analysis/anonymeter.csv')
predictive_results = pd.read_csv('../output_analysis/modeling_results.csv')

# %% remove ds38, ds43, ds100
priv_results = priv_results.loc[~priv_results.dsn.isin(['ds38', 'ds43', 'ds100'])]
predictive_results = predictive_results.loc[~predictive_results.ds.isin(['ds38', 'ds43', 'ds100'])]
# %%
predictive_results_max = predictive_results.loc[predictive_results.groupby(['ds', 'technique'])['test_roc_auc_perdif'].idxmax()].reset_index(drop=True)
# %% remove "qi" from privacy results file to merge the tables correctly
priv_results['ds_complete'] = priv_results['ds_complete'].apply(lambda x: re.sub(r'_qi[0-9]','', x) if (('TVAE' in x) or ('CTGAN' in x) or ('copulaGAN' in x) or ('dpart' in x) or ('synthpop' in x) or ('smote' in x) or ('under' in x)) else x)

# %%
priv_performance = pd.merge(priv_results, predictive_results_max, how='left')
# %%
priv_performance = priv_performance.dropna()

# %%
priv_performance_best = priv_performance.loc[priv_performance.groupby(['dsn', 'technique'])['value'].idxmin()].reset_index(drop=True)

# %%
order = ['PPT', 'RUS', 'SMOTE', 'BorderlineSMOTE', 'Copula GAN', 'TVAE', 'CTGAN', 'Independent', 'Synthpop', 'PrivateSMOTE','PrivateSMOTE *', 'PrivateSMOTE Force', r'$\epsilon$-PrivateSMOTE', r'$\epsilon$-PrivateSMOTE Force', r'$\epsilon$-PrivateSMOTE Force *']
# %%  PRIVACY RISK FOR ALL (BEST PERFORMANCE)
plt.figure(figsize=(11,6))
ax = sns.boxplot(data=priv_performance_best, x='technique', y='value',
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

# %% PRIVATE SMOTE VERSIONS BLUES
PROPS = {
    'boxprops':{'facecolor':'#00BFC4', 'edgecolor':'black'},
    'medianprops':{'color':'black'},
    'whiskerprops':{'color':'black'},
    'capprops':{'color':'black'}
}
# %%
privsmote = priv_performance_best.loc[priv_performance_best.technique.str.contains('PrivateSMOTE')]
order = ['PPT', 'RUS', 'SMOTE', 'BorderlineSMOTE', 'Copula GAN', 'TVAE', 'CTGAN', 'Independent', 'Synthpop', 'PrivateSMOTE','PrivateSMOTE *', 'PrivateSMOTE Force', r'$\epsilon$-PrivateSMOTE', r'$\epsilon$-PrivateSMOTE Force', r'$\epsilon$-PrivateSMOTE Force *']
sns.set_style("darkgrid")
plt.figure(figsize=(8,6))
ax = sns.boxplot(data=privsmote, x='technique', y='value',order=order, **PROPS)
sns.set(font_scale=1.1)
plt.xticks(rotation=45)
plt.xlabel("")
plt.ylabel("Privacy Risk")
plt.autoscale(True)
plt.show()
# figure = ax.get_figure()
# figure.savefig(f'{os.path.dirname(os.getcwd())}/output/plots/performance_privateSMOTE.pdf', bbox_inches='tight')

# %%  PRIVATE SMOTE VERSIONS WITH BEST PERFORMANCE
privsmote.loc[privsmote['technique']==r'$\epsilon$-PrivateSMOTE', 'technique'] = 'Laplace PrivateSMOTE'
privsmote.loc[privsmote['technique']==r'$\epsilon$-PrivateSMOTE Force', 'technique'] =  'Laplace PrivateSMOTE \nForce'
privsmote.loc[privsmote['technique']==r'$\epsilon$-PrivateSMOTE Force *', 'technique'] =  'Laplace PrivateSMOTE \nForce *'
order = ['PPT', 'RUS', 'SMOTE', 'BorderlineSMOTE', 'Copula GAN', 'TVAE', 'CTGAN', 'Independent', 'Synthpop', 'PrivateSMOTE','PrivateSMOTE *', 'PrivateSMOTE Force', 'Laplace PrivateSMOTE','Laplace PrivateSMOTE \nForce', 'Laplace PrivateSMOTE \nForce *']
sns.set_style("darkgrid")
fig, axes = plt.subplots(1, 2, figsize=(15,8))
sns.boxplot(ax=axes[0], data=privsmote,
    x='technique', y='test_roc_auc_perdif', order=order,**PROPS)
sns.boxplot(ax=axes[1], data=privsmote,
    x='technique', y='value', order=order, **PROPS)
sns.set(font_scale=1)
axes[0].set_ylabel("Percentage difference of predictive performance \n (F-score)")
axes[0].set_xlabel("")
axes[0].set_xticklabels(axes[0].get_xticklabels(), rotation=45)
axes[1].set_ylabel("Privacy Risk (linkability)")
axes[1].set_xlabel("")
axes[1].set_xticklabels(axes[1].get_xticklabels(), rotation=45)
sns.set(font_scale=1.55)
axes[0].set_ylim(-40,60)
axes[1].set_ylim(-0.02,1.02)
axes[0].margins(y=0.2)
axes[0].use_sticky_edges = False
axes[1].use_sticky_edges = False
axes[0].autoscale_view(scaley=True)
axes[1].autoscale_view(scaley=True)
# plt.savefig(f'{os.path.dirname(os.getcwd())}/output/plots/ablation_privateSMOTE_out_rocauc.pdf', bbox_inches='tight')

# %%
privsmote_ = privsmote.loc[(privsmote.technique!='PrivateSMOTE *') &\
                            (privsmote.technique!='Laplace PrivateSMOTE Force *')]

order = ['PPT', 'RUS', 'SMOTE', 'BorderlineSMOTE', 'Copula GAN', 'TVAE', 'CTGAN', 'Independent', 'Synthpop', 'PrivateSMOTE', 'PrivateSMOTE Force', 'Laplace PrivateSMOTE','Laplace PrivateSMOTE \nForce']
sns.set_style("darkgrid")
fig, axes = plt.subplots(1, 2, figsize=(15,7))
sns.boxplot(ax=axes[0], data=privsmote_,
    x='technique', y='test_roc_auc_perdif', order=order,**PROPS)
sns.boxplot(ax=axes[1], data=privsmote_,
    x='technique', y='value', order=order, **PROPS)
sns.set(font_scale=1)
axes[0].set_ylabel("Percentage difference of \n predictive performance (AUC)")
axes[0].set_xlabel("")
axes[0].set_xticklabels(axes[0].get_xticklabels(), rotation=45)
axes[1].set_ylabel("Privacy Risk (linkability)")
axes[1].set_xlabel("")
axes[1].set_xticklabels(axes[1].get_xticklabels(), rotation=45)
sns.set(font_scale=1.65)
# axes[0].yaxis.set_ticks(np.arange(-80,20, 10))
axes[0].set_ylim(-52,62)
axes[1].set_ylim(-0.02,1.02)
axes[0].margins(y=0.2)
axes[0].use_sticky_edges = False
axes[1].use_sticky_edges = False
axes[0].autoscale_view(scaley=True)
axes[1].autoscale_view(scaley=True)
# plt.savefig(f'{os.path.dirname(os.getcwd())}/output/plots/ablation_privateSMOTE_out_rocauc.pdf', bbox_inches='tight')

# %% 
# BEST PERFORMANCE WITH BEST PRIVATESMOTE VERSION 
order_performance_bestprivsmote = ['PPT', 'RUS', 'SMOTE', 'BorderlineSMOTE', 'Copula GAN', 'TVAE', 'CTGAN', 'Independent', 'Synthpop', r'$\epsilon$-PrivateSMOTE']

performance_bestprivsmote = priv_performance_best.loc[(priv_performance_best.technique!='PrivateSMOTE') &\
                                                   (priv_performance_best.technique!='PrivateSMOTE Force')\
                                                   & (priv_performance_best.technique!='PrivateSMOTE *')\
                                                   & (priv_performance_best.technique!=r'$\epsilon$-PrivateSMOTE *')\
                                                   & (priv_performance_best.technique!=r'$\epsilon$-PrivateSMOTE')]
# %%
performance_bestprivsmote.loc[performance_bestprivsmote.technique==r'$\epsilon$-PrivateSMOTE Force', 'technique'] = r'$\epsilon$-PrivateSMOTE'
sns.set_style("darkgrid")
fig, axes = plt.subplots(1, 2, figsize=(25,8.8))
sns.boxplot(ax=axes[0], data=performance_bestprivsmote,
    x='technique', y='test_roc_auc_perdif', order=order_performance_bestprivsmote, **PROPS)
sns.boxplot(ax=axes[1], data=performance_bestprivsmote,
    x='technique', y='value', order=order_performance_bestprivsmote, **PROPS)
sns.set(font_scale=2.2)
sns.light_palette("seagreen", as_cmap=True)
axes[0].set_ylabel("Percentage difference of \n predictive performance (AUC)")
axes[0].set_xlabel("")
axes[0].set_xticklabels(axes[0].get_xticklabels(), rotation=60)
axes[1].set_ylabel("Privacy Risk (linkability)")
axes[1].set_xlabel("")
axes[1].set_xticklabels(axes[1].get_xticklabels(), rotation=60)
axes[0].margins(y=0.2)
#axes[0].yaxis.set_ticks(np.arange(-80,20, 10))
axes[0].set_ylim(-52,62)
axes[1].set_ylim(-0.02,1.02)
axes[0].use_sticky_edges = False
axes[1].use_sticky_edges = False
axes[0].autoscale_view(scaley=True)
axes[1].autoscale_view(scaley=True)
# plt.savefig(f'{os.path.dirname(os.getcwd())}/output/plots/performance_risk_rocauc.pdf', bbox_inches='tight')

# %% 
#####################################
#           PRIVACY FIRST           #
#####################################
# BEST IN PRIVACY WITH BEST IN PERFORMANCE
#bestpriv_results = results.loc[results.groupby(['dsn','value'])['value'].min()]
# %%
bestpriv_results = priv_results[(priv_results['value'] == priv_results.groupby(['dsn', 'technique'])['value'].transform('min'))]

# %%
performance_priv = pd.merge(bestpriv_results, predictive_results, how='left')

# %%
pred_ = performance_priv.loc[performance_priv.groupby(['ds', 'technique'])['test_roc_auc_perdif'].idxmax()].reset_index(drop=True)

# %%
order_privsmote_bestperformance = ['PPT', 'RUS', 'SMOTE', 'BorderlineSMOTE', 'Copula GAN', 'TVAE', 'CTGAN', 'Independent','Synthpop', r'$\epsilon$-PrivateSMOTE']

privsmote_bestperformance = pred_.loc[(pred_.technique!='PrivateSMOTE') &\
                                                   (pred_.technique!='PrivateSMOTE Force')\
                                                   & (pred_.technique!='PrivateSMOTE *')\
                                                   & (pred_.technique!=r'$\epsilon$-PrivateSMOTE *')\
                                                  & (pred_.technique!=r'$\epsilon$-PrivateSMOTE')]
# %%
privsmote_bestperformance.loc[privsmote_bestperformance.technique==r'$\epsilon$-PrivateSMOTE Force', 'technique'] = r'$\epsilon$-PrivateSMOTE'
sns.set_style("darkgrid")
fig, axes = plt.subplots(1, 2, figsize=(25,8.8))
sns.boxplot(ax=axes[0], data=privsmote_bestperformance,
    x='technique', y='value', order=order_privsmote_bestperformance, **PROPS)
sns.boxplot(ax=axes[1], data=privsmote_bestperformance,
    x='technique', y='test_roc_auc_perdif', order=order_privsmote_bestperformance, **PROPS)
sns.set(font_scale=2.2)
sns.light_palette("seagreen", as_cmap=True)
axes[0].set_ylabel("Privacy Risk (linkability)")
axes[0].set_xlabel("")
axes[0].set_xticklabels(axes[0].get_xticklabels(), rotation=60)
axes[1].set_ylabel("Percentage difference of \n predictive performance (AUC)")
axes[1].set_xlabel("")
axes[1].set_xticklabels(axes[1].get_xticklabels(), rotation=60)
axes[0].margins(y=0.2)
#axes[1].yaxis.set_ticks(np.arange(-80,20, 10))
axes[1].set_ylim(-52,62)
axes[0].set_ylim(-0.02,1.02)
axes[0].use_sticky_edges = False
axes[1].use_sticky_edges = False
axes[0].autoscale_view(scaley=True)
axes[1].autoscale_view(scaley=True)
# plt.savefig(f'{os.path.dirname(os.getcwd())}/output/plots/risk_performance_rocauc.pdf', bbox_inches='tight')

# %%
