# %%
from os import walk
import os
import pandas as pd
import numpy as np
import seaborn as sns
import re
from matplotlib import pyplot as plt
# %%
priv_results = pd.read_csv('../output_analysis/anonymeter.csv')
predictive_results = pd.read_csv('../output_analysis/modeling_results.csv')
# TODO: RUN ALL QIs FOR PRIVATESMOTE (QI3 AND QI4 MISSING)
# %%
priv_results['ds_complete'] = priv_results['ds_complete'].apply(lambda x: re.sub(r'_qi[0-9]','', x) if (('TVAE' in x) or ('CTGAN' in x) or ('CopulaGAN' in x) or ('dpgan' in x) or ('pategan' in x) or ('smote' in x) or ('under' in x)) else x)

priv_util = priv_results.merge(predictive_results, on=['technique', 'ds_complete', 'ds'], how='left')

# %% Remove ds32, 33 and 38 because it do not have borderline and smote
priv_util = priv_util[~priv_util.ds.isin(['ds32', 'ds33', 'ds38'])]
# %%
privsmote = priv_util.loc[priv_util.technique.str.contains('PrivateSMOTE')].reset_index(drop=True)
privsmote['epsilon'] = np.nan
for idx, file in enumerate(privsmote.ds_complete):
    if 'privateSMOTE' in file:
        privsmote['epsilon'][idx] = str(list(map(float, re.findall(r'\d+\.\d+', file.split('_')[1])))[0])
        
# %%
# privsmote = privsmote.loc[privsmote.groupby(['ds', 'epsilon'])['roc_auc_perdif'].idxmax()]
hue_order = ['0.1', '0.5', '1.0', '5.0', '10.0']
sns.set(style='darkgrid')
sns.scatterplot(x="roc_auc_perdif",
            y="value",
            hue='epsilon',
            hue_order=hue_order,
            data=privsmote)
plt.ylabel("Re-identification Risk")
plt.xlabel("Percentage difference of predictive performance (AUC)")
# plt.savefig(f'{os.path.dirname(os.getcwd())}/plots/privateSMOTE_tradeoff_allds.pdf', bbox_inches='tight')
# %%
privsmote_max = privsmote.loc[privsmote.groupby(['ds', 'epsilon'])['roc_auc_perdif'].idxmax()]
hue_order = ['0.1', '0.5', '1.0', '5.0', '10.0']
sns.set(style='darkgrid')
sns.scatterplot(x="roc_auc_perdif",
            y="value",
            hue='epsilon',
            hue_order=hue_order,
            data=privsmote_max)
plt.ylabel("Re-identification Risk")
plt.xlabel("Percentage difference of predictive performance (AUC)")
# plt.savefig(f'{os.path.dirname(os.getcwd())}/plots/privateSMOTE_tradeoff_allds.pdf', bbox_inches='tight')

# %%
privsmote_ds16 = privsmote.loc[privsmote.ds=='ds14']

ax = sns.lmplot(x="roc_auc_perdif",
                    y="value",
                    hue='epsilon',
                    hue_order=hue_order,
                    data=privsmote_ds16)
plt.ylabel("Re-identification Risk")
plt.xlabel("Percentage difference of predictive performance (AUC)")
# ax.savefig(f'{os.path.dirname(os.getcwd())}/plots/privateSMOTE_tradeoff_ds14.pdf', bbox_inches='tight')

# %%
ppt = priv_util.loc[priv_util.technique.str.contains('PPT')].reset_index(drop=True)
sns.scatterplot(x="roc_auc_perdif",
                    y="value",
                    data=ppt)
# %%
resampling = priv_util.loc[(priv_util.ds_complete.str.contains('_smote')) |
                           (priv_util.ds_complete.str.contains('_border')) |
                           (priv_util.ds_complete.str.contains('_under'))
                           ].reset_index(drop=True)
sns.scatterplot(x="roc_auc_perdif",
                    y="value",
                    hue='technique',
                    data=resampling)
# there are more vertical lines because we have different re-identification risk for the same roc auc (QIs only in re-identification)
# %%
deep_learning = priv_util.loc[(priv_util.ds_complete.str.contains('_TVAE')) |
                           (priv_util.ds_complete.str.contains('_CTGAN')) |
                           (priv_util.ds_complete.str.contains('_Copula'))
                           ].reset_index(drop=True)
sns.scatterplot(x="roc_auc_perdif",
                    y="value",
                    hue='technique',
                    data=deep_learning)
# %%
city = priv_util.loc[(priv_util.ds_complete.str.contains('_dpgan')) |
                           (priv_util.ds_complete.str.contains('_pategan'))
                           ].reset_index(drop=True)
sns.scatterplot(x="roc_auc_perdif",
                    y="value",
                    hue='technique',
                    data=city)
# %%
###############################
#       MAX PERFORMANCE       #
###############################
# df.groupby('group').agg({'column1': 'idxmax', 'column2': 'idxmin'})

# Find the maximum value within each group
max_values = priv_util[(priv_util['roc_auc_perdif'] == priv_util.groupby(['ds', 'technique'])['roc_auc_perdif'].transform('max'))]

# Merge to get the corresponding rows
predictive_results_max = pd.merge(priv_util, max_values, how='inner')

# %%
performance_priv = predictive_results_max.loc[predictive_results_max.groupby(['ds', 'technique'])['value'].idxmin()].reset_index(drop=True)

# %% 
order = ['PPT', 'RUS', 'SMOTE', 'BorderlineSMOTE', 'Copula GAN', 'TVAE', 'CTGAN', 'DPGAN', 'PATEGAN', r'$\epsilon$-PrivateSMOTE']

PROPS = {
    'boxprops':{'facecolor':'#00BFC4', 'edgecolor':'black'},
    'medianprops':{'color':'black'},
    'whiskerprops':{'color':'black'},
    'capprops':{'color':'black'}
}

# %% 
# BEST PERFORMANCE WITH BEST PRIVACY
sns.set_style("darkgrid")
fig, axes = plt.subplots(1, 2, figsize=(25,8.8))
sns.boxplot(ax=axes[0], data=performance_priv,
    x='technique', y='roc_auc_perdif', order=order, **PROPS)
sns.boxplot(ax=axes[1], data=performance_priv,
    x='technique', y='value', order=order, **PROPS)
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
# plt.savefig(f'{os.path.dirname(os.getcwd())}/plots/performance_risk.pdf', bbox_inches='tight')

# %% 
#####################################
#           PRIVACY FIRST           #
#####################################
# BEST IN PRIVACY WITH BEST IN PERFORMANCE
#bestpriv_results = results.loc[results.groupby(['dsn','value'])['value'].min()]
# %%
bestpriv_results = priv_util[(priv_util['value'] == priv_util.groupby(['ds', 'technique'])['value'].transform('min'))]

# %%
priv_performance = bestpriv_results.loc[bestpriv_results.groupby(['ds', 'technique'])['roc_auc_perdif'].idxmax()].reset_index(drop=True)

# %%
sns.set_style("darkgrid")
fig, axes = plt.subplots(1, 2, figsize=(25,8.8))
sns.boxplot(ax=axes[0], data=priv_performance,
    x='technique', y='value', order=order, **PROPS)
sns.boxplot(ax=axes[1], data=priv_performance,
    x='technique', y='roc_auc_perdif', order=order, **PROPS)
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
# plt.savefig(f'{os.path.dirname(os.getcwd())}/plots/risk_performance.pdf', bbox_inches='tight')

# %%
