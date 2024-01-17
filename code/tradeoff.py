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
# %% Extract 'qi' values 
priv_results['qi'] = priv_results['ds_complete'].str.extract(r'qi(\d+)', flags=re.IGNORECASE).astype(float)

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
sns.scatterplot(x="fscore_perdif",
            y="value",
            hue='epsilon',
            hue_order=hue_order,
            data=privsmote_max)
plt.ylabel("Re-identification Risk")
plt.xlabel("Percentage difference of predictive performance (AUC)")
# plt.savefig(f'{os.path.dirname(os.getcwd())}/plots/privateSMOTE_tradeoff_allds.pdf', bbox_inches='tight')

# %%
privsmote_ds16 = privsmote.loc[privsmote.ds=='ds16']
# privsmote_ds16 = privsmote_ds16.loc[privsmote_ds16.fscore_perdif>-10]

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
axes[0].set_ylim(-70,120)
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
axes[1].set_ylim(-70,120)
axes[0].set_ylim(-0.02,1.02)
axes[0].use_sticky_edges = False
axes[1].use_sticky_edges = False
axes[0].autoscale_view(scaley=True)
axes[1].autoscale_view(scaley=True)
# plt.savefig(f'{os.path.dirname(os.getcwd())}/plots/risk_performance.pdf', bbox_inches='tight')

# %% Imabalance ratios
# orig_folder = '../original'
# _, _, orig_files = next(os.walk(f'{orig_folder}'))

# indexes = np.load('../indexes.npy', allow_pickle=True).item()
# indexes = pd.DataFrame.from_dict(indexes)
# for fl in orig_files:
#     f = list(map(int, re.findall(r'\d+', fl.split('_')[0])))
#     if f[0] not in [0,1,3,13,23,28,34,36,40,48,54,66,87, 100,43]:
#         index = indexes.loc[indexes['ds']==str(f[0]), 'indexes'].values[0]
#         orig_file = [fl for fl in orig_files if list(map(int, re.findall(r'\d+', fl.split('.')[0])))[0] == f[0]]
#         print(orig_file)

#         data = pd.read_csv(f'{orig_folder}/{orig_file[0]}')
#         # split data 80/20
#         idx = list(set(list(data.index)) - set(index))
#         orig_data = data.iloc[idx, :].reset_index(drop=True)
#         y_train = orig_data[orig_data.columns[-1]]
#         # Calculate the ratio of imbalance
#         print(orig_data[orig_data.columns[-1]].value_counts())

# %%
#############################
#       TRY WITH RANKS      #
#############################
# Calculate ranks for each metric
priv_util['roc_auc_rank'] = priv_util.groupby(['ds', 'technique', 'qi'])['roc_auc_perdif'].transform(lambda x: x.rank())
priv_util['linkability_rank'] = priv_util.groupby(['ds', 'technique', 'qi'])['value'].transform(lambda x: x.rank(ascending=False))

# priv_util['roc_auc_rank'] = priv_util['roc_auc_perdif'].rank()
# priv_util['linkability_rank'] = priv_util['value'].rank(ascending=False)

# Calculate mean rank
priv_util['mean_rank'] = (priv_util['roc_auc_rank'] + priv_util['linkability_rank']) / 2

# %%
best_rank = priv_util.loc[priv_util.groupby(['ds', 'technique', 'qi'])['mean_rank'].idxmax()].reset_index(drop=True)
# %%
sns.set_style("darkgrid")
plt.figure(figsize=(9,8))
ax = sns.boxplot(data=best_rank, x='technique', y='mean_rank', order=order, **PROPS)
sns.set(font_scale=1.5)
plt.xticks(rotation=45)
plt.xlabel("")
plt.ylabel("Mean rank between ROC AUC and Linkability")
# ax.set_xlim(0,1.02)
plt.show()
# plt.savefig(f'{os.path.dirname(os.getcwd())}/plots/rank.pdf', bbox_inches='tight')
 
# %%
sns.set_style("darkgrid")
fig, axes = plt.subplots(1, 2, figsize=(25,8.8))
sns.boxplot(ax=axes[0], data=best_rank,
    x='technique', y='roc_auc_perdif', order=order, **PROPS)
sns.boxplot(ax=axes[1], data=best_rank,
    x='technique', y='value', order=order, **PROPS)
sns.set(font_scale=2.2)
sns.light_palette("seagreen", as_cmap=True)
axes[0].set_ylabel("Percentage difference of \n predictive performance (ROC AUC)")
axes[0].set_xlabel("")
axes[0].set_xticklabels(axes[0].get_xticklabels(), rotation=60)
axes[1].set_ylabel("Privacy Risk (linkability)")
axes[1].set_xlabel("")
axes[1].set_xticklabels(axes[1].get_xticklabels(), rotation=60)
axes[0].margins(y=0.2)
#axes[0].yaxis.set_ticks(np.arange(-80,20, 10))
axes[0].set_ylim(-55,90)
axes[1].set_ylim(-0.02,1.02)
axes[0].use_sticky_edges = False
axes[1].use_sticky_edges = False
axes[0].autoscale_view(scaley=True)
axes[1].autoscale_view(scaley=True)
# plt.savefig(f'{os.path.dirname(os.getcwd())}/plots/rank_individually.pdf', bbox_inches='tight')
 
# %%
