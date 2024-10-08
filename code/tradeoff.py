# %%
from os import walk
import os
import pandas as pd
import numpy as np
import seaborn as sns
import re
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import ticker
# %%
priv_results = pd.read_csv('../output_analysis/anonymeter.csv')
predictive_results = pd.read_csv('../output_analysis/modeling_results.csv')

# %%
priv_results['ds_complete'] = priv_results['ds_complete'].apply(lambda x: re.sub(r'_qi[0-9]','', x) if (('TVAE' in x) or ('CTGAN' in x) or ('CopulaGAN' in x) or ('dpgan' in x) or ('pategan' in x) or ('smote' in x) or ('under' in x)) else x)
priv_util = priv_results.merge(predictive_results, on=['technique', 'ds_complete', 'ds'], how='left')
priv_util = priv_util[priv_util.roc_auc_perdif.notna()]
# %% Remove ds32, 33 and 38 because they do not have borderline and smote
priv_util = priv_util[~priv_util.ds.isin(['ds32', 'ds33', 'ds38', 'ds55'])]
# %%
priv_util.loc[priv_util['technique']=='PATEGAN', 'technique'] = 'PATE-GAN'
priv_util.loc[priv_util['technique']=='PPT', 'technique'] = 'Generalisation'
#%%
PROPS = {
    'boxprops':{'facecolor':'#00BFC4', 'edgecolor':'black'},
    'medianprops':{'color':'black'},
    'whiskerprops':{'color':'black'},
    'capprops':{'color':'black'}
}
PROPS_RISK = {
    'boxprops':{'facecolor':'#F1948A', 'edgecolor':'black'},
    'medianprops':{'color':'black'},
    'whiskerprops':{'color':'black'},
    'capprops':{'color':'black'}
}
order = ['Generalisation', 'RUS', 'SMOTE', 'BorderlineSMOTE', 'Copula GAN', 'TVAE', 'CTGAN', 'DPGAN', 'PATE-GAN',
         r'$\epsilon$-2PrivateSMOTE', r'$\epsilon$-3PrivateSMOTE', r'$\epsilon$-5PrivateSMOTE']
order_eps = ['0.1', '0.5', '1.0', '5.0', '10.0']
# %%
###############################
#       MAX PERFORMANCE       #
###############################

# Find the maximum value within each group
max_values = priv_util[(priv_util['roc_auc_perdif'] == priv_util.groupby(['ds', 'technique'])['roc_auc_perdif'].transform('max'))]

# Merge to get the corresponding rows
predictive_results_max = pd.merge(priv_util, max_values, how='inner')

# %%
performance_priv = predictive_results_max.loc[predictive_results_max.groupby(['ds', 'technique'])['value'].idxmin()].reset_index(drop=True)

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
axes[0].set_xticklabels(axes[0].get_xticklabels(), rotation=70)
axes[1].set_ylabel("Privacy Risk (linkability)")
axes[1].set_xlabel("")
axes[1].set_xticklabels(axes[1].get_xticklabels(), rotation=70)
axes[0].margins(y=0.2)
#axes[0].yaxis.set_ticks(np.arange(-80,20, 10))
#axes[0].set_ylim(-70,120)
#axes[1].set_ylim(-0.02,1.02)
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
axes[0].set_xticklabels(axes[0].get_xticklabels(), rotation=70)
axes[1].set_ylabel("Percentage difference of \n predictive performance (AUC)")
axes[1].set_xlabel("")
axes[1].set_xticklabels(axes[1].get_xticklabels(), rotation=70)
axes[0].margins(y=0.2)
axes[1].set_ylim(-70,120)
axes[0].set_ylim(-0.02,1.02)
axes[0].use_sticky_edges = False
axes[1].use_sticky_edges = False
axes[0].autoscale_view(scaley=True)
axes[1].autoscale_view(scaley=True)
# plt.savefig(f'{os.path.dirname(os.getcwd())}/plots/risk_performance.pdf', bbox_inches='tight')

# %% together 
sns.set_style("darkgrid")
fig, axes = plt.subplots(2, 2, figsize=(32,20))
sns.boxplot(ax=axes[0,0], data=performance_priv,
    x='technique', y='roc_auc_perdif', order=order, **PROPS)
sns.boxplot(ax=axes[0,1], data=performance_priv,
    x='technique', y='value', order=order, **PROPS)
sns.boxplot(ax=axes[1,0], data=priv_performance,
    x='technique', y='roc_auc_perdif', order=order, **PROPS_RISK)
sns.boxplot(ax=axes[1,1], data=priv_performance,
    x='technique', y='value', order=order, **PROPS_RISK)
sns.set(font_scale=3)
axes[0,1].set_ylabel("Privacy Risk (linkability)")
axes[1,1].set_ylabel("Privacy Risk (linkability)")
axes[0,0].set_xlabel("")
axes[0,1].set_xlabel("")
axes[1,0].set_xlabel("")
axes[1,1].set_xlabel("")
axes[1,0].set_xticklabels(axes[1,0].get_xticklabels(), rotation=70)
axes[0,0].set_xticklabels("")
axes[0,0].set_ylabel("Percentage difference of \n predictive performance (AUC)")
axes[1,0].set_ylabel("Percentage difference of \n predictive performance (AUC)")
axes[1,1].set_xticklabels(axes[1,1].get_xticklabels(), rotation=70)
axes[0,1].set_xticklabels("")
axes[0,0].set_ylim(-70,100)
axes[1,0].set_ylim(-70,100)
axes[0,1].set_ylim(-0.02,1.02)
axes[1,1].set_ylim(-0.02,1.02)
plt.savefig(f'{os.path.dirname(os.getcwd())}/plots/performance_risk_together_khighest.pdf', bbox_inches='tight')

# %% together only with 2PrivateSMOTE
performance_priv_paper = performance_priv[~performance_priv['ds_complete'].str.contains(r'privateSMOTE3|privateSMOTE5')]
priv_performance_paper = priv_performance[~priv_performance['ds_complete'].str.contains(r'privateSMOTE3|privateSMOTE5')]
order_ = ['Generalisation', 'RUS', 'SMOTE', 'BorderlineSMOTE', 'Copula GAN', 'TVAE', 'CTGAN', 'DPGAN', 'PATE-GAN',
         r'$\epsilon$-PrivateSMOTE']
performance_priv_paper.loc[performance_priv_paper['technique']==r'$\epsilon$-2PrivateSMOTE', 'technique'] = r'$\epsilon$-PrivateSMOTE'
priv_performance_paper.loc[priv_performance_paper['technique']==r'$\epsilon$-2PrivateSMOTE', 'technique'] = r'$\epsilon$-PrivateSMOTE'
# %%
sns.set_style("darkgrid")
fig, axes = plt.subplots(2, 2, figsize=(30,20))
sns.boxplot(ax=axes[0,0], data=performance_priv_paper,
    x='technique', y='roc_auc_perdif', order=order_, **PROPS)
sns.boxplot(ax=axes[0,1], data=performance_priv_paper,
    x='technique', y='value', order=order_, **PROPS)
sns.boxplot(ax=axes[1,0], data=priv_performance_paper,
    x='technique', y='roc_auc_perdif', order=order_, **PROPS_RISK)
sns.boxplot(ax=axes[1,1], data=priv_performance_paper,
    x='technique', y='value', order=order_, **PROPS_RISK)
sns.set(font_scale=3)
axes[0,1].set_ylabel("Privacy Risk (linkability)")
axes[1,1].set_ylabel("Privacy Risk (linkability)")
axes[0,0].set_xlabel("")
axes[0,1].set_xlabel("")
axes[1,0].set_xlabel("")
axes[1,1].set_xlabel("")
axes[1,0].set_xticklabels(axes[1,0].get_xticklabels(), rotation=70)
axes[0,0].set_xticklabels("")
axes[0,0].set_ylabel("Percentage difference of \n predictive performance (AUC)")
axes[1,0].set_ylabel("Percentage difference of \n predictive performance (AUC)")
axes[1,1].set_xticklabels(axes[1,1].get_xticklabels(), rotation=70)
axes[0,1].set_xticklabels("")
axes[0,0].set_ylim(-70,100)
axes[1,0].set_ylim(-70,100)
axes[0,1].set_ylim(-0.02,1.02)
axes[1,1].set_ylim(-0.02,1.02)
#plt.savefig(f'{os.path.dirname(os.getcwd())}/plots/performance_risk_together.jpg', dpi=300, bbox_inches='tight')

#############################
#       Each technique      #
#############################
# %%
privsmote = priv_util.loc[priv_util.technique.str.contains('PrivateSMOTE')].reset_index(drop=True)
privsmote[r'$\epsilon$'] = np.nan
privsmote['kanon'] = np.nan
for idx, file in enumerate(privsmote.ds_complete):
    privsmote[r'$\epsilon$'][idx] = str(list(map(float, re.findall(r'\d+\.\d+', file.split('_')[1])))[0])
    if len(file.split('_')[1].split('-')[1])>12:
        privsmote['kanon'][idx] = int(list(map(float, re.findall(r'\d+', file.split('_')[1].split('-')[1])))[0])
    else:
        privsmote['kanon'][idx] = int(2)

# %%
privsmote['kanon'] = privsmote['kanon'].astype(float)
privsmote[r'$\epsilon$'] = privsmote[r'$\epsilon$'].astype(float)
# %%
sns.set(style='darkgrid')
sns.scatterplot(x="roc_auc_perdif",
            y="value",
            hue=r'$\epsilon$',
            data=privsmote,
            alpha=0.7,
            palette='Set2')
plt.ylabel("Privacy Risk (Linkability)")
plt.xlabel("Percentage difference of \npredictive performance (AUC)")
# plt.savefig(f'{os.path.dirname(os.getcwd())}/plots/privateSMOTE_tradeoff_allds.pdf', bbox_inches='tight')
# %%
sns.set_style("darkgrid")
fig, axes = plt.subplots(1, 2, figsize=(25,9))
palette=["#90CAF9", "#2196F3", "#1976D2"]
sns.boxplot(ax=axes[0], data=privsmote,       
    x=r'$\epsilon$', y='roc_auc_perdif', hue='kanon', order=order_eps, palette=palette)
sns.boxplot(ax=axes[1], data=privsmote,
    x=r'$\epsilon$', y='value', hue='kanon', order=order_eps, palette=palette)
sns.set(font_scale=2.5)
sns.light_palette("seagreen", as_cmap=True)
axes[0].set_ylabel("Percentage difference of \n predictive performance (AUC)")
axes[0].set_xlabel(r'$\epsilon$')
axes[0].set_xticklabels(axes[0].get_xticklabels(), rotation=60)
axes[1].set_ylabel("Privacy Risk (linkability)")
axes[1].set_xlabel(r'$\epsilon$')
axes[1].set_xticklabels(axes[1].get_xticklabels(), rotation=60)
axes[0].margins(y=0.2)
#axes[0].yaxis.set_ticks(np.arange(-80,20, 10))
axes[0].set_ylim(-70,100)
axes[1].set_ylim(-0.02,1.02)
axes[0].use_sticky_edges = False
axes[1].use_sticky_edges = False
axes[0].autoscale_view(scaley=True)
axes[1].autoscale_view(scaley=True)
# plt.savefig(f'{os.path.dirname(os.getcwd())}/plots/privatesmote_epsilons_allvariants.pdf', bbox_inches='tight')
# %% #predictive performance first
privsmote_max = privsmote.loc[privsmote.groupby(['ds', r'$\epsilon$', 'kanon'])['roc_auc_perdif'].idxmax()].reset_index(drop=True)
sns.set_style("darkgrid")
fig = plt.subplots(figsize=(14,9))
palette=["#81D4FA", "#29B6F6", "#0288D1"]
ax = sns.boxplot(data=privsmote_max,       
    x=r'$\epsilon$', y='roc_auc_perdif', hue='kanon', order=order_eps, palette=palette)
sns.set(font_scale=2)
ax.set_ylabel("Percentage difference of \n predictive performance (AUC)")
ax.set_xlabel(r'$\epsilon$')
ax.set_xticklabels(ax.get_xticklabels(), rotation=60)
ax.margins(y=0.2)
ax.set_ylim(-70,100)
ax.use_sticky_edges = False
sns.move_legend(ax, bbox_to_anchor=(0.5,1.15), loc='upper center', borderaxespad=0., frameon=False, ncol=3, title='k-highest-risk', labels=['2','3','5'])

# %% #privacy first 
bestpriv_privatesmote = privsmote[(privsmote['value'] == privsmote.groupby(['ds', r'$\epsilon$', 'kanon'])['value'].transform('min'))]
priv_performance_privatesmote = bestpriv_privatesmote.loc[bestpriv_privatesmote.groupby(['ds', r'$\epsilon$', 'kanon'])['roc_auc_perdif'].idxmax()].reset_index(drop=True)
sns.set_style("darkgrid")
fig = plt.subplots(figsize=(14,9))
palette=["#F1948A", "#EF5350", "#D32F2F"]
ax = sns.boxplot(data=priv_performance_privatesmote,       
    x=r'$\epsilon$', y='roc_auc_perdif', hue='kanon', order=order_eps, palette=palette)
sns.set(font_scale=2)
ax.set_ylabel("Percentage difference of \n predictive performance (AUC)")
ax.set_xlabel(r'$\epsilon$')
ax.set_xticklabels(ax.get_xticklabels(), rotation=60)
ax.margins(y=0.2)
ax.set_ylim(-70,100)
ax.use_sticky_edges = False
sns.move_legend(ax, bbox_to_anchor=(0.5,1.15), loc='upper center', borderaxespad=0., frameon=False, ncol=3, title='k-highest-risk', labels=['2','3','5'])
plt.savefig(f'{os.path.dirname(os.getcwd())}/plots/privatesmote_epsilons_khighest.jpg', dpi=300, bbox_inches='tight')

# %%
privsmote_ds16 = privsmote.loc[privsmote.ds=='ds16']
#plt.figure(figsize=(12,10))
ax = sns.lmplot(x="roc_auc_perdif",
                    y="value",
                    hue=r'$\epsilon$',
                    #hue_order=order_eps,
                    data=privsmote_ds16, height=4, aspect=1.3)
sns.set(font_scale=1.3)
ax.set(ylim=(-0.03, 1.02))
sns.set_palette("viridis")
plt.ylabel("Privacy Risk (Linkability)")
plt.xlabel("Percentage difference of \n predictive performance (AUC)")
# plt.savefig(f'{os.path.dirname(os.getcwd())}/plots/privateSMOTE_tradeoff_ds16.pdf', bbox_inches='tight')

# %%
# Create a density plot
privsmote2 = privsmote.loc[privsmote.ds_complete.str.contains('SMOTE_')].reset_index(drop=True)
color_epsilons = ['#3F51B5', '#AB47BC', '#FFA000', '#FFEB3B', '#AED581'] # '#F06292',
#plt.figure(figsize=(8,6))
axs = sns.kdeplot(x=privsmote2['roc_auc_perdif'], y=privsmote2['value'],
                  fill=False, thresh=0, levels=100, hue=privsmote2[r'$\epsilon$'],
                   #hue_order=order_eps,
                   palette=color_epsilons, alpha=0.7)
sns.set(font_scale=1.3)
axs.set(ylim=(-0.15, 1.02))
# axs.set(xlim=(-60, 85))
#sns.set_palette("viridis")
plt.xlabel('Percentage difference of \n predictive performance (AUC)')
plt.ylabel('Privacy Risk (Linkability)')
sns.move_legend(axs, bbox_to_anchor=(1.25,0.5), loc='center right', borderaxespad=0., frameon=False)
plt.savefig(f'{os.path.dirname(os.getcwd())}/plots/privateSMOTE_tradeoff_density.jpg', dpi=300, bbox_inches='tight')

# %% 3D plot
x_values = privsmote_max[r'$\epsilon$'].values
y_values = privsmote_max['kanon'].values
z_values = privsmote_max['roc_auc_perdif'].values
hue_values = privsmote_max['value'].values

# Create a 3D scatter plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

scatter = ax.scatter(x_values, y_values, z_values, c=hue_values, cmap='viridis')

# Add colorbar
colorbar = plt.colorbar(scatter, ax=ax, pad=0.1)
colorbar.set_label('Privacy Risk (linkability)', fontsize=8)
#colorbar.set_ticks(np.arange(0, 1,0.1))  # Set the color bar limits explicitly
colorbar.ax.tick_params(labelsize=6)

# Set ticks according to the values in the attributes
ax.set_xticks(np.arange(0,10, 2))
ax.set_yticks([2,3, 5])
ax.set_zticks(np.arange(-30, 100, 20))  # Assuming we want ticks at every 10 units in z range

# Set explicit axis limits
# ax.set_xlim(0, 10)
# ax.set_ylim(2, 3)
# ax.set_zlim(-25, 100)
ax.tick_params(axis='both', which='major', labelsize=8)
ax.tick_params(axis='both', which='minor', labelsize=8)

# Set labels
ax.set_xlabel(r'$\epsilon$', fontsize=8)
ax.set_ylabel('k', fontsize=8)
ax.set_zlabel('AUC', rotation=-90, fontsize=8)
plt.show()
# %%