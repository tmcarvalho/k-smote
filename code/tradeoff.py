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
priv_results['ds_complete'] = priv_results['ds_complete'].apply(lambda x: re.sub(r'_qi[0-9]','', x) if (('TVAE' in x) or ('CTGAN' in x) or ('copulaGAN' in x) or ('dpgan' in x) or ('pategan' in x) or ('smote' in x) or ('under' in x)) else x)

priv_util = priv_results.merge(predictive_results, on=['technique', 'ds_complete', 'ds'], how='left')
# %%
privsmote = priv_util.loc[priv_util.technique.str.contains('PrivateSMOTE')].reset_index(drop=True)
privsmote['epsilon'] = np.nan
for idx, file in enumerate(privsmote.ds_complete):
    if 'privateSMOTE' in file:
        privsmote['epsilon'][idx] = str(list(map(float, re.findall(r'\d+\.\d+', file.split('_')[1])))[0])
        
# %%
hue_order = ['0.1', '0.5', '1.0', '5.0', '10.0']
sns.set(style='darkgrid')
sns.scatterplot(x="roc_auc_perdif",
                    y="value",
                    hue='epsilon',
                    hue_order=hue_order,
                    data=privsmote)
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
###########################
#       MAX UTILITY       #
###########################
# df.groupby('group').agg({'column1': 'idxmax', 'column2': 'idxmin'})

predictive_results_max = priv_util.loc[priv_util.groupby(['ds', 'technique'])['roc_auc_perdif'].idxmax()].reset_index(drop=True)

# %% remove "qi" from privacy results file to merge the tables correctly
priv_results['ds_complete'] = priv_results['ds_complete'].apply(lambda x: re.sub(r'_qi[0-9]','', x) if (('TVAE' in x) or ('CTGAN' in x) or ('copulaGAN' in x) or ('dpart' in x) or ('synthpop' in x) or ('smote' in x) or ('under' in x)) else x)

# %%
priv_performance = pd.merge(priv_results, predictive_results_max, how='left')
# %%
priv_performance = priv_performance.dropna()

# %%
priv_performance_best = priv_performance.loc[priv_performance.groupby(['dsn', 'technique'])['value'].idxmin()].reset_index(drop=True)

# %%
order = ['PPT', 'RUS', 'SMOTE', 'BorderlineSMOTE', 'Copula GAN', 'TVAE', 'CTGAN', 'DPGAN', 'PATEGAN', r'$\epsilon$-PrivateSMOTE']

plt.figure(figsize=(11,6))
ax = sns.boxplot(data=predictive_results_max, x='technique', y='value',
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
#####################################
#         PERFORMANCE FIRST         #
#####################################
priv_results = pd.read_csv('../output_analysis/anonymeter.csv')
predictive_results = pd.read_csv('../output_analysis/modeling_results.csv')

# %% remove ds38, ds43, ds100
priv_results = priv_results.loc[~priv_results.dsn.isin(['ds38', 'ds43', 'ds100'])]
predictive_results = predictive_results.loc[~predictive_results.ds.isin(['ds38', 'ds43', 'ds100'])]
# %%
predictive_results_max = predictive_results.loc[predictive_results.groupby(['ds', 'technique'])['roc_auc_perdif'].idxmax()].reset_index(drop=True)
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
# BEST PERFORMANCE WITH BEST PRIVATESMOTE VERSION 
order_performance_bestprivsmote = ['PPT', 'RUS', 'SMOTE', 'BorderlineSMOTE', 'Copula GAN', 'TVAE', 'CTGAN', 'Independent', 'Synthpop', r'$\epsilon$-PrivateSMOTE']

sns.set_style("darkgrid")
fig, axes = plt.subplots(1, 2, figsize=(25,8.8))
sns.boxplot(ax=axes[0], data=priv_performance_best,
    x='technique', y='roc_auc_perdif', order=order_performance_bestprivsmote, **PROPS)
sns.boxplot(ax=axes[1], data=priv_performance_best,
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
pred_ = performance_priv.loc[performance_priv.groupby(['ds', 'technique'])['roc_auc_perdif'].idxmax()].reset_index(drop=True)

# %%
order_privsmote_bestperformance = ['PPT', 'RUS', 'SMOTE', 'BorderlineSMOTE', 'Copula GAN', 'TVAE', 'CTGAN', 'Independent','Synthpop', r'$\epsilon$-PrivateSMOTE']

# %%
sns.set_style("darkgrid")
fig, axes = plt.subplots(1, 2, figsize=(25,8.8))
sns.boxplot(ax=axes[0], data=pred_,
    x='technique', y='value', order=order_privsmote_bestperformance, **PROPS)
sns.boxplot(ax=axes[1], data=pred_,
    x='technique', y='roc_auc_perdif', order=order_privsmote_bestperformance, **PROPS)
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
##################### PARETO FRONT ####################
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score

# Assuming 'accuracy_values' and 'linkability_risk_values' are your lists
accuracy_values = [0.8, 0.85, 0.9, 0.92, 0.88]
linkability_risk_values = [0.2, 0.15, 0.1, 0.08, 0.12]

# Calculate F1 score for each point on the curve
f1_scores = [f1_score([1 if acc >= threshold else 0 for acc in accuracy_values], [1 if risk <= threshold else 0 for risk in linkability_risk_values]) for threshold in linkability_risk_values]

# Find the index of the maximum F1 score
optimal_index = np.argmax(f1_scores)

# Plotting the tradeoff curve with the optimal point
plt.figure(figsize=(8, 8))
plt.plot(linkability_risk_values, accuracy_values, marker='o', linestyle='-', color='b', label='Tradeoff Curve')
plt.scatter([linkability_risk_values[optimal_index]], [accuracy_values[optimal_index]], color='red', label='Optimal Point')
plt.title('Tradeoff between Accuracy and Linkability Risk')
plt.xlabel('Linkability Risk')
plt.ylabel('Accuracy')
plt.grid(True)
plt.legend()
plt.show()

# Display the optimal threshold and corresponding F1 score
optimal_threshold = linkability_risk_values[optimal_index]
optimal_f1_score = f1_scores[optimal_index]
print(f"Optimal Tradeoff Point: Linkability Risk = {optimal_threshold}, Accuracy = {accuracy_values[optimal_index]}, F1 Score = {optimal_f1_score}")

# %%