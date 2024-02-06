# %%
import pandas as pd
import numpy as np
import re
import seaborn as sns
import matplotlib.pyplot as plt
import os
# %%
coverage_privsmote = pd.read_csv('../output_analysis/coverage_PrivateSMOTE.csv')
coverage_deep_learning = pd.read_csv('../output_analysis/coverage_deep_learning.csv')
coverage_city = pd.read_csv('../output_analysis/coverage_city_data.csv')

# %%
results = pd.concat([coverage_privsmote, coverage_city, coverage_deep_learning])
# %%
results['technique'] = ['PrivateSMOTE' if 'privateSMOTE' in file else file.split('_')[1] for file in results.ds]

# %%
results.loc[results['technique']=='CopulaGAN', 'technique'] = 'Copula GAN'
results.loc[results['technique']=='dpgan', 'technique'] = 'DPGAN'
results.loc[results['technique']=='pategan', 'technique'] = 'PATE-GAN'
# %%
mean_cols = results.groupby(['ds', 'technique'])['Correlation', 'Statistic Similarity (Mean)','Statistic Similarity (Median)','Statistic Similarity (Standard Deviation)', 'Boundary Adherence', 'Range Coverage', 'Category Coverage'].mean()
mean_cols = mean_cols.reset_index()
# %%
mean_cols['dsn'] = mean_cols['ds'].apply(lambda x: x.split('_')[0])

# %%
mean_cols.loc[mean_cols['technique']=='PrivateSMOTE', 'technique'] = r'$\epsilon$-PrivateSMOTE'
# %% Remove ds32, 33 and 38 because they do not have borderline and smote
mean_cols = mean_cols[~mean_cols.dsn.isin(['ds32', 'ds33', 'ds38'])]
# %%
order = ['Copula GAN', 'TVAE', 'CTGAN', 'DPGAN', 'PATE-GAN', r'$\epsilon$-PrivateSMOTE']

sns.set_style("darkgrid")
plt.figure(figsize=(12,10))
ax = sns.boxplot(x=mean_cols["Statistic Similarity (Mean)"], hue=mean_cols["technique"], hue_order=order)
sns.move_legend(ax, bbox_to_anchor=(1,0.5), loc='center left', title='Transformation', borderaxespad=0., frameon=False)
ax.set_xlim(0,1.02)
sns.set(font_scale=2)
sns.set_palette("Paired")
# plt.savefig(f'{os.path.dirname(os.getcwd())}/plots/statistic_similarity.pdf', bbox_inches='tight')

# %%
sns.set_style("darkgrid")
plt.figure(figsize=(12,10))
ax = sns.boxplot(x=mean_cols["Statistic Similarity (Median)"], hue=mean_cols["technique"], hue_order=order)
sns.move_legend(ax, bbox_to_anchor=(1,0.5), loc='center left', title='Transformation', borderaxespad=0., frameon=False)
ax.set_xlim(0,1.02)
sns.set(font_scale=2)
sns.set_palette("Paired")
# plt.savefig(f'{os.path.dirname(os.getcwd())}/plots/statistic_similarity_median.pdf', bbox_inches='tight')

# %%
sns.set_style("darkgrid")
plt.figure(figsize=(12,10))
ax = sns.boxplot(x=mean_cols["Statistic Similarity (Standard Deviation)"], hue=mean_cols["technique"], hue_order=order)
sns.move_legend(ax, bbox_to_anchor=(1,0.5), loc='center left', title='Transformation', borderaxespad=0., frameon=False)
ax.set_xlim(0,1.02)
sns.set(font_scale=2)
sns.set_palette("Paired")
# plt.savefig(f'{os.path.dirname(os.getcwd())}/plots/statistic_similarity_std.pdf', bbox_inches='tight')

# %%
sns.set_style("darkgrid")
plt.figure(figsize=(12,10))
ax = sns.boxplot(x=mean_cols["Correlation"], hue=mean_cols["technique"], hue_order=order)
sns.move_legend(ax, bbox_to_anchor=(1,0.5), loc='center left', title='Transformation', borderaxespad=0., frameon=False)
ax.set_xlim(0,1.02)
sns.set(font_scale=2)
sns.set_palette("Paired")
# plt.savefig(f'{os.path.dirname(os.getcwd())}/plots/correlation.pdf', bbox_inches='tight')

# %%
sns.set_style("darkgrid")
plt.figure(figsize=(12,10))
ax = sns.boxplot(x=mean_cols["Boundary Adherence"], hue=mean_cols["technique"], hue_order=order)
sns.move_legend(ax, bbox_to_anchor=(1,0.5), loc='center left', title='Transformation', borderaxespad=0., frameon=False)
ax.set_xlim(0,1.02)
sns.set(font_scale=2)
sns.set_palette("Paired")
# plt.savefig(f'{os.path.dirname(os.getcwd())}/plots/boundary_adherence.pdf', bbox_inches='tight')

# %%
plt.figure(figsize=(12,10))
ax1 = sns.boxplot(x=mean_cols["Range Coverage"], hue=mean_cols["technique"], hue_order=order)
sns.move_legend(ax1, bbox_to_anchor=(1,0.5), loc='center left', title='Transformation', borderaxespad=0., frameon=False)
ax1.set_xlim(0,1.02)
sns.set(font_scale=2)
sns.set_palette("Paired")
# plt.savefig(f'{os.path.dirname(os.getcwd())}/plots/range_coverage.pdf', bbox_inches='tight')

# %%
plt.figure(figsize=(12,10))
ax2 = sns.boxplot(x=mean_cols["Category Coverage"], hue=mean_cols["technique"], hue_order=order)
sns.move_legend(ax2, bbox_to_anchor=(1,0.5), loc='center left', title='Transformation', borderaxespad=0., frameon=False)
ax2.set_xlim(0,1.02)
sns.set(font_scale=2)
sns.set_palette("Paired")
# plt.savefig(f'{os.path.dirname(os.getcwd())}/plots/category_coverage.pdf', bbox_inches='tight')
# %% Best for each DS and technique
priv_results = pd.read_csv('../output_analysis/anonymeter.csv')
predictive_results = pd.read_csv('../output_analysis/modeling_results.csv')
priv_results['ds_complete'] = priv_results['ds_complete'].apply(lambda x: re.sub(r'_qi[0-9]','', x) if (('TVAE' in x) or ('CTGAN' in x) or ('CopulaGAN' in x) or ('dpgan' in x) or ('pategan' in x) or ('smote' in x) or ('under' in x)) else x)
priv_util = priv_results.merge(predictive_results, on=['technique', 'ds_complete', 'ds'], how='left')
priv_util = priv_util[~priv_util.ds.isin(['ds32', 'ds33', 'ds38'])]
# %%
priv_util.loc[priv_util['technique']=='PATEGAN', 'technique'] = 'PATE-GAN'
# %%
cov_performance = mean_cols.merge(priv_util, left_on='ds', right_on='ds_complete', how='left')
# %%
bestpriv_results = cov_performance[(cov_performance['value'] == cov_performance.groupby(['dsn', 'technique_x'])['value'].transform('min'))]

# %%
cov_performance_best = bestpriv_results.loc[bestpriv_results.groupby(['dsn', 'technique_x'])['roc_auc_perdif'].idxmax()].reset_index(drop=True)

# %%
sns.set_style("darkgrid")
fig, axes = plt.subplots(1, 4, figsize=(26.5,8))
sns.boxplot(ax=axes[0], data=cov_performance_best,
    x='Range Coverage', hue=cov_performance_best['technique_x'], hue_order=order)
sns.boxplot(ax=axes[1], data=cov_performance_best,
    x='Boundary Adherence', hue=cov_performance_best['technique_x'], hue_order=order)
sns.boxplot(ax=axes[2], data=cov_performance_best,
    x='Statistic Similarity (Mean)', hue=cov_performance_best['technique_x'], hue_order=order)
sns.boxplot(ax=axes[3], data=cov_performance_best,
    x='Correlation', hue=cov_performance_best['technique_x'], hue_order=order)
sns.set(font_scale=2.4)
sns.set_palette("flare")
axes[0].set_xlim(0,1.02)
axes[1].set_xlim(0,1.02)
axes[2].set_xlim(0,1.02)
axes[3].set_xlim(0,1.02)
axes[0].get_legend().set_visible(False)
axes[1].get_legend().set_visible(False)
axes[3].get_legend().set_visible(False)
# plt.subplots_adjust(wspace = 0.1)
sns.move_legend(axes[2], title='Transformation Techniques', bbox_to_anchor=(-0.1,1.3), loc='upper center', borderaxespad=0., ncol=6, frameon=False)
plt.savefig(f'{os.path.dirname(os.getcwd())}/plots/utility_best_techniques.pdf', bbox_inches='tight')

# %%
privsmote = mean_cols.loc[mean_cols.technique.str.contains('PrivateSMOTE')].reset_index(drop=True)
privsmote[r'$\epsilon$'] = np.nan
for idx, file in enumerate(privsmote.ds):
    if 'privateSMOTE' in file:
        privsmote[r'$\epsilon$'][idx] = str(list(map(float, re.findall(r'\d+\.\d+', file.split('_')[1])))[0])
        
# %%
hue_order = ['0.1', '0.5', '1.0', '5.0', '10.0']
# %%
plt.figure(figsize=(6,4))
sns.set_style("darkgrid")
ax3 = sns.boxplot(x=privsmote["Boundary Adherence"], hue=privsmote[r'$\epsilon$'], hue_order=hue_order)
sns.move_legend(ax3, bbox_to_anchor=(1,0.5), loc='center left', borderaxespad=0., frameon=False)
# ax3.set_xlim(0,1.02)
sns.set(font_scale=1)
sns.set_palette("Paired")
# plt.savefig(f'{os.path.dirname(os.getcwd())}/plots/boundary_adherence_privatesmote.pdf', bbox_inches='tight')
# %%
plt.figure(figsize=(6,4))
sns.set_style("darkgrid")
ax3 = sns.boxplot(x=privsmote["Range Coverage"], hue=privsmote[r'$\epsilon$'], hue_order=hue_order)
sns.move_legend(ax3, bbox_to_anchor=(1,0.5), loc='center left', borderaxespad=0., frameon=False)
# ax3.set_xlim(0,1.02)
sns.set(font_scale=1)
sns.set_palette("Set2")
# plt.savefig(f'{os.path.dirname(os.getcwd())}/plots/range_coverage_privatesmote.pdf', bbox_inches='tight')

# %%
plt.figure(figsize=(6,4))
sns.set_style("darkgrid")
ax3 = sns.boxplot(x=privsmote["Statistic Similarity (Mean)"], hue=privsmote[r'$\epsilon$'], hue_order=hue_order)
sns.move_legend(ax3, bbox_to_anchor=(1,0.5), loc='center left', borderaxespad=0., frameon=False)
sns.set(font_scale=1)
sns.set_palette("Paired")
# plt.savefig(f'{os.path.dirname(os.getcwd())}/plots/statistic_similarity_privatesmote.pdf', bbox_inches='tight')

# %%
plt.figure(figsize=(6,4))
sns.set_style("darkgrid")
ax3 = sns.boxplot(x=privsmote["Statistic Similarity (Median)"], hue=privsmote[r'$\epsilon$'], hue_order=hue_order)
sns.move_legend(ax3, bbox_to_anchor=(1,0.5), loc='center left', borderaxespad=0., frameon=False)
sns.set(font_scale=1)
sns.set_palette("Paired")
# plt.savefig(f'{os.path.dirname(os.getcwd())}/plots/statistic_similarity_privatesmote.pdf', bbox_inches='tight')
# %%
plt.figure(figsize=(6,4))
sns.set_style("darkgrid")
ax3 = sns.boxplot(x=privsmote["Statistic Similarity (Standard Deviation)"], hue=privsmote[r'$\epsilon$'], hue_order=hue_order)
sns.move_legend(ax3, bbox_to_anchor=(1,0.5), loc='center left', borderaxespad=0., frameon=False)
sns.set(font_scale=1)
sns.set_palette("Paired")
# plt.savefig(f'{os.path.dirname(os.getcwd())}/plots/statistic_similarity_privatesmote.pdf', bbox_inches='tight')

# %%
plt.figure(figsize=(6,4))
sns.set_style("darkgrid")
ax3 = sns.boxplot(x=privsmote["Correlation"], hue=privsmote[r'$\epsilon$'], hue_order=hue_order)
sns.move_legend(ax3, bbox_to_anchor=(1,0.5), loc='center left', borderaxespad=0., frameon=False)
sns.set(font_scale=1)
sns.set_palette("Paired")
plt.savefig(f'{os.path.dirname(os.getcwd())}/plots/correlation_privatesmote.pdf', bbox_inches='tight')
# %%
sns.set_style("darkgrid")
fig, axes = plt.subplots(1, 4, figsize=(23,6))
sns.boxplot(ax=axes[0], data=privsmote,
    x='Range Coverage', hue=privsmote[r'$\epsilon$'], hue_order=hue_order)
sns.boxplot(ax=axes[1], data=privsmote,
    x='Boundary Adherence', hue=privsmote[r'$\epsilon$'], hue_order=hue_order)
sns.boxplot(ax=axes[2], data=privsmote,
    x='Statistic Similarity (Mean)', hue=privsmote[r'$\epsilon$'], hue_order=hue_order)
sns.boxplot(ax=axes[3], data=privsmote,
    x='Correlation', hue=privsmote[r'$\epsilon$'], hue_order=hue_order)
sns.set(font_scale=2.25)
sns.set_palette("viridis")
axes[0].set_xlim(0.45,1.02)
axes[1].set_xlim(0.45,1.02)
axes[2].set_xlim(0.45,1.02)
axes[3].set_xlim(0.45,1.02)
axes[0].get_legend().set_visible(False)
axes[1].get_legend().set_visible(False)
axes[3].get_legend().set_visible(False)
sns.move_legend(axes[2], bbox_to_anchor=(-0.1,1.3), loc='upper center', borderaxespad=0., ncol=6, frameon=False)
# plt.savefig(f'{os.path.dirname(os.getcwd())}/plots/utility_epsilons.pdf', bbox_inches='tight')

# %% DS16
privsmote_ds16 = privsmote.loc[privsmote.dsn=='ds16']
sns.set_style("darkgrid")
fig, axes = plt.subplots(1, 4, figsize=(23,6))
sns.boxplot(ax=axes[0], data=privsmote_ds16,
    x='Range Coverage', hue=privsmote_ds16[r'$\epsilon$'], hue_order=hue_order)
sns.boxplot(ax=axes[1], data=privsmote_ds16,
    x='Boundary Adherence', hue=privsmote_ds16[r'$\epsilon$'], hue_order=hue_order)
sns.boxplot(ax=axes[2], data=privsmote_ds16,
    x='Statistic Similarity (Mean)', hue=privsmote_ds16[r'$\epsilon$'], hue_order=hue_order)
sns.boxplot(ax=axes[3], data=privsmote_ds16,
    x='Correlation', hue=privsmote_ds16[r'$\epsilon$'], hue_order=hue_order)
sns.set(font_scale=2.25)
sns.set_palette("crest")
axes[0].set_xlim(0.8,1.02)
axes[1].set_xlim(0.8,1.02)
axes[2].set_xlim(0.8,1.02)
axes[3].set_xlim(0.8,1.02)
axes[0].get_legend().set_visible(False)
axes[1].get_legend().set_visible(False)
axes[3].get_legend().set_visible(False)
sns.move_legend(axes[2], bbox_to_anchor=(-0.1,1.3), loc='upper center', borderaxespad=0., ncol=6, frameon=False)
# plt.savefig(f'{os.path.dirname(os.getcwd())}/plots/utility_epsilons_ds16.pdf', bbox_inches='tight')

# %% Particular analysis
mean_cols.loc[(mean_cols["Range Coverage"]<0.5) & (mean_cols["technique"]=='SDV')]
# %%
mean_cols.loc[(mean_cols["Range Coverage"]<0.35) & (mean_cols["technique"]=='dpgan')]
# %%
low_range_privsmote = mean_cols.loc[(mean_cols["Range Coverage"]<0.9) & (mean_cols["technique"]=='PrivateSMOTE')]
low_range_privsmote.groupby('dsn').size()
# %%
similarity_privsmote = mean_cols.loc[(mean_cols["Statistic Similarity (Standard Deviation)"]<0.7) & (mean_cols["technique"].str.contains('PrivateSMOTE'))]
similarity_privsmote.groupby('dsn').size()
# %% ds51_0.5-privateSMOTE_QI2_knn1_per1.csv
priv = pd.read_csv('../output/oversampled/PrivateSMOTE/ds51_0.5-privateSMOTE_QI2_knn1_per1.csv')
orig = pd.read_csv('../original/51.csv')
# %%
priv_min = priv.min()
# %%
orig_min = orig.min()
# %%
