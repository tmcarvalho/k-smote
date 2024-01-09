# %%
import pandas as pd
import numpy as np
import re
import seaborn as sns
import matplotlib.pyplot as plt
import os
# %%
coverage_privsmote = pd.read_csv('../output/coverage_PrivateSMOTE.csv')
coverage_deep_learning = pd.read_csv('../output/coverage_deep_learning.csv')
coverage_city = pd.read_csv('../output/coverage_city_data.csv')

# %%
results = pd.concat([coverage_privsmote, coverage_city, coverage_deep_learning])
# %%
results['technique'] = ['PrivateSMOTE' if 'privateSMOTE' in file else file.split('_')[1] for file in results.ds]

# %%
results.loc[results['technique']=='CopulaGAN', 'technique'] = 'Copula GAN'
results.loc[results['technique']=='dpgan', 'technique'] = 'DPGAN'
results.loc[results['technique']=='pategan', 'technique'] = 'PATEGAN'
# %%
mean_cols = results.groupby(['ds', 'technique'])['Statistic similarity', 'Range Coverage', 'Category Coverage'].mean()
mean_cols = mean_cols.reset_index()
# %%
mean_cols['dsn'] = mean_cols['ds'].apply(lambda x: x.split('_')[0])
# %%
sns.set_style("darkgrid")
plt.figure(figsize=(12,10))
ax = sns.boxplot(x=mean_cols["Statistic similarity"], hue=mean_cols["technique"])
sns.move_legend(ax, bbox_to_anchor=(1,0.5), loc='center left', title='Transformation', borderaxespad=0., frameon=False)
ax.set_xlim(0,1.02)
sns.set(font_scale=1.7)
sns.set_palette("Paired")
# plt.savefig(f'{os.path.dirname(os.getcwd())}/plots/statistic_similarity.pdf', bbox_inches='tight')

# %%
plt.figure(figsize=(12,10))
ax1 = sns.boxplot(x=mean_cols["Range Coverage"], hue=mean_cols["technique"])
sns.move_legend(ax1, bbox_to_anchor=(1,0.5), loc='center left', title='Transformation', borderaxespad=0., frameon=False)
ax1.set_xlim(0,1.02)
sns.set(font_scale=1.7)
sns.set_palette("Paired")
# plt.savefig(f'{os.path.dirname(os.getcwd())}/plots/range_coverage.pdf', bbox_inches='tight')

# %%
plt.figure(figsize=(12,10))
ax2 = sns.boxplot(x=mean_cols["Category Coverage"], hue=mean_cols["technique"])
sns.move_legend(ax2, bbox_to_anchor=(1,0.5), loc='center left', title='Transformation', borderaxespad=0., frameon=False)
ax2.set_xlim(0,1.02)
sns.set(font_scale=1.7)
sns.set_palette("Paired")
# plt.savefig(f'{os.path.dirname(os.getcwd())}/plots/category_coverage.pdf', bbox_inches='tight')

# %%
privsmote = mean_cols.loc[mean_cols.technique=='PrivateSMOTE'].reset_index(drop=True)
privsmote['epsilon'] = np.nan
for idx, file in enumerate(privsmote.ds):
    if 'privateSMOTE' in file:
        privsmote['epsilon'][idx] = str(list(map(float, re.findall(r'\d+\.\d+', file.split('_')[1])))[0])
        
# %%
hue_order = ['0.1', '0.5', '1.0', '5.0', '10.0']
plt.figure(figsize=(6,4))
sns.set_style("darkgrid")
ax3 = sns.boxplot(x=privsmote["Range Coverage"], hue=privsmote['epsilon'], hue_order=hue_order)
sns.move_legend(ax3, bbox_to_anchor=(1,0.5), loc='center left', title='Epsilon', borderaxespad=0., frameon=False)
# ax3.set_xlim(0,1.02)
sns.set(font_scale=1)
sns.set_palette("Paired")
# plt.savefig(f'{os.path.dirname(os.getcwd())}/plots/range_coverage_privatesmote.pdf', bbox_inches='tight')

# %%
hue_order = ['0.1', '0.5', '1.0', '5.0', '10.0']
plt.figure(figsize=(6,4))
sns.set_style("darkgrid")
ax3 = sns.boxplot(x=privsmote["Statistic similarity"], hue=privsmote['epsilon'], hue_order=hue_order)
sns.move_legend(ax3, bbox_to_anchor=(1,0.5), loc='center left', title='Epsilon', borderaxespad=0., frameon=False)
sns.set(font_scale=1)
sns.set_palette("Paired")
# plt.savefig(f'{os.path.dirname(os.getcwd())}/plots/statistic_similarity_privatesmote.pdf', bbox_inches='tight')

# %% Particular analysis
mean_cols.loc[(mean_cols["Range Coverage"]<0.5) & (mean_cols["technique"]=='SDV')]
# %%
mean_cols.loc[(mean_cols["Range Coverage"]<0.35) & (mean_cols["technique"]=='dpgan')]
# %%
low_range_privsmote = mean_cols.loc[(mean_cols["Range Coverage"]<0.9) & (mean_cols["technique"]=='PrivateSMOTE')]
low_range_privsmote.groupby('dsn').size()
# %%
similarity_privsmote = mean_cols.loc[(mean_cols["Statistic similarity"]<0.85) & (mean_cols["technique"]=='PrivateSMOTE')]
similarity_privsmote.groupby('dsn').size()
# %% ds51_0.5-privateSMOTE_QI2_knn1_per1.csv
priv = pd.read_csv('../output/oversampled/PrivateSMOTE/ds51_0.5-privateSMOTE_QI2_knn1_per1.csv')
orig = pd.read_csv('../original/51.csv')
# %%
priv_min = priv.min()
# %%
orig_min = orig.min()
# %%
