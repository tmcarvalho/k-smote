# %%
import pandas as pd
import re
import seaborn as sns
# %%
results = pd.read_csv('../output/coverage.csv')

# %%
mean_cols = results.groupby('ds')['Statistic similarity', 'Range Coverage', 'Category Coverage'].mean()
# %%
sns.boxplot(x=mean_cols["Statistic similarity"])
# %%
sns.boxplot(x=mean_cols["Range Coverage"])
# %%
sns.boxplot(x=mean_cols["Category Coverage"])
# %%
mean_cols = mean_cols.reset_index()

# %%
for idx, file in enumerate(mean_cols.ds):
    if 'privateSMOTE' in file:
        mean_cols['epsilon'][idx] = str(list(map(float, re.findall(r'\d+\.\d+', file.split('_')[1])))[0])
        
# %%
hue_order = ['0.1', '0.5', '1.0', '5.0', '10.0']
sns.set_style("darkgrid")
ax = sns.boxplot(x=mean_cols["Range Coverage"], hue=mean_cols['epsilon'], hue_order=hue_order)
sns.move_legend(ax, bbox_to_anchor=(1,0.5), loc='center left', title='Epsilon', borderaxespad=0., frameon=False)

# %% Particular analysis
mean_cols.loc[mean_cols["Range Coverage"]<0.7]
# %% ds51_0.5-privateSMOTE_QI2_knn1_per1.csv
priv = pd.read_csv('../output/oversampled/PrivateSMOTE/ds51_0.5-privateSMOTE_QI2_knn1_per1.csv')
orig = pd.read_csv('../original/51.csv')
# %%
priv_min = priv.min()
# %%
orig_min = orig.min()
# %%
