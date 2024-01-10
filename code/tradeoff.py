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

# %%
priv_results['ds_complete'] = priv_results['ds_complete'].apply(lambda x: re.sub(r'_qi[0-9]','', x) if (('TVAE' in x) or ('CTGAN' in x) or ('copulaGAN' in x) or ('dpgan' in x) or ('pategan' in x) or ('smote' in x) or ('under' in x)) else x)

priv_util = priv_results.merge(predictive_results, on=['technique', 'ds_complete'], how='left')
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

# %
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
