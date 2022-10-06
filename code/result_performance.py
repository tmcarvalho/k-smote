"""Predictive performance analysis
This script will analyse the predictive performance in the out-of-sample.
"""
# %%
import os
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

# %% 'all_results' are processed in the bayesTest file
all_results = pd.read_csv('../output/bayesianTest_baseline_org_auc.csv')
# %%
results_max = all_results.groupby(['ds', 'technique'], as_index=False)['test_roc_auc_perdif'].max()

# %%
results_max = results_max.loc[results_max['technique']!='Over']
results_max.loc[results_max['technique']=='Under', 'technique'] = 'RUS'
results_max.loc[results_max['technique']=='Smote', 'technique'] = 'SMOTE'
results_max.loc[results_max['technique']=='privateSMOTE', 'technique'] = 'privateSMOTE A'
results_max.loc[results_max['technique']=='privateSMOTE \n regardless of \n the class', 'technique'] = 'privateSMOTE B'

# %%
order = ['PPT', 'RUS', 'SMOTE', 'BorderlineSMOTE', 'Copula GAN', 'TVAE', 'CTGAN', 'privateSMOTE A', 'privateSMOTE B']
# %%
sns.set_style("darkgrid")
plt.figure(figsize=(12,8))
ax = sns.boxplot(data=results_max, x='technique', y='test_roc_auc_perdif', palette='Spectral_r', order=order)
# ax.set(ylim=(-60, 30))
#ax.set_yscale("log")
sns.set(font_scale=1.5)
plt.xticks(rotation=45)
plt.xlabel("")
plt.ylabel("Percentage difference of predictive performance (F-score)")
plt.yscale('symlog')
plt.autoscale(True)
plt.show()
# figure = ax.get_figure()
# figure.savefig(f'{os.path.dirname(os.getcwd())}/output/plots/alltechniques_outofsample.pdf', bbox_inches='tight')


# %%
results_max_privateSMOTE = results_max.loc[results_max.technique!='privateSMOTE A']
results_max_privateSMOTE.loc[results_max_privateSMOTE['technique']=='privateSMOTE B', 'technique'] = 'privateSMOTE'
order = ['PPT', 'RUS', 'SMOTE', 'BorderlineSMOTE', 'Copula GAN', 'TVAE', 'CTGAN', 'privateSMOTE']
sns.set_style("darkgrid")
plt.figure(figsize=(15,11))
ax = sns.boxplot(data=results_max_privateSMOTE, x='technique', y='test_roc_auc_perdif', palette='Spectral_r', order=order)
ax.margins(x=0.03)
ax.margins(y=0.08)
ax.use_sticky_edges = False
ax.autoscale_view(scaley=True)
sns.set(font_scale=2)
plt.xticks(rotation=45)
plt.xlabel("")
plt.ylabel("Percentage difference of predictive performance (AUC)")
plt.yscale('symlog')
plt.autoscale(True)
plt.show()
figure = ax.get_figure()
figure.savefig(f'{os.path.dirname(os.getcwd())}/output/plots/alltechniques_outofsample_auc.pdf', bbox_inches='tight')

# %%
