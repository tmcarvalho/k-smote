# %%
import re
import numpy as np
import pandas as pd
from os import walk
import seaborn as sns
import matplotlib.pyplot as plt

# %% ORACLE SETTING
# percentage difference in out of sample setting
anonymeter_results = pd.read_csv('../output_analysis/anonymeter.csv')
performance_priv = pd.read_csv("../output_analysis/modeling_results.csv")
# %%
anonymeter_results['ds_complete'] = anonymeter_results['ds_complete'].apply(lambda x: re.sub(r'_qi[0-9]','', x) if (('TVAE' in x) or ('CTGAN' in x) or ('CopulaGAN' in x) or ('dpgan' in x) or ('pategan' in x) or ('smote' in x) or ('under' in x)) else x)
priv_util = anonymeter_results.merge(performance_priv, on=['technique', 'ds_complete', 'ds'], how='left')

# %% Remove ds32, 33 and 38 because they do not have borderline and smote
priv_util = priv_util[~priv_util.ds.isin(['ds32', 'ds33', 'ds38'])]
# %%
priv_util.loc[priv_util['technique']=='PATEGAN', 'technique'] = 'PATE-GAN'
# %%

def bayesian_sign_test(diff_vector, rope_min, rope_max):
    prob_left = np.mean(diff_vector < rope_min)
    prob_rope = np.mean((diff_vector > rope_min) & (diff_vector < rope_max))
    prob_right = np.mean(diff_vector > rope_max)
    alpha = [prob_left, prob_rope, prob_right]
    alpha = [a + 0.0001 for a in alpha]
    res = np.random.dirichlet(alpha, 30000).mean(axis=0)
    return res

def assign_hyperband(df, transfs_name):
    solution_res = pd.DataFrame(columns=['Solution', 'Result', 'Probability'])

    c = 0
    for j in range(3):
        for i in range(len(df)):
            c += 1
            if j == 0:
                solution_res.loc[c] = [transfs_name[i], 'Lose', df[i][j]]
            elif j == 1:
                solution_res.loc[c] = [transfs_name[i], 'Draw', df[i][j]]
            else:
                solution_res.loc[c] = [transfs_name[i], 'Win', df[i][j]]
    return solution_res    

def apply_test(candidates, metric):
    solutions_f1 = [bayesian_sign_test(candidate, -1, 1) for candidate in candidates[metric]]
    solutions_names = candidates['technique'].tolist()

    solution_res = assign_hyperband(solutions_f1, solutions_names)
    return solution_res

def custom_palette(df):
    custom_palette = {'Win': '#27AE60', 'Draw': '#FBC02D', 'Lose': '#2471A3'}
    return {q: custom_palette[q] for q in set(df['Result'])}

def solutions_concat(candidates, metric):
    solutions = apply_test(candidates, metric)
    solutions = solutions[solutions['Probability'] > 0.005]

    palette = custom_palette(solutions)   
    return solutions, palette

def sorter(column):
    reorder = [
        'PPT', 'RUS', 'SMOTE', 'BorderlineSMOTE', 'Copula GAN', 'TVAE',
        'CTGAN', 'DPGAN', 'PATE-GAN', r'$\epsilon$-PrivateSMOTE'
    ]
    cat = pd.Categorical(column, categories=reorder, ordered=True)
    return pd.Series(cat)

# %% PRIVACY FIRST
best_priv = priv_util[(priv_util['value'] == priv_util.groupby(['ds', 'technique'])['value'].transform('min'))].reset_index(drop=True)
best_priv = best_priv.dropna().reset_index(drop=True)
# %%
best_priv_performance = best_priv.loc[best_priv.groupby(['ds', 'technique'])['test_roc_auc_oracle'].idxmax()].reset_index(drop=True)

# %% oracle percentage difference
oracle_priv_performance = best_priv_performance.loc[best_priv_performance.groupby(['ds'])["test_roc_auc_oracle"].idxmax()].reset_index(drop=True)

# %% 
best_priv_performance['test_roc_auc_perdif_oracle'] = None
for i in range(len(best_priv_performance)):
    ds_oracle = oracle_priv_performance.loc[best_priv_performance.at[i,'ds'] == oracle_priv_performance.ds,:].reset_index(drop=True)
    best_priv_performance['test_roc_auc_perdif_oracle'][i] = 100 * (best_priv_performance['test_roc_auc_oracle'][i] - ds_oracle['test_roc_auc_oracle'][0]) / ds_oracle['test_roc_auc_oracle'][0]

best_priv_performance['test_roc_auc_perdif_oracle'] =best_priv_performance['test_roc_auc_perdif_oracle'].astype(np.float)

# %% PRIVACY
solutions_org_candidates_priv, palette_candidates_priv = solutions_concat(best_priv_performance, 'value')   
solutions_org_candidates_priv = solutions_org_candidates_priv.reset_index(drop=True)
solutions_org_candidates_priv = solutions_org_candidates_priv.sort_values(by="Solution", key=sorter)

# %% PREDICTIVE PERFORMANCE
solutions_org_candidates, palette_candidates = solutions_concat(best_priv_performance, 'test_roc_auc_perdif_oracle')   
solutions_org_candidates = solutions_org_candidates.reset_index(drop=True)
solutions_org_candidates = solutions_org_candidates.sort_values(by="Solution", key=sorter)

# %%
sns.set_style("darkgrid")
fig, axes= plt.subplots(2,1,figsize=(11, 6))
sns.histplot(ax=axes[0], data=solutions_org_candidates_priv, stat='probability', multiple='fill', x='Solution', hue='Result', edgecolor='none',
            palette = palette_candidates_priv, shrink=0.8, hue_order=['Draw'])
sns.histplot(ax=axes[1], data=solutions_org_candidates, stat='probability', multiple='fill', x='Solution', hue='Result', edgecolor='none',
            palette = palette_candidates, shrink=0.8, hue_order=['Lose', 'Draw'])
axes[0].axhline(0.5, linewidth=0.5, color='lightgrey')
axes[0].margins(x=0.2)
axes[0].set_xlabel("")
axes[0].set_xticklabels("")
axes[0].set_ylabel('Proportion of probability \n (Linkability)')
axes[1].legend_.remove()
axes[1].axhline(0.5, linewidth=0.5, color='lightgrey')
axes[1].margins(x=0.2)
sns.move_legend(axes[0], bbox_to_anchor=(0.5,1.35), loc='upper center', borderaxespad=0., ncol=3, frameon=False)         
sns.set(font_scale=1.2)
# plt.yticks(np.arange(0, 1.25, 0.25))
plt.xticks(rotation=45)
axes[1].set_ylabel('Proportion of probability (AUC)')
axes[1].set_xlabel('')
# plt.savefig(f'../output/plots/bayes_newprivatesmote.pdf', bbox_inches='tight')

# %%
sns.set_style("darkgrid")
fig, ax= plt.subplots(figsize=(10, 2.7))
sns.histplot(data=solutions_org_candidates, stat='probability', multiple='fill', x='Solution', hue='Result', edgecolor='none',
            palette = palette_candidates, shrink=0.8, hue_order=['Lose', 'Draw'])
ax.axhline(0.5, linewidth=0.5, color='lightgrey')
ax.margins(x=0.2)
ax.set_xlabel("")
ax.set_ylabel('Proportion of probability')
sns.move_legend(ax, bbox_to_anchor=(0.5,1.23), loc='upper center', borderaxespad=0., ncol=3, frameon=False, title="")         
sns.set(font_scale=1.3)
# plt.yticks(np.arange(0, 1.25, 0.25))
plt.xticks(rotation=45)
# plt.savefig(f'../plots/bayes_riskperformance.pdf', bbox_inches='tight')

# %%
###### BEST IN PERFORMANCE
best_performance = priv_util[(priv_util['test_roc_auc_oracle'] == priv_util.groupby(['ds', 'technique'])['test_roc_auc_oracle'].transform('max'))].reset_index(drop=True)
best_performance = best_performance.dropna().reset_index(drop=True)
# %%
best_performance_priv = best_performance.loc[best_performance.groupby(['ds', 'technique'])['value'].idxmin()].reset_index(drop=True)

# %% oracle percentage difference
oracle_performance_priv = best_performance_priv.loc[best_performance_priv.groupby(['ds'])["test_roc_auc_oracle"].idxmax()].reset_index(drop=True)

# %% 
best_performance_priv['test_roc_auc_perdif_oracle'] = None
for i in range(len(best_performance_priv)):
    ds_oracle = oracle_performance_priv.loc[best_performance_priv.at[i,'ds'] == oracle_performance_priv.ds,:].reset_index(drop=True)
    best_performance_priv['test_roc_auc_perdif_oracle'][i] = 100 * (best_performance_priv['test_roc_auc_oracle'][i] - ds_oracle['test_roc_auc_oracle'][0]) / ds_oracle['test_roc_auc_oracle'][0]

best_performance_priv['test_roc_auc_perdif_oracle'] = best_performance_priv['test_roc_auc_perdif_oracle'].astype(np.float)

# %%
performance_priv_candidates, performance_priv_palette = solutions_concat(best_performance_priv, 'test_roc_auc_perdif_oracle')   
performance_priv_candidates = performance_priv_candidates.reset_index(drop=True)
performance_priv_candidates = performance_priv_candidates.sort_values(by="Solution", key=sorter)

# %%
sns.set_style("darkgrid")
fig, ax= plt.subplots(figsize=(10, 2.5))
sns.histplot(data=performance_priv_candidates, stat='probability', multiple='fill', x='Solution', hue='Result', edgecolor='none',
            palette = performance_priv_palette, shrink=0.8, hue_order=['Lose', 'Draw'])
ax.axhline(0.5, linewidth=0.5, color='lightgrey')
ax.margins(x=0.2)
ax.set_xlabel("")
# ax.set_xticklabels("")
ax.set_ylabel('Proportion of probability')
sns.move_legend(ax, bbox_to_anchor=(0.5,1.35), loc='upper center', borderaxespad=0., ncol=3, frameon=False)         
sns.set(font_scale=1.2)
# plt.yticks(np.arange(0, 1.25, 0.25))
plt.xticks(rotation=45)
# plt.savefig(f'../output/plots/bayes_newprivatesmote.pdf', bbox_inches='tight')

# %%
performance_priv_palette_ = {}
for q in set(performance_priv_candidates.Result):
    if q == 'Win':
        performance_priv_palette_[q] = 'tab:green'
    elif q == 'Draw':
        performance_priv_palette_[q] = '#00BFC4'
    elif q == 'Lose':
        performance_priv_palette_[q] = '#839192'
# %%
palette_candidates_ = {}
for q in set(solutions_org_candidates.Result):
    if q == 'Win':
        palette_candidates_[q] = 'tab:green'
    elif q == 'Draw':
        palette_candidates_[q] = '#F1948A'
    elif q == 'Lose':
        palette_candidates_[q] = '#839192' #21618C


# %%
sns.set_style("darkgrid")
fig, axes= plt.subplots(2,1,figsize=(12, 6.5))
sns.histplot(ax=axes[0], data=performance_priv_candidates, stat='probability', multiple='fill', x='Solution', hue='Result', edgecolor='none',
            palette = performance_priv_palette_, shrink=0.8, hue_order=['Lose','Draw'])
sns.histplot(ax=axes[1], data=solutions_org_candidates, stat='probability', multiple='fill', x='Solution', hue='Result', edgecolor='none',
            palette = palette_candidates_, shrink=0.8, hue_order=['Lose', 'Draw'])
axes[0].axhline(0.5, linewidth=0.5, color='lightgrey')
axes[0].margins(x=0.2)
axes[0].set_xlabel("")
axes[0].set_xticklabels("")
axes[0].set_ylabel('Proportion of probability')
#axes[1].legend_.remove()
axes[1].axhline(0.5, linewidth=0.5, color='lightgrey')
axes[1].margins(x=0.2)
sns.move_legend(axes[0], bbox_to_anchor=(0.5,1.35), loc='upper center', borderaxespad=0.,
                ncol=3, frameon=False, title='Result for predictive performance optimisation path')
sns.move_legend(axes[1], bbox_to_anchor=(0.5,1.35), loc='upper center', borderaxespad=0.,
                ncol=3, frameon=False, title='Result for privacy risk optimisation path')         
sns.set(font_scale=1.3)
# plt.yticks(np.arange(0, 1.25, 0.25))
plt.xticks(rotation=45)
axes[1].set_ylabel('Proportion of probability')
axes[1].set_xlabel('')
plt.subplots_adjust(hspace = 0.5)
# plt.savefig(f'../plots/bayes.pdf', bbox_inches='tight')

# %%
