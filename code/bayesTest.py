# %%
import re
import numpy as np
import pandas as pd
from os import walk
import seaborn as sns
import matplotlib.pyplot as plt

# %% ORACLE SETTING
# percentage difference in out of sample setting
all_results_baseorg_out = pd.read_csv('../output/test_cv_roc_auc.csv')

# %%
def BayesianSignTest(diffVector, rope_min, rope_max):
    # for the moment we implement the sign test. Signedrank will follows
    probLeft = np.mean(diffVector < rope_min)
    probRope = np.mean((diffVector > rope_min) & (diffVector < rope_max))
    probRight = np.mean(diffVector > rope_max)
    alpha = [probLeft, probRope, probRight]
    alpha = [a + 0.0001 for a in alpha]
    res = np.random.dirichlet(alpha, 30000).mean(axis=0)

    return res

def assign_hyperband(df, transfs_name):
    solution_res = pd.DataFrame(columns=['Solution', 'Result', 'Probability'])

    c = 0
    for j in range(0, 3):
        for i in range(0, len(df)):
            c += 1
            if j == 0:
                solution_res.loc[c] = [transfs_name[i], 'Lose', df[i][j]]
            elif j == 1:
                solution_res.loc[c] = [transfs_name[i], 'Draw', df[i][j]]
            else:
                solution_res.loc[c] = [transfs_name[i], 'Win', df[i][j]]
    return solution_res    


def apply_test(candidates):
    solutions_f1 = [i for i in candidates['test_roc_auc_perdif_oracle']]
    solutions_names = [i for i in candidates.technique]

    for i in range(0, len(candidates)):
        solutions_f1[i] = BayesianSignTest(solutions_f1[i], -1, 1)

    solution_res = assign_hyperband(solutions_f1, solutions_names)

    return solution_res


def custom_palette(df):
    custom_palette = {}
    for q in set(df.Result):
        if q == 'Win':
            custom_palette[q] = 'tab:green'
        elif q == 'Draw':
            custom_palette[q] = 'orange'
        elif q == 'Lose':
            custom_palette[q] = 'tab:blue'
    return custom_palette  


def solutions_concat(candidates):
    solutions_concat = []  
    solutions = apply_test(candidates)
    solutions = solutions[solutions['Probability'] > 0.005]

    solutions_concat.append(solutions)

    solutions_concat = [f for f in solutions_concat ]
    solutions_concat = pd.concat(solutions_concat)
    palette = custom_palette(solutions_concat)   

    return solutions_concat, palette


def sorter(column):
    reorder = [
        'PPT',
        'Over',
        'RUS',
        'SMOTE',
        'BorderlineSMOTE',
        'Copula GAN',
        'TVAE',
        'CTGAN',
        'PrivateSMOTE'
        ]
    cat = pd.Categorical(column, categories=reorder, ordered=True)
    return pd.Series(cat)

 # %%
results_baseorg_out = all_results_baseorg_out.reset_index(drop=True)
# %%
#results_baseorg_out['test_roc_auc_perdif_oracle'] = results_baseorg_out['test_roc_auc_perdif_oracle'].fillna(0)
baseline_org_max = results_baseorg_out.loc[results_baseorg_out.groupby(['ds', 'technique'])['test_roc_auc_perdif_oracle'].idxmax()].reset_index(drop=True)

# %%
solutions_org_candidates, palette_candidates = solutions_concat(baseline_org_max)   
solutions_org_candidates = solutions_org_candidates.reset_index(drop=True)

# %%
solutions_org_candidates = solutions_org_candidates.loc[solutions_org_candidates['Solution']!='Over']
solutions_org_candidates.loc[solutions_org_candidates['Solution']=='Under', 'Solution'] = 'RUS'
solutions_org_candidates.loc[solutions_org_candidates['Solution']=='Bordersmote', 'Solution'] = 'BorderlineSMOTE'
solutions_org_candidates.loc[solutions_org_candidates['Solution']=='Smote', 'Solution'] = 'SMOTE'
solutions_org_candidates.loc[solutions_org_candidates['Solution']=='copulaGAN', 'Solution'] = 'Copula GAN'
solutions_org_candidates.loc[solutions_org_candidates['Solution']=='privateSMOTE', 'Solution'] = 'PrivateSMOTE'

# %%
solutions_org_candidates = solutions_org_candidates.sort_values(by="Solution", key=sorter)
# %%
sns.set_style("darkgrid")
fig, ax= plt.subplots(figsize=(10, 2.5))
sns.histplot(data=solutions_org_candidates, stat='probability', multiple='fill', x='Solution', hue='Result', edgecolor='none',
            palette = palette_candidates, shrink=0.8, hue_order=['Lose', 'Draw'])
ax.axhline(0.5, linewidth=0.5, color='lightgrey')
ax.margins(x=0.2)
sns.move_legend(ax, bbox_to_anchor=(0.5,1.35), loc='upper center', borderaxespad=0., ncol=3, frameon=False)         
sns.set(font_scale=1.2)
plt.yticks(np.arange(0, 1.25, 0.25))
plt.xticks(rotation=45)
ax.set_ylabel('Proportion of probability')
ax.set_xlabel('')
# plt.savefig(f'../output/plots/bayes_oracle.pdf', bbox_inches='tight')

# %%
