# %%
import re
import numpy as np
import pandas as pd
from os import walk
import seaborn as sns
import matplotlib.pyplot as plt

# %% VALIDATION SETTING
# percentage difference in out of sample setting
# process predictive results
def percentage_difference(baseline_folder, baseline_files, candidate_folder, candidate_files, technique):
    df_concat = []
    c=0
    for baseline in baseline_files:
        for _, transf in enumerate(candidate_files):
            if 'npy' in baseline and 'npy' in transf:
                baseline_result = np.load(f'{baseline_folder}{baseline}', allow_pickle='TRUE').item()
                b = list(map(int, re.findall(r'\d+', baseline.split('.')[0])))[0]
                t = list(map(int, re.findall(r'\d+', transf.split('.')[0])))[0]
            
                if (t not in [0,1,3,13,23,28,34,36,40,48,54,66,87]) and (b==t): 
                    baseline_result = dict([(k, baseline_result[k]) for k in ['model', 'test_f1_weighted']])
                    baseline_result_df = pd.DataFrame(baseline_result.items())
                    baseline_result_df = baseline_result_df.T
                    baseline_result_df = baseline_result_df.rename(columns=baseline_result_df.iloc[0]).drop(baseline_result_df.index[0])

                    candidate_result = np.load(f'{candidate_folder}/{transf}', allow_pickle='TRUE').item()
                    candidate_result = dict([(k, candidate_result[k]) for k in ['model', 'test_f1_weighted']])
                    candidate_result_df = pd.DataFrame.from_dict(candidate_result.items())
                    candidate_result_df = candidate_result_df.T
                    candidate_result_df = candidate_result_df.rename(columns=candidate_result_df.iloc[0]).drop(candidate_result_df.index[0])
 
                    # calculate the percentage difference
                    # 100 * (Sc - Sb) / Sb
                    #print(candidate_result_df)
                    candidate_result_df['test_f1_weighted_perdif'] = 100 * (candidate_result_df['test_f1_weighted'] - baseline_result_df['test_f1_weighted']) / baseline_result_df['test_f1_weighted']
                    
                    if str(candidate_result_df['model'][1]).startswith("{RandomForest"):
                        candidate_result_df['classifier'] = 'Random Forest'
                    if str(candidate_result_df['model'][1]).startswith("{XGB"):
                        candidate_result_df['classifier'] = 'XGBoost'
                    if str(candidate_result_df['model'][1]).startswith("{Logistic"):
                        candidate_result_df['classifier'] = 'Logistic Regression'   

                    # get technique
                    if technique!='smote_under_over':
                        candidate_result_df.loc[:, 'technique'] = technique
                    else:
                        candidate_result_df.loc[:, 'technique'] = transf.split('_')[1].title()
                    
                    # get dataset number
                    candidate_result_df['ds'] = transf.split('_')[0]

                    # concat each test result
                    if c == 0:
                        df_concat = candidate_result_df
                        c += 1
                    else:     
                        df_concat = pd.concat([df_concat, candidate_result_df])

    return df_concat    

# %% 
# path to predictive results
baseline_folder = '../output/modeling/original/test/'
_, _, baseline_file = next(walk(f'{baseline_folder}'))

# %% PPT
ppt_folder = '../output/modeling/PPT/test/'
_, _, ppt_file = next(walk(f'{ppt_folder}'))

baseline_org_ppt = percentage_difference(baseline_folder, baseline_file, ppt_folder, ppt_file, 'PPT')
# %% smote_under_over
smote_under_over_folder = '../output/modeling/smote_under_over/test/'
_, _, smote_under_over_file = next(walk(f'{smote_under_over_folder}'))

baseline_org_smote_under_over = percentage_difference(baseline_folder, baseline_file, smote_under_over_folder, smote_under_over_file, 'smote_under_over')

# %% smote_singleouts
smote_singleouts_folder = '../output/modeling/smote_singleouts/test/'
_, _, smote_singleouts_file = next(walk(f'{smote_singleouts_folder}'))

baseline_org_smote_singleouts = percentage_difference(baseline_folder, baseline_file, smote_singleouts_folder, smote_singleouts_file, 'Synthetisation \n one class')

# %% smote_singleouts_scratch
smote_singleouts_scratch_folder = '../output/modeling/smote_singleouts_scratch/test/'
_, _, smote_singleouts_scratch_file = next(walk(f'{smote_singleouts_scratch_folder}'))

baseline_org_smote_singleouts_scratch = percentage_difference(baseline_folder, baseline_file, smote_singleouts_scratch_folder, smote_singleouts_scratch_file, 'Synthetisation \n two classes')

# %% concat all data sets
results_baseline_org = pd.concat([baseline_org_ppt, baseline_org_smote_under_over, baseline_org_smote_singleouts, baseline_org_smote_singleouts_scratch])
# %%
# results_baseline_org.to_csv('../output/bayesianTest_baseline_org.csv', index=False)
results_baseline_org = pd.read_csv('../output/bayesianTest_baseline_org.csv')

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

     
# %%
def apply_test(candidates):
    solutions_f1 = [i for i in candidates.test_f1_weighted_perdif]
    solutions_names = [i for i in candidates.technique]

    for i in range(0, len(candidates)):
        solutions_f1[i] = BayesianSignTest(solutions_f1[i], -1, 1)

    solution_res = assign_hyperband(solutions_f1, solutions_names)

    return solution_res


# %%
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
# %% 
def solutions_concat(candidates):
    solutions_concat = []  
    solutions = apply_test(candidates)
    solutions = solutions[solutions['Probability'] > 0.05]

    solutions_concat.append(solutions)

    solutions_concat = [f for f in solutions_concat ]
    solutions_concat = pd.concat(solutions_concat)
    palette = custom_palette(solutions_concat)   

    return solutions_concat, palette

# %%
baseline_org_max = results_baseline_org.groupby(['ds', 'technique'], as_index=False)['test_f1_weighted_perdif'].max()
# %%
solutions_org_candidates, palette_candidates = solutions_concat(baseline_org_max)   
solutions_org_candidates = solutions_org_candidates.reset_index(drop=True)
# %%
def sorter(column):
    reorder = [
        'PPT',
        'Over',
        'Under',
        'Smote',
        'Synthetisation \n one class',
        'Synthetisation \n two classes']

    cat = pd.Categorical(column, categories=reorder, ordered=True)
    return pd.Series(cat)

solutions_org_candidates = solutions_org_candidates.sort_values(by="Solution", key=sorter)
# %%
def plot_hyperband(solutions, name):
    sns.set_style("darkgrid")
    fig = plt.figure(figsize=(8, 3))
    gg = sns.histplot(data=solutions, stat='probability', multiple='fill', x='Solution', hue='Result', edgecolor='none',
                palette = palette_candidates, shrink=0.8, hue_order=['Lose', 'Draw', 'Win'])
    gg.axhline(0.5, linewidth=0.5, color='lightgrey')
    gg.margins(x=0.2)
    sns.move_legend(gg, bbox_to_anchor=(0.5,1.3), loc='upper center', borderaxespad=0., ncol=3, frameon=False)         
    sns.set(font_scale=1.2)
    plt.yticks(np.arange(0, 1.25, 0.25))
    plt.xticks(rotation=30)
    gg.set_ylabel('Proportion of probability')
    gg.set_xlabel('')
    plt.savefig(f'../output/plots/{name}.pdf', bbox_inches='tight')

# %%
plot_hyperband(solutions_org_candidates, 'baseline_org')

# %% ##################################
# Repeat for PPT as candidate

ppt = results_baseline_org.groupby(['technique', 'ds'], as_index=False)['test_f1_weighted', 'classifier', 'ds', 'technique'].max()
ppt = ppt.loc[ppt['technique']=='PPT', :].reset_index(drop=True)

# %%
candidates = results_baseline_org.loc[results_baseline_org['technique']!='PPT',:].reset_index(drop=True)
# %%
def percentage_difference_ppt(ppt, candidates):
    df_concat = []
    c=0
    for i in range(len(ppt)):
        for j in range(len(candidates)):
            if candidates['ds'][j] == ppt['ds'][i]:
            # calculate the percentage difference
                # 100 * (Sc - Sb) / Sb
                candidates['test_f1_weighted_perdif'][i] = 100 * (candidates['test_f1_weighted'][i] - ppt['test_f1_weighted'][i]) / ppt['test_f1_weighted'][i]

        return candidates    
# %%
results_baseline_ppt = percentage_difference_ppt(ppt, candidates)
# %%
results_baseline_ppt.to_csv('../output/bayesianTest_baseline_ppt.csv', index=False)
#results_baseline_ppt = pd.read_csv('../output/bayesianTest_baseline_ppt.csv')

# %%
baseline_ppt_max = results_baseline_ppt.groupby(['ds', 'technique'], as_index=False)['test_f1_weighted_perdif'].max()
# %%
solutions_ppt_candidates, palette_candidates = solutions_concat(baseline_ppt_max)   
solutions_ppt_candidates = solutions_ppt_candidates.reset_index(drop=True)

solutions_ppt_candidates = solutions_ppt_candidates.sort_values(by="Solution", key=sorter)

# %%
plot_hyperband(solutions_ppt_candidates, 'baseline_ppt')

# %%
