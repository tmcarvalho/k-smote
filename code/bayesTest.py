# %%
from logging import error
import re
import numpy as np
import pandas as pd
from os import walk
import seaborn as sns
import matplotlib.pyplot as plt

# %% VALIDATION SETTING
# percentage difference in out of sample setting
# process predictive results
def percentage_difference_org(baseline_folder, baseline_files, candidate_folder, candidate_files, technique):
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
# ppt_folder = '../output/modeling/PPT/test/'
ppt_folder = '../output/modeling/PPT_ARX/test/'
_, _, ppt_file = next(walk(f'{ppt_folder}'))

baseline_org_ppt = percentage_difference_org(baseline_folder, baseline_file, ppt_folder, ppt_file, 'PPT')
# %% smote_under_over
smote_under_over_folder = '../output/modeling/smote_under_over/test/'
_, _, smote_under_over_file = next(walk(f'{smote_under_over_folder}'))

baseline_org_smote_under_over = percentage_difference_org(baseline_folder, baseline_file, smote_under_over_folder, smote_under_over_file, 'smote_under_over')

# %% Borderline SMOTE
borderline_folder = '../output/modeling/borderlineSmote/test/'
_, _, borderline_file = next(walk(f'{borderline_folder}'))

baseline_org_borderline = percentage_difference_org(baseline_folder, baseline_file, borderline_folder, borderline_file, 'BorderlineSMOTE')

# %% gaussianCopula
gaussianCopula_folder = '../output/modeling/gaussianCopula/test/'
_, _, gaussianCopula_file = next(walk(f'{gaussianCopula_folder}'))

baseline_org_gaussianCopula = percentage_difference_org(baseline_folder, baseline_file, gaussianCopula_folder, gaussianCopula_file, 'Gaussian Copula')

# %% copulaGAN
copulaGAN_folder = '../output/modeling/copulaGAN/test/'
_, _, copulaGAN_file = next(walk(f'{copulaGAN_folder}'))

baseline_org_copulaGAN = percentage_difference_org(baseline_folder, baseline_file, copulaGAN_folder, copulaGAN_file, 'Copula GAN')

# %% TVAE
tvae_folder = '../output/modeling/TVAE/test/'
_, _, tvae_file = next(walk(f'{tvae_folder}'))

baseline_org_tvae = percentage_difference_org(baseline_folder, baseline_file, tvae_folder, tvae_file, 'TVAE')

# %% CTGAN
ctgan_folder = '../output/modeling/CTGAN/test/'
_, _, ctgan_file = next(walk(f'{ctgan_folder}'))

baseline_org_ctgan = percentage_difference_org(baseline_folder, baseline_file, ctgan_folder, ctgan_file, 'CTGAN')

# %% smote_singleouts
smote_singleouts_folder = '../output/modeling/smote_singleouts/test/'
_, _, smote_singleouts_file = next(walk(f'{smote_singleouts_folder}'))

baseline_org_smote_singleouts = percentage_difference_org(baseline_folder, baseline_file, smote_singleouts_folder, smote_singleouts_file, 'privateSMOTE')

# %% smote_singleouts_scratch
smote_singleouts_scratch_folder = '../output/modeling/smote_singleouts_scratch/test/'
_, _, smote_singleouts_scratch_file = next(walk(f'{smote_singleouts_scratch_folder}'))

baseline_org_smote_singleouts_scratch = percentage_difference_org(baseline_folder, baseline_file, smote_singleouts_scratch_folder, smote_singleouts_scratch_file, 'privateSMOTE \n regardless of \n the class')

# %% concat all data sets
results_baseline_org = pd.concat([baseline_org_ppt, baseline_org_smote_under_over, baseline_org_borderline, baseline_org_gaussianCopula, baseline_org_copulaGAN, baseline_org_tvae, baseline_org_ctgan, baseline_org_smote_singleouts, baseline_org_smote_singleouts_scratch])
# %%
# results_baseline_org.to_csv('../output/bayesianTest_baseline_org.csv', index=False)
# results_baseline_org = pd.read_csv('../output/bayesianTest_baseline_org.csv')

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
    solutions_f1 = [i for i in candidates.test_f1_weighted_perdif]
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
        'Gaussian Copula',
        'Copula GAN',
        'TVAE',
        'CTGAN',
        'privateSMOTE',
        'privateSMOTE \n regardless of \n the class']
    cat = pd.Categorical(column, categories=reorder, ordered=True)
    return pd.Series(cat)

 # %%
results_baseline_org = results_baseline_org.reset_index(drop=True)
results_baseline_org['test_f1_weighted_perdif'] = pd.to_numeric(results_baseline_org['test_f1_weighted_perdif'])
# %%
baseline_org_max = results_baseline_org.loc[results_baseline_org.groupby(['ds', 'technique'])['test_f1_weighted_perdif'].idxmax()].reset_index(drop=True)

# %%
solutions_org_candidates, palette_candidates = solutions_concat(baseline_org_max)   
solutions_org_candidates = solutions_org_candidates.reset_index(drop=True)

# %%
solutions_org_candidates = solutions_org_candidates.loc[solutions_org_candidates['Solution']!='Over']
solutions_org_candidates.loc[solutions_org_candidates['Solution']=='Under', 'Solution'] = 'RUS'
solutions_org_candidates.loc[solutions_org_candidates['Solution']=='Smote', 'Solution'] = 'SMOTE'

# %%
solutions_org_candidates = solutions_org_candidates.sort_values(by="Solution", key=sorter)
# %%
sns.set_style("darkgrid")
fig, ax= plt.subplots(figsize=(10, 2.5))
sns.histplot(data=solutions_org_candidates, stat='probability', multiple='fill', x='Solution', hue='Result', edgecolor='none',
            palette = palette_candidates, shrink=0.8, hue_order=['Lose', 'Draw', 'Win'])
ax.axhline(0.5, linewidth=0.5, color='lightgrey')
ax.margins(x=0.2)
sns.move_legend(ax, bbox_to_anchor=(0.5,1.35), loc='upper center', borderaxespad=0., ncol=3, frameon=False)         
sns.set(font_scale=1.2)
plt.yticks(np.arange(0, 1.25, 0.25))
plt.xticks(rotation=45)
ax.set_ylabel('Proportion of probability')
ax.set_xlabel('')
# plt.savefig(f'../output/plots/baseline_org_all.pdf', bbox_inches='tight')


# %% ##################################
# Repeat for PPT as candidate

ppt = results_baseline_org.groupby(['technique', 'ds'], as_index=False)['test_f1_weighted', 'classifier', 'ds', 'technique'].max()
ppt = ppt.loc[ppt['technique']=='PPT', :].reset_index(drop=True)

# %%
candidates = results_baseline_org.loc[results_baseline_org['technique']!='PPT',:].reset_index(drop=True)
# %%
def percentage_difference(baseline, candidates):
    candidates['test_f1_weighted_perdif']=None
    for i in range(0, len(baseline)):
        for j in range(0, len(candidates)):
            if baseline['ds'][i] == candidates['ds'][j]:
                # calculate the percentage difference
                # 100 * (Sc - Sb) / Sb
                candidates['test_f1_weighted_perdif'][j] = 100 * (candidates['test_f1_weighted'][j] - baseline['test_f1_weighted'][i]) / baseline['test_f1_weighted'][i]

    return candidates    
# %%
results_baseline_ppt = percentage_difference(ppt, candidates)
# %%
results_baseline_ppt['test_f1_weighted_perdif'] = results_baseline_ppt['test_f1_weighted_perdif'].astype(float)
# %%
#results_baseline_ppt.to_csv('../output/bayesianTest_baseline_ppt.csv', index=False)
# results_baseline_ppt = pd.read_csv('../output/bayesianTest_baseline_ppt.csv')

# %%
baseline_ppt_max = results_baseline_ppt.loc[results_baseline_ppt.groupby(['ds', 'technique'])['test_f1_weighted_perdif'].idxmax()].reset_index(drop=True)
# %%
solutions_ppt_candidates, palette_candidates = solutions_concat(baseline_ppt_max)   
solutions_ppt_candidates = solutions_ppt_candidates.reset_index(drop=True)
solutions_ppt_candidates = solutions_ppt_candidates.loc[solutions_ppt_candidates['Solution']!='Over']
solutions_ppt_candidates.loc[solutions_ppt_candidates['Solution']=='Under', 'Solution'] = 'RUS'
solutions_ppt_candidates.loc[solutions_ppt_candidates['Solution']=='Smote', 'Solution'] = 'SMOTE'
solutions_ppt_candidates = solutions_ppt_candidates.sort_values(by="Solution", key=sorter)

# %%
solutions_ppt_candidates = solutions_ppt_candidates.loc[solutions_ppt_candidates['Solution']!='Over']
sns.set_style("darkgrid")
fig = plt.figure(figsize=(9, 2.5))
gg = sns.histplot(data=solutions_ppt_candidates, stat='probability', multiple='fill', x='Solution', hue='Result', edgecolor='none',
            palette = palette_candidates, shrink=0.8, hue_order=['Lose', 'Draw', 'Win'])
gg.axhline(0.5, linewidth=0.5, color='lightgrey')
gg.margins(x=0.2)
sns.move_legend(gg, bbox_to_anchor=(0.5,1.3), loc='upper center', borderaxespad=0., ncol=3, frameon=False)         
sns.set(font_scale=1)
plt.yticks(np.arange(0, 1.25, 0.25))
plt.xticks(rotation=45)
gg.set_ylabel('Proportion of probability')
gg.set_xlabel('')
# plt.savefig(f'../output/plots/baseline_ppt_arx.pdf', bbox_inches='tight')

# %% all against all ################################
def add_original(baseline_folder, baseline_files, candidates):
    for baseline in baseline_files:
        if 'npy' in baseline:
            baseline_result = np.load(f'{baseline_folder}{baseline}', allow_pickle='TRUE').item()
            b = int(baseline.split(".")[0])
            if b not in [0,1,3,13,23,28,34,36,40,48,54,66,87]: 
                baseline_result = dict([(k, baseline_result[k]) for k in ['model', 'test_f1_weighted']])
                baseline_result_df = pd.DataFrame(baseline_result.items())
                baseline_result_df = baseline_result_df.T
                baseline_result_df = baseline_result_df.rename(columns=baseline_result_df.iloc[0]).drop(baseline_result_df.index[0])

                if str(baseline_result_df['model'][1]).startswith("{RandomForest"):
                    baseline_result_df['classifier'] = 'Random Forest'
                if str(baseline_result_df['model'][1]).startswith("{XGB"):
                    baseline_result_df['classifier'] = 'XGBoost'
                if str(baseline_result_df['model'][1]).startswith("{Logistic"):
                    baseline_result_df['classifier'] = 'Logistic Regression'   

                # get technique
                baseline_result_df.loc[:, 'technique'] = 'Original'

                # get dataset number
                baseline_result_df['ds'] = f'ds{baseline.split(".")[0]}'

                # concat each test result
                candidates = pd.concat([candidates, baseline_result_df])
    
    return candidates
# %%
# path to predictive results
baseline_folder = '../output/modeling/original/test/'
_, _, baseline_file = next(walk(f'{baseline_folder}'))

# %%
tech_results = results_baseline_org[['test_f1_weighted', 'technique', 'classifier', 'ds']]
# %%
all_results = add_original(baseline_folder, baseline_file, tech_results)
# %%
del all_results['model']
# %%
all_results = all_results.reset_index(drop=True)
all_results = all_results.rename_axis('idx').reset_index()
all_results['test_f1_weighted'] = all_results['test_f1_weighted'].astype(float)
# %%
max_results = all_results.loc[all_results.groupby(['ds'])['test_f1_weighted'].idxmax()].reset_index(drop=True)
# %%
all_results = all_results.loc[~all_results['idx'].isin(max_results['idx'])].reset_index(drop=True)

# %%
all_results['test_f1_weighted_perdif']=None
for i in range(0, len(max_results)):
    for j in range(0, len(all_results)):
        if max_results['ds'][i] == all_results['ds'][j]:
            # calculate the percentage difference
            # 100 * (Sc - Sb) / Sb
            all_results['test_f1_weighted_perdif'][j] = 100 * (all_results['test_f1_weighted'][j] - max_results['test_f1_weighted'][i]) / max_results['test_f1_weighted'][i]

# %%
all_results['test_f1_weighted_perdif'] = all_results['test_f1_weighted_perdif'].astype(float)
# %%
all_results_percdiff_max = all_results.loc[all_results.groupby(['ds', 'technique'])['test_f1_weighted_perdif'].idxmax()].reset_index(drop=True)
# %%
solutions_candidates, palette_candidates = solutions_concat(all_results_percdiff_max)   
solutions_candidates = solutions_candidates.reset_index(drop=True)

def sorter_org(column):
    reorder = [
        'Original',
        'PPT',
        'Over',
        'Under',
        'Smote',
        'BorderlineSMOTE',
        'Gaussian Copula',
        'Copula GAN',
        'TVAE',
        'CTGAN',
        'privateSMOTE',
        'privateSMOTE \n regardless of \n the class']
    cat = pd.Categorical(column, categories=reorder, ordered=True)
    return pd.Series(cat)

solutions_candidates = solutions_candidates.sort_values(by="Solution", key=sorter_org)

# %%
solutions_candidates = solutions_candidates.loc[solutions_candidates['Solution']!='Over']
solutions_candidates.loc[solutions_candidates['Solution']=='Under', 'Solution'] = 'RUS'
solutions_candidates.loc[solutions_candidates['Solution']=='Smote', 'Solution'] = 'SMOTE'

# %%
sns.set_style("darkgrid")
fig = plt.figure(figsize=(10, 2.5))
gg = sns.histplot(data=solutions_candidates, stat='probability', multiple='fill', x='Solution', hue='Result', edgecolor='none',
            palette = palette_candidates, shrink=0.9, hue_order=['Lose', 'Draw'])
gg.axhline(0.5, linewidth=0.5, color='lightgrey')
gg.margins(x=0.2)
sns.move_legend(gg, bbox_to_anchor=(0.5,1.35), loc='upper center', borderaxespad=0., ncol=3, frameon=False)         
sns.set(font_scale=1.2)
plt.yticks(np.arange(0, 1.25, 0.25))
plt.xticks(rotation=45)
gg.set_ylabel('Proportion of probability')
gg.set_xlabel('')
plt.savefig(f'../output/plots/baseline_best_all.pdf', bbox_inches='tight')

# %%
