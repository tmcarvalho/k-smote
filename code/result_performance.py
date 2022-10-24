"""Predictive performance analysis
This script will analyse the predictive performance in the out-of-sample.
"""
# %%
from os import walk
import os
import re
import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt

# %% 
# process predictive results
def percentage_difference_org(baseline_folder, candidate_folder, technique):
    df_concat_cv = []
    c=0
    _,_,baseline_result_files = next(walk(f'{baseline_folder}/test/'))
    _,_,candidate_result_files = next(walk(f'{candidate_folder}/test/'))

    for baseline in baseline_result_files:
        baseline_result_cv_train = pd.read_csv(f'{baseline_folder}/validation/{baseline}')
        baseline_result_cv = pd.read_csv(f'{baseline_folder}/test/{baseline}')
        
        # guaranteeing that we have the best model in the grid search instead of 3 models
        best_baseline_cv = baseline_result_cv_train.loc[baseline_result_cv_train['rank_test_roc_auc_curve'] == 1,:].reset_index(drop=True)
        baseline_result_cv_best = baseline_result_cv.loc[baseline_result_cv.model == best_baseline_cv.model[0],:]
        
        b = list(map(int, re.findall(r'\d+', baseline.split('.')[0])))[0]
        transf_files = [fl for fl in candidate_result_files if list(map(int, re.findall(r'\d+', fl.split('_')[0])))[0] == b]

        for transf_file in transf_files:
            candidate_result_cv_train = pd.read_csv(f'{candidate_folder}/validation/{transf_file}')
            candidate_result_cv = pd.read_csv(f'{candidate_folder}/test/{transf_file}')
            
            # calculate the percentage difference
            # 100 * (Sc - Sb) / Sb
            best_candidate_cv = candidate_result_cv_train.loc[candidate_result_cv_train['rank_test_roc_auc_curve'] == 1,:].reset_index(drop=True)
            candidate_result_cv_best = candidate_result_cv.loc[candidate_result_cv.model == best_candidate_cv.model[0],:]
            candidate_result_cv_best['test_roc_auc_perdif'] = 100 * (candidate_result_cv_best['test_roc_auc'] - baseline_result_cv_best['test_roc_auc']) / baseline_result_cv_best['test_roc_auc']
            # candidate_result_cv['test_fscore_perdif'] = 100 * (candidate_result_cv['test_f1_weighted'] - baseline_result_cv['test_f1_weighted']) / baseline_result_cv['test_f1_weighted']
            # candidate_result_cv['test_gmean_perdif'] = 100 * (candidate_result_cv['test_gmean'] - baseline_result_cv['test_gmean']) / baseline_result_cv['test_gmean']

            oracle = candidate_result_cv.loc[candidate_result_cv['test_roc_auc'].idxmax(),'test_roc_auc']
            candidate_result_cv_best['test_roc_auc_perdif_oracle'] = 100 * (candidate_result_cv_best['test_roc_auc'] - oracle) / oracle

            # get technique
            if technique=='resampling':
                candidate_result_cv_best.loc[:, 'technique'] = transf_file.split('_')[1].title()
            elif technique=='deep_learning':
                candidate_result_cv_best.loc[:, 'technique'] = transf_file.split('_')[1]
            else:
                candidate_result_cv_best.loc[:, 'technique'] = technique
                
            
            # get dataset number
            candidate_result_cv_best['ds'] = transf_file.split('_')[0]
            candidate_result_cv_best['ds_complete'] = transf_file

            # concat each test result
            if c == 0:
                df_concat_cv = candidate_result_cv_best
                c += 1
            else:     
                df_concat_cv = pd.concat([df_concat_cv, candidate_result_cv_best])

    return df_concat_cv    

# %% 
# path to predictive results
orig_folder = '../output/modeling/original/'

# %% PPT
ppt_folder = '../output/modeling/PPT_ARX/'
baseorg_ppt = percentage_difference_org(orig_folder, ppt_folder, 'PPT')
# %% resampling
resampling_folder = '../output/modeling/re-sampling/'
baseorg_resampling = percentage_difference_org(orig_folder, resampling_folder, 'resampling')

# %% deep learning
deeplearn_folder = '../output/modeling/deep_learning/'
_, _, deeplearn_file = next(walk(f'{deeplearn_folder}'))
baseorg_deeplearn = percentage_difference_org(orig_folder, deeplearn_folder, 'deep_learning')

# %% smote_singleouts
smote_singleouts_folder = '../output/modeling/smote_singleouts/'
baseorg_smote_singleouts = percentage_difference_org(orig_folder, smote_singleouts_folder, 'privateSMOTE A')

# %% smote_singleouts_scratch
smote_singleouts_scratch_folder = '../output/modeling/smote_singleouts_scratch/'

baseorg_smote_singleouts_scratch = percentage_difference_org(orig_folder, smote_singleouts_scratch_folder, 'privateSMOTE B')

# %% concat all data sets
results_baseorg_cv = pd.concat([baseorg_resampling, baseorg_deeplearn, baseorg_smote_singleouts])
# %%
results_baseorg_cv.to_csv('../output/test_cv_roc_auc.csv', index=False)
# %%
# all_results_baseorg_cv = pd.read_csv('../output/test_cv_roc_auc.csv')
# %%
results_baseorg_cv['test_roc_auc_perdif'] = results_baseorg_cv['test_roc_auc_perdif'].fillna(0)
#results_max_prSmote = results_baseorg_cv.loc[results_baseorg_cv.technique=='privateSMOTE A',:].reset_index(drop=True)
#results_max_prSmote = results_max_prSmote.loc[results_max_prSmote.groupby(by=['ds', 'technique'])['test_roc_auc_perdif'].idxmax(),:]
results_max = results_baseorg_cv.groupby(['ds', 'technique'], as_index=False)['test_roc_auc_perdif'].max()

# %%
results_max = results_max.loc[results_max['technique']!='Over']
results_max.loc[results_max['technique']=='Under', 'technique'] = 'RUS'
results_max.loc[results_max['technique']=='Bordersmote', 'technique'] = 'BorderlineSMOTE'
results_max.loc[results_max['technique']=='Smote', 'technique'] = 'SMOTE'
results_max.loc[results_max['technique']=='copulaGAN', 'technique'] = 'Copula GAN'

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
plt.ylabel("Percentage difference of predictive performance (AUC)")
plt.yscale('symlog')
plt.autoscale(True)
plt.show()
#figure = ax.get_figure()
#figure.savefig(f'{os.path.dirname(os.getcwd())}/output/plots/performance.pdf', bbox_inches='tight')


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
# figure = ax.get_figure()
# figure.savefig(f'{os.path.dirname(os.getcwd())}/output/plots/alltechniques_outofsample_auc.pdf', bbox_inches='tight')

# %%
