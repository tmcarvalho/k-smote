"""Predictive performance analysis
This script will analyse the predictive performance in the out-of-sample.
"""
# %%
from cmath import nan
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
    df_concat_out = []
    c=0
    _,_,baseline_result_files = next(walk(f'{baseline_folder}/test/'))
    _,_,candidate_result_files = next(walk(f'{candidate_folder}/test/'))

    for baseline in baseline_result_files:
        baseline_result_cv = pd.read_csv(f'{baseline_folder}/test/{baseline}')
        baseline_result_out = pd.read_csv(f'{baseline_folder}/outofsample/{baseline}')
        baseline_result_cv_train = pd.read_csv(f'{baseline_folder}/validation/{baseline}')
        baseline_result_out_train = pd.read_csv(f'{baseline_folder}/outofsample_train/{baseline}')
        # guaranteeing that we have the best model in the grid search instead of 3 models
        best_baseline_cv = baseline_result_cv_train.loc[baseline_result_cv_train['mean_test_roc_auc_curve'] == baseline_result_cv_train['mean_test_roc_auc_curve'].max(),:].reset_index(drop=True)
        best_baseline_out = baseline_result_out_train.loc[baseline_result_out_train['mean_test_roc_auc_curve'] == baseline_result_out_train['mean_test_roc_auc_curve'].max(),:].reset_index(drop=True)
        baseline_result_cv_best = baseline_result_cv.loc[baseline_result_cv.model == best_baseline_cv.model[0],:]
        baseline_result_out_best = baseline_result_out.loc[baseline_result_out.model == best_baseline_out.model[0],:]

        b = list(map(int, re.findall(r'\d+', baseline.split('.')[0])))[0]
        transf_files = [fl for fl in candidate_result_files if list(map(int, re.findall(r'\d+', fl.split('_')[0])))[0] == b]

        for transf_file in transf_files:
            candidate_result_cv = pd.read_csv(f'{candidate_folder}/test/{transf_file}')
            candidate_result_out = pd.read_csv(f'{candidate_folder}/outofsample/{transf_file}')
            candidate_result_cv_train = pd.read_csv(f'{candidate_folder}/validation/{transf_file}')
            candidate_result_out_train = pd.read_csv(f'{candidate_folder}/outofsample_train/{transf_file}')
            # calculate the percentage difference
            # 100 * (Sc - Sb) / Sb
            best_candidate_cv = candidate_result_cv_train.loc[candidate_result_cv_train['mean_test_roc_auc_curve'] == candidate_result_cv_train['mean_test_roc_auc_curve'].max(),:].reset_index(drop=True)
            best_candidate_out = candidate_result_out_train.loc[candidate_result_out_train['mean_test_roc_auc_curve'] == candidate_result_out_train['mean_test_roc_auc_curve'].max(),:].reset_index(drop=True)
            candidate_result_cv_best = candidate_result_cv.loc[candidate_result_cv.model == best_candidate_cv.model[0],:]
            candidate_result_cv_best['test_roc_auc_perdif'] = 100 * (candidate_result_cv_best['test_roc_auc'] - baseline_result_cv_best['test_roc_auc']) / baseline_result_cv_best['test_roc_auc']
            # candidate_result_cv['test_fscore_perdif'] = 100 * (candidate_result_cv['test_f1_weighted'] - baseline_result_cv['test_f1_weighted']) / baseline_result_cv['test_f1_weighted']
            # candidate_result_cv['test_gmean_perdif'] = 100 * (candidate_result_cv['test_gmean'] - baseline_result_cv['test_gmean']) / baseline_result_cv['test_gmean']
            # candidate_result_cv['test_roc_auc_perdif'] = 100 * (candidate_result_cv['test_roc_auc'] - baseline_result_cv['test_roc_auc']) / baseline_result_cv['test_roc_auc']

            # out of sample
            candidate_result_out_best = candidate_result_out.loc[candidate_result_out.model == best_candidate_out.model[0],:]
            candidate_result_out_best['test_roc_auc_perdif'] = 100 * (candidate_result_out_best['test_roc_auc'] - baseline_result_out_best['test_roc_auc']) / baseline_result_out_best['test_roc_auc']
            # candidate_result_out['test_fscore_perdif'] = 100 * (candidate_result_out['test_f1_weighted'] - baseline_result_out['test_f1_weighted']) / baseline_result_out['test_f1_weighted']
            # candidate_result_out['test_gmean_perdif'] = 100 * (candidate_result_out['test_gmean'] - baseline_result_out['test_gmean']) / baseline_result_out['test_gmean']
            # candidate_result_out['test_roc_auc_perdif'] = 100 * (candidate_result_out['test_roc_auc'] - baseline_result_out['test_roc_auc']) / baseline_result_out['test_roc_auc']

            # get technique
            if technique!='resampling':
                candidate_result_cv_best.loc[:, 'technique'] = technique
                candidate_result_out_best.loc[:, 'technique'] = technique
            elif technique=='deep_learning':
                candidate_result_cv_best.loc[:, 'technique'] = transf_file.split('_')[1]
                candidate_result_out_best.loc[:, 'technique'] = transf_file.split('_')[1]
            else:
                candidate_result_cv_best.loc[:, 'technique'] = transf_file.split('_')[1].title()
                candidate_result_out_best.loc[:, 'technique'] = transf_file.split('_')[1].title()
            
            # get dataset number
            candidate_result_cv_best['ds'] = transf_file.split('_')[0]
            candidate_result_out_best['ds'] = transf_file.split('_')[0]
            candidate_result_cv_best['ds_complete'] = transf_file
            candidate_result_out_best['ds_complete'] = transf_file

            # concat each test result
            if c == 0:
                df_concat_cv = candidate_result_cv_best
                df_concat_out = candidate_result_out_best
                c += 1
            else:     
                df_concat_cv = pd.concat([df_concat_cv, candidate_result_cv_best])
                df_concat_out = pd.concat([df_concat_out, candidate_result_out_best])

    return df_concat_cv, df_concat_out    

# %% 
# path to predictive results
orig_folder = '../output/modeling/original/'

# %% PPT
ppt_folder = '../output/modeling/PPT_ARX/'
_, _, ppt_file = next(walk(f'{ppt_folder}'))

baseline_org_ppt = percentage_difference_org(baseline_folder, baseline_file, ppt_folder, ppt_file, 'PPT')
# %% resampling
resampling_folder = '../output/modeling/re-sampling/'

baseorg_resampling_cv, baseorg_resampling_out = percentage_difference_org(orig_folder, resampling_folder, 'resampling')

# %% deep learning
copulaGAN_folder = '../output/modeling/deep_learning/'
_, _, copulaGAN_file = next(walk(f'{copulaGAN_folder}'))

baseline_org_copulaGAN = percentage_difference_org(baseline_folder, baseline_file, copulaGAN_folder, copulaGAN_file, 'deep_learning')

# %% smote_singleouts
smote_singleouts_folder = '../output/modeling/smote_singleouts/'

baseorg_smote_singleouts_cv, baseorg_smote_singleouts_out = percentage_difference_org(orig_folder, smote_singleouts_folder, 'privateSMOTE A')

# %% smote_singleouts_scratch
smote_singleouts_scratch_folder = '../output/modeling/smote_singleouts_scratch/'
_, _, smote_singleouts_scratch_file = next(walk(f'{smote_singleouts_scratch_folder}'))

baseline_org_smote_singleouts_scratch = percentage_difference_org(baseline_folder, baseline_file, smote_singleouts_scratch_folder, smote_singleouts_scratch_file, 'privateSMOTE B')

# %% concat all data sets
results_baseorg_cv = pd.concat([baseorg_resampling_cv, baseorg_smote_singleouts_cv])
results_baseorg_out = pd.concat([baseorg_resampling_out, baseorg_smote_singleouts_out])
# %%
results_baseorg_cv.to_csv('../output/test_cv_roc_auc.csv', index=False)
results_baseorg_out.to_csv('../output/test_outofsample_roc_auc.csv', index=False)
# %%
# all_results_baseorg_cv = pd.read_csv('../output/test_cv_roc_auc.csv')
# %%
results_baseorg_cv['test_roc_auc_perdif'] = results_baseorg_cv['test_roc_auc_perdif'].fillna(0)
results_max = results_baseorg_cv.groupby(['ds', 'technique'], as_index=False)['test_roc_auc_perdif'].max()

# %%
results_max = results_max.loc[results_max['technique']!='Over']
results_max.loc[results_max['technique']=='Under', 'technique'] = 'RUS'
results_max.loc[results_max['technique']=='Bordersmote', 'technique'] = 'BorderlineSMOTE'
results_max.loc[results_max['technique']=='Smote', 'technique'] = 'SMOTE'
#results_max.loc[results_max['technique']=='privateSMOTE', 'technique'] = 'privateSMOTE A'
#results_max.loc[results_max['technique']=='privateSMOTE \n regardless of \n the class', 'technique'] = 'privateSMOTE B'

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
# figure = ax.get_figure()
# figure.savefig(f'{os.path.dirname(os.getcwd())}/output/plots/alltechniques_outofsample_auc.pdf', bbox_inches='tight')

# %%
