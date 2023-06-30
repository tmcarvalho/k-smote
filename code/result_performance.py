"""Predictive performance analysis
This script will analyse the predictive performance in the out-of-sample.
"""
# %%
from os import walk
import re
import pandas as pd
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
        baseline_result_test = pd.read_csv(f'{baseline_folder}/test/{baseline}')

        # guaranteeing that we have the best model in the grid search instead of 3 models
        best_baseline_cv = baseline_result_cv_train.loc[baseline_result_cv_train['rank_test_roc_auc_curve'] == 1,:].reset_index(drop=True)
        baseline_result_best = baseline_result_test.loc[baseline_result_test.model == best_baseline_cv.model[0],:]

        b = list(map(int, re.findall(r'\d+', baseline.split('.')[0])))[0]
        transf_files = [fl for fl in candidate_result_files if list(map(int, re.findall(r'\d+', fl.split('_')[0])))[0] == b]

        for transf_file in transf_files:
            candidate_result_cv_train = pd.read_csv(f'{candidate_folder}/validation/{transf_file}')
            candidate_result_test = pd.read_csv(f'{candidate_folder}/test/{transf_file}')
            
            # calculate the percentage difference
            # 100 * (Sc - Sb) / Sb
            best_candidate_cv = candidate_result_cv_train.loc[candidate_result_cv_train['rank_test_roc_auc_curve'] == 1,:]
            candidate_result_best = candidate_result_test.iloc[best_candidate_cv.index,:]
            candidate_result_best['test_roc_auc_perdif'] = 100 * (candidate_result_best['test_roc_auc'].values[0] - baseline_result_best['test_roc_auc'].values[0]) / baseline_result_best['test_roc_auc'].values[0]
            candidate_result_best['test_fscore_perdif'] = 100 * (candidate_result_best['test_f1_weighted'].values[0] - baseline_result_best['test_f1_weighted'].values[0]) / baseline_result_best['test_f1_weighted'].values[0]

            # candidate_result_cv['test_fscore_perdif'] = 100 * (candidate_result_cv['test_f1_weighted'] - baseline_result_cv['test_f1_weighted']) / baseline_result_cv['test_f1_weighted']
            # candidate_result_cv['test_gmean_perdif'] = 100 * (candidate_result_cv['test_gmean'] - baseline_result_cv['test_gmean']) / baseline_result_cv['test_gmean']

            oracle_candidate = candidate_result_test.loc[candidate_result_test['test_roc_auc'].idxmax(),'test_roc_auc']
            oracle_candidate_fscore = candidate_result_test.loc[candidate_result_test['test_f1_weighted'].idxmax(),'test_f1_weighted']
            candidate_result_best['test_roc_auc_oracle'] = oracle_candidate
            candidate_result_best['test_fscore_oracle'] = oracle_candidate_fscore
            candidate_result_best['test_roc_auc_candidate'] = candidate_result_best['test_roc_auc'].values[0]
            candidate_result_best['test_fscore_candidate'] = candidate_result_best['test_f1_weighted'].values[0]

            # get technique
            if technique=='resampling':
                candidate_result_best.loc[:, 'technique'] = transf_file.split('_')[1].title()
            elif technique=='deep_learning':
                candidate_result_best.loc[:, 'technique'] = transf_file.split('_')[1]
            else:
                candidate_result_best.loc[:, 'technique'] = technique
                
            
            # get dataset number
            candidate_result_best['ds'] = transf_file.split('_')[0]
            candidate_result_best['ds_complete'] = transf_file

            # concat each test result
            if c == 0:
                df_concat_cv = candidate_result_best
                c += 1
            else:     
                df_concat_cv = pd.concat([df_concat_cv, candidate_result_best])

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

# %% PrivateSMOTE
privatesmote_folder = '../output/modeling/PrivateSMOTE_k3/'
baseorg_privatesmote_scratch = percentage_difference_org(orig_folder, privatesmote_folder, 'privateSMOTE')

# %% PrivateSMOTE outliers
privatesmote_out_folder = '../output/modeling/PrivateSMOTE_k3out/'
baseorg_privatesmote_out_scratch = percentage_difference_org(orig_folder, privatesmote_out_folder, 'privateSMOTE_out')

# %% PrivateSMOTE Force
privatesmote_force_folder = '../output/modeling/PrivateSMOTE_force_k3/'
baseorg_privatesmote_force = percentage_difference_org(orig_folder, privatesmote_force_folder, 'privateSMOTE_force')

# %% Ep-PrivateSMOTE
privatesmote_laplace_folder = '../output/modeling/PrivateSMOTE_laplace_k3/'
baseorg_privatesmote_laplace = percentage_difference_org(orig_folder, privatesmote_laplace_folder, 'privateSMOTE_laplace')

# %% Ep-PrivateSMOTE Force
privatesmote_force_laplace_folder = '../output/modeling/PrivateSMOTE_force_laplace_k3/'
baseorg_privatesmote_force_laplace = percentage_difference_org(orig_folder, privatesmote_force_laplace_folder, 'privateSMOTE_force_laplace')

# %% Ep-PrivateSMOTE Force outliers
privatesmote_force_laplace_out_folder = '../output/modeling/PrivateSMOTE_force_laplace_k3out/'
baseorg_privatesmote_force_out_laplace = percentage_difference_org(orig_folder, privatesmote_force_laplace_out_folder, 'privateSMOTE_force_laplace_out')

# %% DPART independent
dpart_folder = '../output/modeling/dpart_independent/'
baseorg_dpart = percentage_difference_org(orig_folder, dpart_folder, 'dpart_independent')

# %% DPART synthpop
dpartsynt_folder = '../output/modeling/dpart_synthpop/'
baseorg_dpart_synt = percentage_difference_org(orig_folder, dpartsynt_folder, 'dpart_synthpop')

# %% concat all data sets
results_baseorg_cv = pd.concat([baseorg_ppt, baseorg_resampling, baseorg_deeplearn, baseorg_privatesmote_scratch, baseorg_privatesmote_out_scratch,
                                baseorg_privatesmote_force, baseorg_privatesmote_laplace, baseorg_privatesmote_force_laplace,
                                baseorg_privatesmote_force_out_laplace, baseorg_dpart, baseorg_dpart_synt])
# %%
results_baseorg_cv = results_baseorg_cv.reset_index(drop=True)
# %%
results_baseorg_cv = results_baseorg_cv.loc[results_baseorg_cv['technique']!='Over']
results_baseorg_cv.loc[results_baseorg_cv['technique']=='Under', 'technique'] = 'RUS'
results_baseorg_cv.loc[results_baseorg_cv['technique']=='Bordersmote', 'technique'] = 'BorderlineSMOTE'
results_baseorg_cv.loc[results_baseorg_cv['technique']=='Smote', 'technique'] = 'SMOTE'
results_baseorg_cv.loc[results_baseorg_cv['technique']=='copulaGAN', 'technique'] = 'Copula GAN'
results_baseorg_cv.loc[results_baseorg_cv['technique']=='privateSMOTE', 'technique'] = 'PrivateSMOTE'
results_baseorg_cv.loc[results_baseorg_cv['technique']=='privateSMOTE_out', 'technique'] = 'PrivateSMOTE *'
results_baseorg_cv.loc[results_baseorg_cv['technique']=='privateSMOTE_force', 'technique'] = 'PrivateSMOTE Force'
results_baseorg_cv.loc[results_baseorg_cv['technique']=='privateSMOTE_laplace', 'technique'] = r'$\epsilon$-PrivateSMOTE'
results_baseorg_cv.loc[results_baseorg_cv['technique']=='privateSMOTE_force_laplace', 'technique'] =  r'$\epsilon$-PrivateSMOTE Force'
results_baseorg_cv.loc[results_baseorg_cv['technique']=='privateSMOTE_force_laplace_out', 'technique'] =  r'$\epsilon$-PrivateSMOTE Force *'
results_baseorg_cv.loc[results_baseorg_cv['technique']=='dpart_independent', 'technique'] = 'Independent'
results_baseorg_cv.loc[results_baseorg_cv['technique']=='dpart_synthpop', 'technique'] =  'Synthpop'

# %%
# results_baseorg_cv.to_csv('../output/test_cv_roc_auc_newprivatesmote_k3.csv', index=False)
# %%
# results_baseorg_cv = pd.read_csv('../output/test_cv_roc_auc_newprivatesmote.csv')
# %%
results_max = results_baseorg_cv.loc[results_baseorg_cv.groupby(['ds', 'technique'])['test_roc_auc_perdif'].idxmax()]

# %%
PROPS = {
    'boxprops':{'facecolor':'#00BFC4', 'edgecolor':'black'},
    'medianprops':{'color':'black'},
    'whiskerprops':{'color':'black'},
    'capprops':{'color':'black'}
}
order = ['PPT', 'RUS', 'SMOTE', 'BorderlineSMOTE', 'Copula GAN', 'TVAE', 'CTGAN', 'Independent', 'Synthpop', 'PrivateSMOTE', 'PrivateSMOTE Force', r'$\epsilon$-PrivateSMOTE', r'$\epsilon$-PrivateSMOTE Force']
# %%
sns.set_style("darkgrid")
plt.figure(figsize=(12,8))
ax = sns.boxplot(data=results_max, x='technique', y='test_roc_auc_perdif', **PROPS, order=order)
sns.set(font_scale=1.5)
plt.xticks(rotation=45)
plt.xlabel("")
plt.ylabel("Percentage difference of predictive performance (AUC)")
plt.autoscale(True)
plt.show()
# figure = ax.get_figure()
# figure.savefig(f'{os.path.dirname(os.getcwd())}/output/plots/performance_all.pdf', bbox_inches='tight')

# %%
privsmote = results_max.loc[results_max.technique.str.contains('PrivateSMOTE')]
sns.set_style("darkgrid")
plt.figure(figsize=(8,6))
ax = sns.boxplot(data=privsmote, x='technique', y='test_roc_auc_perdif',order=order, **PROPS)
sns.set(font_scale=1.1)
plt.xticks(rotation=45)
plt.xlabel("")
plt.ylabel("Percentage difference of predictive performance (AUC)")
#plt.yscale('symlog')
plt.autoscale(True)
plt.show()
#figure = ax.get_figure()
#figure.savefig(f'{os.path.dirname(os.getcwd())}/output/plots/performance_privateSMOTE.pdf', bbox_inches='tight')

# %%
