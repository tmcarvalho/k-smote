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
def join_allresults(candidate_folder, technique):
    concat_results_test = []
    c=0
    _,_,candidate_result_files = next(walk(f'{candidate_folder}/test/'))
    for transf_file in candidate_result_files:
        try: # pass error in roc auc for some transf files (especially with geep learning)
            candidate_result_cv_train = pd.read_csv(f'{candidate_folder}/validation/{transf_file}')
            candidate_result_test = pd.read_csv(f'{candidate_folder}/test/{transf_file}')
            # select the best model in CV
            best_cv_roc = candidate_result_cv_train.iloc[[candidate_result_cv_train['mean_test_roc_auc_curve'].idxmax()]]
            best_cv_f1 = candidate_result_cv_train.iloc[[candidate_result_cv_train['mean_test_f1_weighted'].idxmax()]]
            best_cv_gmean = candidate_result_cv_train.iloc[[candidate_result_cv_train['mean_test_gmean'].idxmax()]]
            best_cv_acc = candidate_result_cv_train.iloc[[candidate_result_cv_train['mean_test_acc'].idxmax()]]
            # use the best model in CV to get the results of it in out of sample
            best_test_roc = candidate_result_test.iloc[best_cv_roc.index,:]
            best_cv_f1 = candidate_result_test.iloc[best_cv_f1.index,:]
            best_cv_gmean = candidate_result_test.iloc[best_cv_gmean.index,:]
            best_cv_acc = candidate_result_test.iloc[best_cv_acc.index,:]

            # save oracle results in out of sample
            oracle_candidate = candidate_result_test.loc[candidate_result_test['test_roc_auc'].idxmax(),'test_roc_auc']
            oracle_candidate_fscore = candidate_result_test.loc[candidate_result_test['test_f1_weighted'].idxmax(),'test_f1_weighted']
            oracle_candidate_gmean = candidate_result_test.loc[candidate_result_test['test_gmean'].idxmax(),'test_gmean']
            oracle_candidate_acc = candidate_result_test.loc[candidate_result_test['test_accuracy'].idxmax(),'test_accuracy']
            
            best_test_roc.loc[:,'best_mean_f1_weighted'] = best_cv_f1.test_f1_weighted
            best_test_roc.loc[:,'best_mean_gmean'] = best_cv_f1.test_gmean
            best_test_roc.loc[:,'best_mean_acc'] = best_cv_f1.test_accuracy
            best_test_roc.loc[:,'test_roc_auc_oracle'] = oracle_candidate
            best_test_roc.loc[:,'test_fscore_oracle'] = oracle_candidate_fscore
            best_test_roc.loc[:,'test_gmean_oracle'] = oracle_candidate_gmean
            best_test_roc.loc[:,'test_accuracy_oracle'] = oracle_candidate_acc

            
            # get technique
            if technique=='resampling':
                best_test_roc.loc[:, 'technique'] = transf_file.split('_')[1].title()
            elif technique=='deep_learning':
                best_test_roc.loc[:, 'technique'] = transf_file.split('_')[1]
            elif technique=='city':
                best_test_roc.loc[:, 'technique'] = transf_file.split('_')[1].upper()
            else:
                best_test_roc.loc[:, 'technique'] = technique

            # get dataset number
            best_test_roc['ds'] = transf_file.split('_')[0]
            best_test_roc['ds_complete'] = transf_file

            # concat each test result
            if c == 0:
                concat_results_test = best_test_roc
                c += 1
            else:
                concat_results_test = pd.concat([concat_results_test, best_test_roc])
        except: pass
    return concat_results_test

# %% 
# path to predictive results
orig = join_allresults('../output/modeling/original/', 'original')
# %% PPT
ppt = join_allresults('../output/modeling/PPT_ARX/', 'PPT')
# %% resampling
resampling = join_allresults('../output/modeling/re-sampling/', 'resampling')

# %% deep learning
deeplearn = join_allresults('../output/modeling/deep_learning/', 'deep_learning')

# %% synthcity
city = join_allresults('../output/modeling/city/', 'city')

# %% 2-PrivateSMOTE
privatesmote = join_allresults('../output/modeling/PrivateSMOTE/', 'PrivateSMOTE')

# %% 3-PrivateSMOTE
privatesmote3 = join_allresults('../output/modeling/PrivateSMOTE3/', 'PrivateSMOTE3')

# %% 5-PrivateSMOTE
privatesmote5 = join_allresults('../output/modeling/PrivateSMOTE5/', 'PrivateSMOTE5')

# %% concat all data sets
results = pd.concat([orig, ppt, privatesmote, deeplearn, resampling, city,
                     privatesmote3, privatesmote5]).reset_index(drop=True)
# %%
results.loc[results['technique']=='Under', 'technique'] = 'RUS'
results.loc[results['technique']=='Bordersmote', 'technique'] = 'BorderlineSMOTE'
results.loc[results['technique']=='Smote', 'technique'] = 'SMOTE'
results.loc[results['technique']=='CopulaGAN', 'technique'] = 'Copula GAN'
results.loc[results['technique']=='PrivateSMOTE', 'technique'] = r'$\epsilon$-2PrivateSMOTE'
results.loc[results['technique']=='PrivateSMOTE3', 'technique'] = r'$\epsilon$-3PrivateSMOTE'
results.loc[results['technique']=='PrivateSMOTE5', 'technique'] = r'$\epsilon$-5PrivateSMOTE'
# %% remove wrong results (dpgan in deep learning folders) 
results = results.loc[results.technique != 'dpgan'].reset_index(drop=True)
# %% prepare to calculate percentage difference
original_results = results.loc[results['technique']=='original'].reset_index(drop=True)
results = results.loc[results['technique'] != 'original'].reset_index(drop=True)
# %% match ds name with transformed files
original_results['ds'] = original_results['ds'].apply(lambda x: f'ds{x.split(".")[0]}')

# %% percentage difference
results['roc_auc_perdif'] = np.NaN
results['fscore_perdif'] = np.NaN
results['gmean_perdif'] = np.NaN
results['acc_perdif'] = np.NaN
for idx in results.index:
    orig_file = original_results.loc[(original_results.ds == results.ds[idx])].reset_index(drop=True)

    # calculate the percentage difference
    # 100 * (Sc - Sb) / Sb
    results['roc_auc_perdif'][idx] = ((results['test_roc_auc'][idx] - orig_file['test_roc_auc'].iloc[0]) / orig_file['test_roc_auc'].iloc[0]) * 100
    results['fscore_perdif'][idx] = ((results['test_f1_weighted'][idx] - orig_file['test_f1_weighted'].iloc[0]) / orig_file['test_f1_weighted'].iloc[0]) * 100
    results['gmean_perdif'][idx] = ((results['test_gmean'][idx] - orig_file['test_gmean'].iloc[0]) / orig_file['test_gmean'].iloc[0]) * 100
    results['acc_perdif'][idx] = ((results['test_accuracy'][idx] - orig_file['test_accuracy'].iloc[0]) / orig_file['test_accuracy'].iloc[0]) * 100

# %%
# results.to_csv('../output_analysis/modeling_results.csv', index=False)
# %%
# results = pd.read_csv('../output_analysis/modeling_results.csv')

# %%
PROPS = {
    'boxprops':{'facecolor':'#00BFC4', 'edgecolor':'black'},
    'medianprops':{'color':'black'},
    'whiskerprops':{'color':'black'},
    'capprops':{'color':'black'}
}
order = ['PPT', 'RUS', 'SMOTE', 'BorderlineSMOTE', 'Copula GAN', 'TVAE', 'CTGAN', 'DPGAN', 'PATEGAN',
         r'$\epsilon$-2PrivateSMOTE', r'$\epsilon$-3PrivateSMOTE', r'$\epsilon$-5PrivateSMOTE']
# %%
sns.set_style("darkgrid")
plt.figure(figsize=(12,8))
ax = sns.boxplot(data=results, x='technique', y='roc_auc_perdif', **PROPS, order=order)
sns.set(font_scale=1.5)
plt.xticks(rotation=45)
plt.xlabel("")
plt.ylabel("Percentage difference of predictive performance (AUC)")
plt.autoscale(True)

# %%
results_max = results.loc[results.groupby(['ds', 'technique'])['roc_auc_perdif'].idxmax()]
sns.set_style("darkgrid")
plt.figure(figsize=(12,8))
ax = sns.boxplot(data=results_max, x='technique', y='roc_auc_perdif', **PROPS, order=order)
sns.set(font_scale=1.5)
plt.xticks(rotation=45)
plt.xlabel("")
plt.ylabel("Percentage difference of \npredictive performance (ROC AUC)")
# plt.savefig(f'{os.path.dirname(os.getcwd())}/plots/performance_rocauc.pdf', bbox_inches='tight')

# %%
privsmote = results.loc[results.technique.str.contains('PrivateSMOTE')].reset_index(drop=True)
privsmote['epsilon'] = np.nan
privsmote['kanon'] = np.nan
for idx, file in enumerate(privsmote.ds_complete):
    if 'privateSMOTE' in file:
        privsmote['epsilon'][idx] = str(list(map(float, re.findall(r'\d+\.\d+', file.split('_')[1])))[0])
        if len(file.split('_')[1].split('-')[1])>12:
            privsmote['kanon'][idx] = int(list(map(float, re.findall(r'\d+', file.split('_')[1].split('-')[1])))[0])
        else:
            privsmote['kanon'][idx] = int(2)

# %%
privsmote['kanon'] = privsmote['kanon'].astype(float)
privsmote['epsilon'] = privsmote['epsilon'].astype(float)
# %%
privsmote_max = privsmote.loc[privsmote.groupby(['ds', 'epsilon', 'kanon'])['roc_auc_perdif'].idxmax()]

ep_order = ['0.1', '0.5', '1.0', '5.0', '10.0']
sns.set_style("darkgrid")
plt.figure(figsize=(9,8))
ax = sns.boxplot(data=privsmote_max, x='epsilon', y='roc_auc_perdif', hue='kanon',
                 order=ep_order, palette='Set2')
sns.set(font_scale=1.5)
plt.xticks(rotation=45)
plt.xlabel("")
plt.ylabel("Percentage difference of \npredictive performance (ROC AUC)")
ax.legend(loc='center right', bbox_to_anchor=(1.2, 0.5), title='Kanon')
# plt.savefig(f'{os.path.dirname(os.getcwd())}/plots/privateSMOTE_epsilons.pdf', bbox_inches='tight')


# %%
fig = plt.figure()
ax = plt.axes(projection='3d')
# plotting
ax.scatter(privsmote_max['epsilon'], privsmote_max['kanon'],privsmote_max['fscore_perdif'])
#ax.set_xticks([0.1,0.5,1.0,5.0,10.0])
#ax.set_yticks([2,3,5])
plt.show()
# %%

