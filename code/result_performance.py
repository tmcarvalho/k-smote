"""Re-identification of single outs
This script will analyse the re-identification risk with
record linkage for single outs.
"""
# %%
import os
from os import walk
import re
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

# %%
def transform_npy_to_csv(original_foler, folder_validation):
    """Transform npy to csv with percentage difference for eeach metric.

    Args:
        original_foler (string): path to original data folder
        folder_validation (string): path to validatin results
    """
    _, _, performance_files = next(walk(f'{folder_validation}/'))
    performance_files.sort()
    _, _, orig_file_validation = next(walk(f'{original_foler}/'))
    orig_file_validation.sort()
    metrics = ['gmean', 'acc', 'bal_acc', 'f1', 'f1_weighted', 'roc_auc_curve']

    for org in orig_file_validation:
        if '.npy' in org:
            org_result = np.load(f'{original_foler}/{org}', allow_pickle='TRUE').item()
            org_result_rf = pd.DataFrame.from_dict(org_result['cv_results_Random Forest'])
            org_result_xgb = pd.DataFrame.from_dict(org_result['cv_results_Boosting'])
            org_result_logr = pd.DataFrame.from_dict(org_result['cv_results_Logistic Regression'])

            for file in performance_files:
                if '.npy' in file and str(org.split('.')[0] + '_') in file:
                    result = np.load(f'{folder_validation}/{file}', allow_pickle='TRUE').item()
                    result = pd.DataFrame.from_dict(result)

                    # for each CV calculate the percentage difference
                    for metric in metrics:
                        # 100 * (Sc - Sb) / Sb
                        max_rf = org_result_rf['mean_test_' + metric].max()
                        max_xgb = org_result_xgb['mean_test_' + metric].max()
                        max_logr = org_result_logr['mean_test_' + metric].max()

                        result['mean_test_' + metric + '_perdif'] = np.where(result['param_classifier'].str.contains("RandomForestClassifier"), 100 * (result['mean_test_' + metric] - max_rf) / max_rf, None)
                        result['mean_test_' + metric + '_perdif'] = np.where(result['param_classifier'].str.contains("XGBClassifier"), 100 * (result['mean_test_' + metric] - max_xgb) / max_xgb, None)
                        result['mean_test_' + metric + '_perdif'] = np.where(result['param_classifier'].str.contains("LogisticRegression"), 100 * (result['mean_test_' + metric] - max_logr) / max_logr, None)

                    # add attribute with model's name
                    result['model'] = np.where(result['param_classifier'].str.contains("RandomForestClassifier"), 'Random Forest', result['param_classifier'])
                    result['model'] = np.where(result['param_classifier'].str.contains("XGBClassifier"), 'XGBoost', result['param_classifier'])
                    result['model'] = np.where(result['param_classifier'].str.contains("LogisticRegression"), 'Logistic Regression', result['param_classifier'])

                    # save csv with the 5 algorithms
                    result.to_csv(f'{folder_validation}/{file.replace(".npy", ".csv")}', index=False)

# %%
original_folder = f'{os.path.dirname(os.getcwd())}/output/modeling/original/validation'
transform_npy_to_csv(original_folder, f'{os.path.dirname(os.getcwd())}/output/modeling/PPT/validation')
transform_npy_to_csv(original_folder, f'{os.path.dirname(os.getcwd())}/output/modeling/smote_under_over/validation')
transform_npy_to_csv(original_folder, f'{os.path.dirname(os.getcwd())}/output/modeling/smote_singleouts/validation')
transform_npy_to_csv(original_folder, f'{os.path.dirname(os.getcwd())}/output/modeling/smote_singleouts_scratch/validation')

# %%

def concat_ppt_results():
    """Join all csv results for privacy preserving techniques

    Returns:
        pd.Dataframe: joined results
    """
    folder_validation = f'{os.path.dirname(os.getcwd())}/output/modeling/PPT/validation'
    results = []
    _, _, performance_files = next(walk(f'{folder_validation}/'))
    performance_files.sort()
    performance_files = [file for file in performance_files if '.csv' in file]

    for _, file in enumerate(performance_files):
        f = list(map(int, re.findall(r'\d+', file.split('_')[0])))[0]
        if f in [2,4,5,8,10,14,16,32]:
            if '.csv' in file:
                result = pd.read_csv(f'{folder_validation}/{file}')
                quasi_id_int = file.split('_')[1]
                quasi_id_int = list(map(int, re.findall(r'\d+', quasi_id_int)))[0]
                quasi_id = f'Set of quasi-identifiers #{quasi_id_int+1}'
                result['qi'] = quasi_id
                result['ds'] = file.split('_')[0]
                result['technique'] = 'PPT'
                result['knn'] = None
                result['per'] = None

                results.append(result)

    concat_results = pd.concat(results)

    return concat_results


def concat_smote_under_over_results():
    """Join all csv results for smote, under and over

    Returns:
        pd.Dataframe: joined results
    """
    folder_validation = f'{os.path.dirname(os.getcwd())}/output/modeling/smote_under_over/validation'
    results = []
    _, _, performance_files = next(walk(f'{folder_validation}/'))
    performance_files.sort()
    performance_files = [file for file in performance_files if '.csv' in file]

    for _, file in enumerate(performance_files):
        f = list(map(int, re.findall(r'\d+', file.split('_')[0])))[0]
        if f in [2,4,5,8,10,14,16,32]:
            if '.csv' in file:
                result = pd.read_csv(f'{folder_validation}/{file}')
                per = file.split('_')[2]
                per = per.replace(".csv", "")
                per = list(map(float, re.findall(r'\d+\.\d+', per)))[0] if '.' in per else list(map(float, re.findall(r'\d', per)))[0]
                result['technique'] = file.split('_')[1]
                if file.split('_')[1] == 'smote':
                    result['knn'] = list(map(int, re.findall(r'\d+', file.split('_')[2])))[0]
                result['per'] = per
                result['ds'] = file.split('_')[0]
                result['qi'] = None

                results.append(result)

    concat_results = pd.concat(results)

    return concat_results


def concat_results_smote_singleouts(folder_validation, technique):
    """Join all csv results for smote with single outs

    Args:
        technique (string): distinction between smote techniques

    Returns:
        pd.Dataframe: joined results
    """
    results = []
    _, _, performance_files = next(walk(f'{folder_validation}/'))
    performance_files.sort()
    performance_files = [file for file in performance_files if '.csv' in file]

    for _, file in enumerate(performance_files):
        f = list(map(int, re.findall(r'\d+', file.split('_')[0])))[0]
        if f in [2,4,5,8,10,14,16,32]:
            if '.csv' in file:
                result = pd.read_csv(f'{folder_validation}/{file}')
                quasi_id_int = file.split('_')[2]
                quasi_id_int = list(map(int, re.findall(r'\d+', quasi_id_int)))[0]
                quasi_id = f'Set of quasi-identifiers #{quasi_id_int+1}'
                k_nn = file.split('_')[3]
                k_nn = list(map(int, re.findall(r'\d+', k_nn)))[0]
                per = file.split('_')[4]
                per = per.replace(".csv", "")
                per = per = list(map(float, re.findall(r'\d+\.\d+', per)))[0] if '.' in per else list(map(float, re.findall(r'\d', per)))[0]
                result['qi'] = quasi_id
                result['knn'] = k_nn
                result['per'] = per
                result['ds'] = file.split('_')[0]
                result['technique'] = technique

                results.append(result)

    concat_results = pd.concat(results)

    return concat_results

# %%
ppt_results = concat_ppt_results()
smote_under_over_results = concat_smote_under_over_results()
smote_singleouts_oneclass = concat_results_smote_singleouts(
    f'{os.path.dirname(os.getcwd())}/output/modeling/smote_singleouts/validation',
    'Smote one class')
smote_singleouts_twoclasses = concat_results_smote_singleouts(
    f'{os.path.dirname(os.getcwd())}/output/modeling/smote_singleouts_scratch/validation',
    'Smote two classes')

# %%
all_results = pd.concat([ppt_results, smote_under_over_results, smote_singleouts_oneclass, smote_singleouts_twoclasses])

# %%
results_max = all_results.groupby(['ds', 'technique'], as_index=False)['mean_test_f1_weighted', 'mean_test_f1_weighted_perdif', 'mean_test_gmean_perdif', 'mean_test_roc_auc_curve_perdif'].max()
# %%
plt.figure(figsize=(12,10))
ax = sns.boxplot(data=results_max, x='technique', y='mean_test_f1_weighted_perdif', palette='muted')
# ax.set(ylim=(0, 30))
sns.set(font_scale=1.5)
plt.xticks(rotation=30)
plt.xlabel("")
plt.ylabel("Percentage difference of predictive performance (F-score)")
plt.show()
figure = ax.get_figure()
figure.savefig(f'{os.path.dirname(os.getcwd())}/output/plots/8results_technique_testindexes.png', bbox_inches='tight')

# %%
melted_results = results_max.melt(id_vars=["mean_test_f1_weighted_perdif", "mean_test_gmean_perdif", "mean_test_roc_auc_curve_perdif"], 
        var_name="Measure", 
        value_name="Value")
# %% ###########################################################
grp_dataset = all_results.groupby('ds')
for name, grp in grp_dataset:
    # print(name)
    # print(grp['qi'].nunique())
    sns.set_style('darkgrid')
    sns.set(font_scale=1.5)
    g = sns.FacetGrid(grp,row='qi', height=8, aspect=1.5)
    g.map(sns.boxplot, "knn", "mean_test_f1_weighted",
    "per", palette='muted').add_legend(title='Ratio \n sampling')
    g.set_axis_labels("Nearest Neighbours", "Fscore")
    g.set_titles("{row_name}")
    g.savefig(
        f'{os.path.dirname(os.getcwd())}/output/plots/each/fscore/{name}_fscore.pdf',
        bbox_inches='tight')
# %%
for name, grp in grp_dataset:
    sns.set_style('darkgrid')
    sns.set(font_scale=1.5)
    gg = sns.FacetGrid(grp,row='qi', height=8, aspect=1.5)
    gg.map(sns.barplot, "knn", "privacy_risk_50", "per",
    palette='muted').add_legend(title='Ratio \n sampling')
    gg.set_axis_labels("Nearest Neighbours", "Privacy Risk")
    gg.set_titles("{row_name}")
    gg.savefig(
    f'{os.path.dirname(os.getcwd())}/output/plots/each/risk_at50%/{name}_risk.pdf',
    bbox_inches='tight')

# %%
mean_ds_fscore = all_results.groupby(
    ['ds', 'model', 'knn', 'per'])['mean_test_f1_weighted'].mean().reset_index(name='mean')
mean_ds_fscore['rank'] = mean_ds_fscore.groupby(['ds'])['mean'].transform('rank', method='dense')
# %%
plt.figure(figsize=(10,7))
ax = sns.boxplot(data=mean_ds_fscore, x='model', y='rank', palette='muted', width=0.5)
ax.set(ylim=(0, 140))
sns.set(font_scale=2)
plt.xticks(rotation=45)
plt.xlabel("")
plt.ylabel("Rank of Predictive Performance (Fscore)")
plt.show()
figure = ax.get_figure()
figure.savefig(f'{os.path.dirname(os.getcwd())}/output/plots/fscore_rank.pdf', bbox_inches='tight')

# %%
mean_ds_knn = all_results.groupby(
    ['ds', 'knn', 'per'])['mean_test_f1_weighted'].mean().reset_index(name='mean')
mean_ds_knn['rank'] = mean_ds_knn.groupby(['ds'])['mean'].transform('rank' , method='dense')
# %%
plt.figure(figsize=(18,10))
ax = sns.boxplot(data=mean_ds_knn, x='knn', y='rank', palette='muted', hue='per')
ax.legend(loc='lower center', bbox_to_anchor=(0.5, 1), ncol=5, title='Ratio sampling')
# ax.set(ylim=(0, 30))
sns.set(font_scale=2)
plt.xlabel("Nearest Neighbours")
plt.ylabel("Rank of Predictive Performance (Fscore)")
plt.show()
figure = ax.get_figure()
figure.savefig(f'{os.path.dirname(os.getcwd())}/output/plots/knn_rank.pdf', bbox_inches='tight')


# %%
val = np.load('../output/modeling/PPT/validation/ds8_transf2_qi4.npy', allow_pickle=True).item()
# %%
val_df = pd.DataFrame.from_dict(val)
# %%
val_df
# %%
original_folder = f'{os.path.dirname(os.getcwd())}/original/'
_, _, orig_file_validation = next(walk(f'{original_folder}/'))
# %%
ncat = []
nnum = []
nsize = []
nfeat = []
c=0
for file in orig_file_validation:
    if int(file.split('.csv')[0]) not in [0,1,3,13,23,28,34,36,40,48,54,66,87]:
        c=c+1
        df = pd.read_csv(f'{original_folder}/{file}')
        nfeat.append(df.shape[1])
        nsize.append(df.shape[0])
        nnum.append(len(list(df.select_dtypes(include=[np.number]).columns)))
        ncat.append(len(list(df.select_dtypes(exclude=[np.number]).columns)))

# %%
