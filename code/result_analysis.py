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
def transform_npy_to_csv():
    """_summary_
    """
    folder_validation = f'{os.path.dirname(os.getcwd())}/output/modeling/oversampled/validation'
    _, folders_performance, _ = next(walk(folder_validation))

    for folder in folders_performance:
        _, _, performance_files = next(walk(f'{folder_validation}/{folder}/'))
        performance_files.sort()
        for file in performance_files:
            if '.npy' in file:
                result = np.load(f'{folder_validation}/{folder}/{file}', allow_pickle='TRUE').item()

                result_rf = pd.DataFrame.from_dict(
                    {key:pd.Series(value) for key, value in result['cv_results_Random Forest'].items()})
                result_bag = pd.DataFrame.from_dict(
                    {key:pd.Series(value) for key, value in result['cv_results_Bagging'].items()})
                result_xgb = pd.DataFrame.from_dict(
                    {key:pd.Series(value) for key, value in result['cv_results_Boosting'].items()})
                result_logr = pd.DataFrame.from_dict(
                    {key:pd.Series(value) for key,
                    value in result['cv_results_Logistic Regression'].items()})
                result_nn = pd.DataFrame.from_dict(
                    {key:pd.Series(value) for key,
                    value in result['cv_results_Neural Network'].items()})

                # add attribute with model's name
                result_rf['model'] = 'Random Forest'
                result_bag['model'] = 'Bagging'
                result_xgb['model'] = 'XGBoost'
                result_logr['model'] = 'Logistic Regression'
                result_nn['model'] = 'Neural Network'

                # save csv with the 5 algorithms
                res = pd.concat([result_rf, result_bag, result_xgb, result_logr, result_nn])
                res.to_csv(f'{folder_validation}/{folder}/{file.replace(".npy", ".csv")}', index=False)

# %%

def concat_all_results():
    """_summary_

    Returns:
        _type_: _description_
    """
    folder_validation = f'{os.path.dirname(os.getcwd())}/output/modeling/oversampled/validation'
    _, folders_performance, _ = next(walk(folder_validation))
    results = []
    for folder in folders_performance:
        _, _, performance_files = next(walk(f'{folder_validation}/{folder}/'))
        performance_files.sort()
        performance_files = [file for file in performance_files if '.csv' in file]

        for _, file in enumerate(performance_files):
            result = pd.read_csv(f'{folder_validation}/{folder}/{file}')
            quasi_id_int = file.split('_')[1]
            quasi_id_int = list(map(int, re.findall(r'\d+', quasi_id_int)))[0]
            quasi_id = f'Set of quasi-identifiers #{quasi_id_int+1}'
            k_nn = file.split('_')[2]
            k_nn = list(map(int, re.findall(r'\d+', k_nn)))[0]
            per = file.split('_')[3]
            per = per.replace(".npy", "")
            per = list(map(float, re.findall(r'\d+\.\d+', per)))[0]
            result['qi'] = quasi_id
            result['knn'] = k_nn
            result['per'] = per
            result['privacy_risk_50'] = result.groupby(
                ['knn','per'], sort=False)['privacy_risk_50'].apply(lambda x: x.ffill().bfill())
            result['privacy_risk_75'] = result.groupby(
                ['knn','per'], sort=False)['privacy_risk_75'].apply(lambda x: x.ffill().bfill())
            result['privacy_risk_100'] = result.groupby(
                ['knn','per'], sort=False)['privacy_risk_100'].apply(lambda x: x.ffill().bfill())
            result['ds'] = folder

            results.append(result)

    concat_results = pd.concat(results)

    return concat_results

# %%
transform_npy_to_csv()
# %%
all_results = concat_all_results()
# %%
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
risk50 = all_results.loc[:, ['ds', 'knn', 'per', 'privacy_risk_50']]
risk75 = all_results.loc[:, ['ds', 'knn', 'per', 'privacy_risk_75']]
risk100 = all_results.loc[:, ['ds', 'knn', 'per', 'privacy_risk_100']]
# %%
privacy_risk = pd.concat([risk50, risk75, risk100])
# %%
privacy_risk['privacy_thr'] = None
privacy_risk['privacy_thr'] = np.where(
    ~(privacy_risk['privacy_risk_50'].isna()), 'Threshold at 50%', privacy_risk['privacy_thr'])
privacy_risk['privacy_thr'] = np.where(
    ~(privacy_risk['privacy_risk_75'].isna()), 'Threshold at 75%', privacy_risk['privacy_thr'])
privacy_risk['privacy_thr'] = np.where(
    ~(privacy_risk['privacy_risk_100'].isna()), 'Threshold at 100%', privacy_risk['privacy_thr'])

# %%
privacy_risk['privacy'] = privacy_risk.loc[
    :,['privacy_risk_50', 'privacy_risk_75', 'privacy_risk_100']].apply(
    lambda x: ''.join(x.dropna().astype('float64').astype('str')), 1)
# %%
privacy_risk['privacy'] = privacy_risk['privacy'].astype('float64')
# %%
privacy_risk['rank'] = privacy_risk.groupby(
    ['ds', 'privacy_thr'])['privacy'].transform('rank', method='dense')
# %%
sns.set_style('darkgrid')
g = sns.FacetGrid(privacy_risk,row='privacy_thr', height=8, aspect=1.5)
g.map(sns.boxplot, "knn", "rank",
"per", palette='muted').add_legend(title='Ratio \n sampling')
# g.set(yscale='log')
g.set_axis_labels("Nearest Neighbours", "Privacy Risk Rank")
g.set_titles("{row_name}")
g.savefig(f'{os.path.dirname(os.getcwd())}/output/plots/knn_privacy.pdf', bbox_inches='tight')

# %%
