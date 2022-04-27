"""Re-identification of single outs
This script will analyse the re-identification risk with
record linkage for single outs.
"""
# %%
from os import walk
import re
import numpy as np
import pandas as pd
import seaborn as sns
# %%
def transform_npy_to_csv():
    """_summary_
    """
    folder_performance = 'output/modeling/oversampled/validation'
    _, _, performance_files = next(walk(f'{folder_performance}'))
    performance_files.sort()

    for file in performance_files:
        result = np.load(f'{folder_performance}/{file}', allow_pickle='TRUE').item()
        result_rf = pd.DataFrame.from_dict(result['cv_results_Random Forest'])
        result_bag = pd.DataFrame.from_dict(result['cv_results_Bagging'])
        result_xgb = pd.DataFrame.from_dict(result['cv_results_Boosting'])
        result_logr = pd.DataFrame.from_dict(result['cv_results_Logistic Regression'])
        result_nn = pd.DataFrame.from_dict(result['cv_results_Neural Network'])

        # add attribute with model's name
        result_rf['model'] = 'Random Forest'
        result_bag['model'] = 'Bagging'
        result_xgb['model'] = 'XGBoost'
        result_logr['model'] = 'Logistic Regression'
        result_nn['model'] = 'Neural Network'

        # save csv with the 5 algorithms
        res = pd.concat([result_rf, result_bag, result_xgb, result_logr, result_nn])
        res.to_csv(f'{folder_performance}/{file.replace(".npy", ".csv")}', index=False)

# %%

def concat_all_results():
    """_summary_

    Returns:
        _type_: _description_
    """
    folder_performance = 'output/modeling/oversampled/validation'
    _, _, performance_files = next(walk(f'{folder_performance}'))
    performance_files.sort()
    performance_files = [file for file in performance_files if '.csv' in file]

    for idx, file in enumerate(performance_files):
        result = pd.read_csv(f'{folder_performance}/{file}')
        quasi_id = file.split('_')[1]
        quasi_id = list(map(int, re.findall(r'\d+', quasi_id)))[0]
        k_nn = file.split('_')[2]
        k_nn = list(map(int, re.findall(r'\d+', k_nn)))[0]
        per = file.split('_')[3]
        per = per.replace(".npy", "")
        per = list(map(float, re.findall(r'\d+\.\d+', per)))[0]
        result['qi'] = quasi_id
        result['knn'] = k_nn
        result['per'] = per
        if idx == 0:
            results = result.copy()
        else:
            results = pd.concat([results, result])

    return results

# %%
# transform_npy_to_csv()
all_results = concat_all_results()
# %%
sns.set_style('darkgrid')
sns.set(font_scale=1.5)
g = sns.FacetGrid(all_results,row='qi', height=8, aspect=1.5)
g.map(sns.boxplot, x="knn", y="mean_test_f1_weighted",
hue="per", data=all_results, palette='muted').add_legend()
g.set_axis_labels("KNN", "Fscore")
g.savefig('output/plots/fscore.pdf', bbox_inches='tight')
# %%
sns.set_style('darkgrid')
sns.set(font_scale=1.5)
g = sns.FacetGrid(all_results,row='qi', height=8, aspect=1.5)
g.map(sns.barplot, x="knn", y="privacy_risk", hue="per",
data=all_results, palette='muted').add_legend()
g.set_axis_labels("KNN", "Privacy Risk")
g.savefig('output/plots/risk.pdf', bbox_inches='tight')
# %%
