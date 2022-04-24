"""Apply sinthetisation with SMOTE
This script will apply SMOTE technique in the single out cases.
"""
# %%
from os import sep, getcwd
from imblearn.over_sampling import SMOTE
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from privacy.kanon import single_outs
from privacy.record_linkage import record_linkage
from predictive_performance.modeling import evaluate_model

# %%
def apply_record_linkage(oversample_data, original_data, keys, args):
    """Apply record linkage and calculate the percentage of re-identification

    Args:
        oversample_data (pd.Dataframe): oversampled data
        original_data (pd.Dataframe): original dataframe
        keys (list): list of quasi-identifiers
        args (list): list with 3 values:
        nr of iteration of QIs; nr of knn; percentage in smote

    Returns:
        _type_: _description_
    """
    oversample_singleouts = oversample_data[oversample_data['single_out']==1]
    original_singleouts = original_data[original_data['single_out']==1]
    potential_matches = record_linkage(oversample_singleouts, original_singleouts, keys)
    # save record linkage results
    potential_matches.to_csv(
        f'{getcwd()}{sep}output{sep}record_linkage{sep}potential_matches_\
            QI{args[0]}_knn{args[1]}_per{args[2]}.csv', index=False)

    # get acceptable score (QIs match at least 50%)
    acceptable_score = potential_matches[potential_matches['Score'] >= \
        0.5*potential_matches['Score'].max()]
    level_1_acceptable_score = acceptable_score.groupby(['level_1'])['level_0'].size()
    per = ((1/level_1_acceptable_score.min()) * 100) / len(oversample_data)

    # get max score (all QIs match)
    max_score = potential_matches[potential_matches['Score'] == len(keys)]
    # find original single outs with an unique match in oversampled data - 100% match
    level_1_max_score = max_score.groupby(['level_1'])['level_0'].size()
    per_100 = (len(level_1_max_score[level_1_max_score == 1]) * 100) / len(oversample_data)

    return per, per_100
# %%
# %%
data = pd.read_csv(f'{getcwd()}{sep}dataset.csv')
set_data, set_key_vars = single_outs(data)

# key_vars_idx = [data.columns.get_loc(c) for c in key_vars if c in data]
knn = [1, 2, 3, 4, 5]
# percentage of majority class
ratios = [0.1, 0.2, 0.3, 0.5, 0.7]

for idx, key_vars in enumerate(set_key_vars):
    # apply LabelEncoder beacause of smote
    dt = set_data[idx].apply(LabelEncoder().fit_transform)
    for nn in knn:
        for ratio in ratios:
            smote = SMOTE(random_state=42,
                        k_neighbors=nn,
                        sampling_strategy=ratio)
            # fit predictor and target variable
            X = dt[dt.columns[:-1]]
            y = dt.iloc[:, -1]
            x_smote, y_smote = smote.fit_resample(X, y)

            # add single out to apply record linkage
            x_smote['single_out'] = y_smote
            # remove original single outs from oversample
            oversample = x_smote.copy()
            oversample = oversample.drop(dt[dt['single_out']==1].index).reset_index(drop=True)
            # save oversampled data
            oversample.to_csv(f'{getcwd()}{sep}output{sep}oversampled_data{sep}oversample_QI{idx}_\
                knn{nn}_per{ratio}.csv', index=False)
            percentage, percentage_100 = apply_record_linkage(
                oversample,
                dt,
                key_vars,
                [idx, nn, ratio])

            # prepare data to modeling
            X, y = dt.iloc[:2500, :-2], dt.iloc[:2500, -2]
            # predictive performance
            validation, test = evaluate_model(
                X,
                y,
                nn,
                percentage,
                percentage_100)

            # save validation and test results
            np.save(
                f'{getcwd()}{sep}output{sep}modeling{sep}oversampled{sep}\
                    validation{sep}validation_QI{idx}_knn{nn}_per{ratio}.npy', validation)
            np.save(
                f'{getcwd()}{sep}output{sep}modeling{sep}oversampled{sep}\
                    test{sep}test_QI{idx}_knn{nn}_per{ratio}.npy', test)
# %%
