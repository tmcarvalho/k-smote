""" Apply smote, under and over
This script interpolate data using smote, under and over techniques.
"""
# %%
from os import walk
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import SMOTE, RandomOverSampler


def interpolation(original_folder, file):
    """Generate several interpolated data sets.

    Args:
        original_folder (string): path of original folder
        file (string): name of file
    """
    output_interpolation_folder = '../output/oversampled/smote_under_over'

    print(f'{original_folder}/{file}')
    data = pd.read_csv(f'{original_folder}/{file}')
    data = data.apply(LabelEncoder().fit_transform)
    knn = [1, 3, 5]
    # percentage of majority and minority class
    ratios_smote = [0.5, 0.75, 1]
    ratios_over = [0.5, 0.75, 1]
    ratios_under = [0.25, 0.5, 0.75, 1]
    for nn in knn:
        for smote in ratios_smote:
            try:
                smote_samp = SMOTE(random_state=42,
                            k_neighbors=nn,
                            sampling_strategy=smote)
                # fit predictor and target variable
                X = data[data.columns[:-1]]
                y = data.iloc[:, -1]
                x_smote, y_smote = smote_samp.fit_resample(X, y)

                # add target
                x_smote[data.columns[-1]] = y_smote
                x_smote.to_csv(f'{output_interpolation_folder}/ds{file.split(".csv")[0]}_smote_knn{nn}_per{smote}.csv', index=False)

            except: pass        

    for over in ratios_over:
        try:
            over_samp = RandomOverSampler(random_state=42,
                        sampling_strategy=over)
            # fit predictor and target variable
            X = data[data.columns[:-1]]
            y = data.iloc[:, -1]
            x_over, y_over = over_samp.fit_resample(X, y)

            # add target
            x_over[data.columns[-1]] = y_over
            x_over.to_csv(f'{output_interpolation_folder}/ds{file.split(".csv")[0]}_over_per{over}.csv', index=False)

        except: pass    
    
    for under in ratios_under:
        try:
            under_samp = RandomUnderSampler(random_state=42,
                        sampling_strategy=under)
            # fit predictor and target variable
            X = data[data.columns[:-1]]
            y = data.iloc[:, -1]
            x_under, y_under = under_samp.fit_resample(X, y)

            # add target
            x_under[data.columns[-1]] = y_under
            x_under.to_csv(f'{output_interpolation_folder}/ds{file.split(".csv")[0]}_under_per{under}.csv', index=False)
        except: pass    

# %%
original_folder = '../original'
_, _, input_files = next(walk(f'{original_folder}'))

not_considered_files = [0,1,3,13,23,28,34,36,40,48,54,66,87]
for idx,file in enumerate(input_files):
    if int(file.split(".csv")[0]) not in not_considered_files:
        print(idx)
        interpolation(original_folder, file)

# %%
