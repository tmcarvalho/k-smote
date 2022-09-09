# %%
from os import walk
import pandas as pd
from collections import defaultdict
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import BorderlineSMOTE

# %%
def interpolation(original_folder, file):
    """Generate several interpolated data sets.

    Args:
        original_folder (string): path of original folder
        file (string): name of file
    """
    output_interpolation_folder = '../output/oversampled/borderlineSmote'

    data = pd.read_csv(f'{original_folder}/{file}')
    label_encoder_dict = defaultdict(LabelEncoder)
    data_encoded = data.apply(lambda x: label_encoder_dict[x.name].fit_transform(x))
    map_dict = dict()
    
    for k in data.columns:
        if data[k].dtype=='object':
            keys = data[k]
            values = data_encoded[k]
            sub_dict = dict(zip(keys, values))
            map_dict[k] = sub_dict

    knn = [1, 3, 5]
    # percentage of majority and minority class
    ratios_smote = [0.5, 0.75, 1]
    for nn in knn:
        for smote in ratios_smote:
            try:
                smote_samp = BorderlineSMOTE(random_state=42,
                            k_neighbors=nn,
                            sampling_strategy=smote)
                # fit predictor and target variable
                X = data_encoded[data_encoded.columns[:-1]]
                y = data_encoded.iloc[:, -1]
                x_smote, y_smote = smote_samp.fit_resample(X, y)
                
                # add target
                x_smote[data_encoded.columns[-1]] = y_smote

                # decoded
                for key in map_dict.keys():  
                    d = dict(map(reversed, map_dict[key].items()))
                    x_smote[key] = x_smote[key].map(d)

                x_smote.to_csv(f'{output_interpolation_folder}/ds{file.split(".csv")[0]}_smote_knn{nn}_per{smote}.csv', index=False)

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
