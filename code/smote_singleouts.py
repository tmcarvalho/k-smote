"""Apply interpolation with SMOTE
This script will apply SMOTE technique in the single out cases.
"""
# %%
from os import sep, walk
from pickle import TRUE
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import LabelEncoder
from kanon import single_outs_sets


def interpolation_singleouts(original_folder, file):
    """Generate several interpolated data sets.

    Args:
        original_folder (string): path of original folder
        file (string): name of file
    """
    output_interpolation_folder = '../output/oversampled/smote_singleouts/'
    data = pd.read_csv(f'{original_folder}/{file}')

    # apply LabelEncoder beacause of smote
    data = data.apply(LabelEncoder().fit_transform)
    set_data, _ = single_outs_sets(data)

    knn = [1, 3, 5]
    # percentage of majority class
    ratios = [0.5, 0.75, 1]
    for idx, dt in enumerate(set_data):
        for nn in knn:
            print(f'NUMBER OF KNN: {nn}')
            for ratio in ratios:
                print(f'NUMER OF RATIO: {ratio}')
                dt = set_data[idx]
                dt_singleouts = dt.loc[dt['single_out']==1, :].reset_index(drop=True)

                X = dt_singleouts.loc[:, dt_singleouts.columns[:-2]]
                y = dt_singleouts.loc[:, dt_singleouts.columns[-2]]

                mijority_class = np.argmax(y.value_counts().sort_index(ascending=True))
                minority_class = np.argmin(y.value_counts().sort_index(ascending=True))

                smote = SMOTE(random_state=42,
                            k_neighbors=3,
                            sampling_strategy={
                                minority_class: int(ratio*len(y[y==mijority_class])),
                                mijority_class: 2*len(y[y==mijority_class])})
                
                # fit predictor and target variable
                x_smote, y_smote = smote.fit_resample(X, y)
                # add target variable
                x_smote[dt.columns[-2]] = y_smote

                # add single out to further apply record linkage
                x_smote[dt.columns[-1]] = 1
                x_smote = pd.concat([x_smote, dt[dt['single_out']==0]])
                
                print(len(dt_singleouts))    
                # remove original single outs from oversample
                oversample = x_smote.copy()
                oversample = oversample.drop(
                    dt_singleouts.index).reset_index(drop=True)
  
                # save oversampled data
                oversample.to_csv(
                    f'{output_interpolation_folder}{sep}ds{file.split(".csv")[0]}_smote_QI{idx}_knn{nn}_per{ratio}.csv',
                    index=False)    
    
                                        

# %%
original_folder = '../original'
_, _, input_files = next(walk(f'{original_folder}'))

not_considered_files = [0,1,3,13,23,28,32,36,40,48,54,66,87]
for idx,file in enumerate(input_files):
    if int(file.split(".csv")[0]) not in not_considered_files:
        print(idx)
        interpolation_singleouts(original_folder, file)



# %%
#################### CASE STUDY SMOTE ######################
from matplotlib import pyplot as plt
import numpy as np
import seaborn as sns
from imblearn import FunctionSampler
from sklearn.datasets import make_classification


def create_dataset(
    n_samples=1000,
    weights=(0.01, 0.98),
    n_classes=2,
    class_sep=0.8,
    n_clusters=1,
):
    return make_classification(
        n_samples=n_samples,
        n_features=2,
        n_informative=2,
        n_redundant=0,
        n_repeated=0,
        n_classes=n_classes,
        n_clusters_per_class=n_clusters,
        weights=list(weights),
        class_sep=class_sep,
        random_state=0,
    )

def plot_resampling(X, y, sampler, ax, title=None):
    X_res, y_res = sampler.fit_resample(X, y)
    print(f'class 0: {len(y_res[y_res==0])}')
    print(f'class 1: {len(y_res[y_res==1])}')
    # ax.scatter(X_res.iloc[:, 0], X_res.iloc[:, 1], c=y_res, alpha=0.8, edgecolor="k")
    ax.scatter(X_res[:, 0], X_res[:, 1], c=y_res, alpha=0.8, edgecolor="k")
    if title is None:
        title = f"Resampling with {sampler.__class__.__name__}"
    ax.set_title(title)
    sns.despine(ax=ax, offset=10)
    ax.get

fig, axs = plt.subplots(nrows=1,ncols=2, figsize=(15, 10))

X, y = create_dataset(n_samples=150, weights=(0.1, 0.7))

samplers = [
    FunctionSampler(),
    SMOTE(random_state=42,
            k_neighbors=1,
            sampling_strategy={0:int(0.5*len(y[y==1]))}),    
]

for ax, sampler in zip(axs.ravel(), samplers):
    title = "Original dataset" if isinstance(sampler, FunctionSampler) else None
    plot_resampling(X, y, sampler, ax, title=title)
fig.tight_layout()
# %%
