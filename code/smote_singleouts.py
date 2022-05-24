"""Apply interpolation with SMOTE
This script will apply SMOTE technique in the single out cases.
"""
# %%
from os import sep, walk
import pandas as pd
import numpy as np
from imblearn.over_sampling import SMOTE
from collections import defaultdict
from sklearn.preprocessing import LabelEncoder
from kanon import single_outs_sets
import random
from sklearn.neighbors import NearestNeighbors
import seaborn as sns
from matplotlib import pyplot as plt

# %%
def interpolation_singleouts(original_folder, file):
    """Generate several interpolated data sets.

    Args:
        original_folder (string): path of original folder
        file (string): name of file
    """
    output_interpolation_folder = '../output/oversampled/smote_singleouts/'
    data = pd.read_csv(f'{original_folder}/{file}')

    # apply LabelEncoder because of smote
    label_encoder_dict = defaultdict(LabelEncoder)
    data_encoded = data.apply(lambda x: label_encoder_dict[x.name].fit_transform(x))
    map_dict = dict()
    
    for k in data.columns:
        if data[k].dtype=='object':
            keys = data[k]
            values = data_encoded[k]
            sub_dict = dict(zip(keys, values))
            map_dict[k] = sub_dict

    set_data, _ = single_outs_sets(data_encoded)

    knn = [1, 3, 5]
    # percentage of majority class
    ratios = [2, 3, 4]
    for idx, dt in enumerate(set_data):
        for nn in knn:
            for ratio in ratios:
                dt = set_data[idx]
                dt_singleouts = dt.loc[dt['single_out']==1, :].reset_index(drop=True)
                
                X = dt_singleouts.loc[:, dt_singleouts.columns[:-2]]
                y = dt_singleouts.loc[:, dt_singleouts.columns[-2]]

                try:
                    mijority_class = np.argmax(y.value_counts().sort_index(ascending=True))
                    minority_class = np.argmin(y.value_counts().sort_index(ascending=True))
                    smote = SMOTE(random_state=42,
                                k_neighbors=3,
                                sampling_strategy={
                                    minority_class: int(ratio*len(y[y==minority_class])),
                                    mijority_class: int(ratio*len(y[y==mijority_class]))})
                    
                    # fit predictor and target variable
                    x_smote, y_smote = smote.fit_resample(X, y)
                    # add target variable
                    x_smote[dt.columns[-2]] = y_smote
                    # add single out to further apply record linkage
                    x_smote[dt.columns[-1]] = 1

                    # remove original single outs from oversample
                    oversample = x_smote.copy()
                    oversample = oversample.drop(dt_singleouts.index)
                    oversample = pd.concat([oversample, dt.loc[dt['single_out']==0,:]]).reset_index(drop=True)   

                    # decoded
                    for key in map_dict.keys():
                        d = dict(map(reversed, map_dict[key].items()))
                        oversample[key] = oversample[key].map(d)

                    # save oversampled data
                    oversample.to_csv(
                        f'{output_interpolation_folder}{sep}ds{file.split(".csv")[0]}_smote_QI{idx}_knn{nn}_per{ratio}.csv',
                        index=False)    

                except: pass
                                        

# %%
original_folder = '../original'
_, _, input_files = next(walk(f'{original_folder}'))

not_considered_files = [0,1,3,13,23,28,34,36,40,48,54,66,87]
for idx,file in enumerate(input_files):
    if int(file.split(".csv")[0]) not in not_considered_files:
        print(idx)
        print(file)
        interpolation_singleouts(original_folder, file)

""" NOTE
Smote from imblearn doesn't work when number of minority class is equal to majority class (e.g. dataset 34.csv)
The minimum to duplicate cases is per=2, if per=1, Smote doesn't create new instances
"""
# %%
#################### CASE STUDY SMOTE ######################
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
    #ax.get

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


""" NOTE
In this casse study it is verified that smote from imblearn creates minority cases from minority neighbours.
Therefore, we need to implement smote from scratch to create cases based on two classes.
"""
# %%
#################### SMOTE FROM SCRATCH #######################
import random
from random import randrange
from sklearn.neighbors import NearestNeighbors

class Smote:
    def __init__(self,samples,y,N,k):
        """Initiate arguments

        Args:
            samples (array): training samples
            y (1D array): target sample
            N (int): number of interpolations per observation
            k (int): number of nearest neighbours
        """
        self.n_samples = samples.shape[0]
        self.n_attrs=samples.shape[1]
        self.y=y
        self.N=N
        self.k=k
        self.samples=samples
        self.newindex=0

    def over_sampling(self):
        N=int(self.N)
        self.synthetic = np.zeros((self.n_samples * N, self.n_attrs+1))
        neighbors=NearestNeighbors(n_neighbors=self.k+1).fit(self.samples)

        # for each observation find nearest neighbours
        for i in range(len(self.samples)):
            nnarray=neighbors.kneighbors(self.samples[i].reshape(1,-1),return_distance=False)[0]
            self._populate(N,i,nnarray)

        return self.synthetic

    def _populate(self,N,i,nnarray):
        # populate N times
        for j in range(N):
            # find index of nearest neighbour excluding the observation in comparison
            neighbour = randrange(1, self.k+1)

            difference = abs(self.samples[i]-self.samples[nnarray[neighbour]])
            # multiply with a weight
            weight = random.uniform(0, 1)
            additive = np.multiply(difference,weight)

            # assign interpolated values
            self.synthetic[self.newindex, 0:len(self.synthetic[self.newindex])-1] = self.samples[i]+additive
            # assign intact target variable
            self.synthetic[self.newindex, len(self.synthetic[self.newindex])-1] = self.y[i]
            self.newindex+=1

# %%
def interpolation_singleouts_scratch(original_folder, file):
    """Generate several interpolated data sets considering all classes.

    Args:
        original_folder (string): path of original folder
        file (string): name of file
    """

    output_interpolation_folder = '../output/oversampled/smote_singleouts_scratch/'
    data = pd.read_csv(f'{original_folder}/{file}')

    # apply LabelEncoder beacause of smote
    label_encoder_dict = defaultdict(LabelEncoder)
    data_encoded = data.apply(lambda x: label_encoder_dict[x.name].fit_transform(x))
    map_dict = dict()
    
    for k in data.columns:
        if data[k].dtype=='object':
            keys = data[k]
            values = data_encoded[k]
            sub_dict = dict(zip(keys, values))
            map_dict[k] = sub_dict

    set_data, _ = single_outs_sets(data_encoded)

    for idx, dt in enumerate(set_data):
        X_train = dt.loc[dt['single_out']==1, dt.columns[:-2]]
        Y_train = dt.loc[dt['single_out']==1, dt.columns[-1]]
        y = dt.loc[dt['single_out']==1, dt.columns[-2]]

        # getting the number of singleouts in training set
        singleouts = Y_train.shape[0]

        # storing the singleouts instances separately
        x1 = np.ones((singleouts, X_train.shape[1]))
        x1=[X_train.iloc[i] for i, v in enumerate(Y_train) if v==1.0]
        x1=np.array(x1)

        y=np.array(y)

        knn = [1,3,5]
        per = [1,2,3]
        for k in knn:
            for p in per:
                try:
                    new = Smote(x1, y, p, k).over_sampling()
                    newDf = pd.DataFrame(new)
                    # restore feature name 
                    newDf.columns = dt.columns[:-1]
                    # assign singleout
                    newDf[dt.columns[-1]] = 1
                    # add non single outs
                    newDf = pd.concat([newDf, dt.loc[dt['single_out']==0]])
                    for col in newDf.columns:
                        if dt[col].dtype == np.int64:
                            newDf[col] = round(newDf[col], 0).astype(int)
                        else:    
                            newDf[col] = newDf[col].astype(dt[col].dtype)
                    
                    # decoded
                    for key in map_dict.keys():
                        d = dict(map(reversed, map_dict[key].items()))
                        newDf[key] = newDf[key].map(d)

                    # save oversampled data
                    newDf.to_csv(
                        f'{output_interpolation_folder}{sep}ds{file.split(".csv")[0]}_smote_QI{idx}_knn{k}_per{p}.csv',
                        index=False)

                except:
                    pass
# %%
original_folder = '../original'
_, _, input_files = next(walk(f'{original_folder}'))

not_considered_files = [0,1,3,13,23,28,34,36,40,48,54,66,87]
for idx,file in enumerate(input_files):
    if int(file.split(".csv")[0]) not in not_considered_files:
        print(idx)
        print(file)
        interpolation_singleouts_scratch(original_folder, file)

# %%
################ EXAMPLE ORIGINAL VS ONE CLASSS VS TWO CLASSES
original_32 = pd.read_csv("../original/32.csv")
package_32 = pd.read_csv("../output/oversampled/smote_singleouts/ds32_smote_QI4_knn3_per2.csv")
scratch_32 = pd.read_csv("../output/oversampled/smote_singleouts_scratch/ds32_smote_QI4_knn3_per1.csv")
# %%
idx=scratch_32[scratch_32['single_out']==1].index
fig, axs = plt.subplots(ncols=2, nrows=2, figsize=(10,10))
sns.scatterplot(data=original_32.loc[idx, :], x='age', y='hoursPerWeek', hue='label', palette="deep",s=14, ax=axs[0,0], legend=False).set_title("Original")
sns.scatterplot(data=package_32[package_32['single_out']==1], x='age', y='hoursPerWeek', hue='label', palette="deep", s=14, ax=axs[0,1], legend=False).set_title("One class")
sns.scatterplot(data=original_32, x='age', y='hoursPerWeek', hue='label', palette="deep", s=14, ax=axs[1,0], legend=False).set_title("Original")
sns.scatterplot(data=scratch_32[scratch_32['single_out']==1], x='age', y='hoursPerWeek', hue='label', palette="deep", s=14, ax=axs[1,1], legend=False).set_title("Both classes")
axs[0,0].set(xlim=(0,100))
axs[0,1].set(xlim=(0,100))
axs[1,0].set(xlim=(0,100))
axs[1,1].set(xlim=(0,100))

fig.savefig('../output/plots/smote_case_study.png',bbox_inches='tight')
# %%
