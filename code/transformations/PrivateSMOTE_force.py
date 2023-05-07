"""Apply interpolation with SMOTE
This script will apply SMOTE technique in the single out cases.
"""
# %%
from os import sep, walk
import re
import ast
import pandas as pd
import numpy as np
import random
from sklearn.neighbors import NearestNeighbors
import random
from random import randrange

def encode(data):
    for col in data.columns:
        try: 
            data[col] = data[col].apply(lambda x: ast.literal_eval(x))
        except: pass
    return data


def aux_singleouts(key_vars, dt):
    k = dt.groupby(key_vars)[key_vars[0]].transform(len)
    dt['single_out'] = np.where(k == 1, 1, 0)
    return dt


class Smote:
    """Apply Smote
    """
    def __init__(self, samples, singleout, y, N, k):
        """Initiate arguments

        Args:
            samples (array): training samples
            y (1D array): target sample
            N (int): number of interpolations per observation
            k (int): number of nearest neighbours
            ep (float): privacy budget
        """
        self.n_samples = samples.shape[0]
        self.n_attrs = samples.shape[1]
        self.singleout = singleout
        self.y = y
        self.N = N
        self.k = k
        self.samples = samples
        self.synthetic = []

    def over_sampling(self):
        """find the nearest neighbors and populate with new data

        Returns:
            pd.DataFrame: synthetic data
        """
        N = int(self.N)

        numeric_vars = self.samples.select_dtypes(include=np.number).columns

        # transform the numerical attributes in ndarray for knn
        x1 = [self.samples[numeric_vars].iloc[i] for i, _ in enumerate(self.singleout.index)]
        x1 = np.array(x1)
        self.y = np.array(self.y)
        
        # transform all data in ndarray
        x = np.ones((self.n_samples, self.n_attrs))
        x = [self.samples.iloc[i] for i, _ in enumerate(self.singleout.index)]
        x = np.array(x)
        neighbors = NearestNeighbors(n_neighbors=self.k+1).fit(x1)

        # for each observation find nearest neighbours
        for i, _ in enumerate(x1):
            nnarray = neighbors.kneighbors(
                x1[i].reshape(1, -1), return_distance=False)[0]
            self._populate(N, i, nnarray, x)

        return self.synthetic

    def _populate(self, N, i, nnarray, x):

        # populate N times
        for j in range(N):
            # find index of nearest neighbour excluding the observation in comparison
            neighbour = randrange(1, self.k+1)
            new_sample = []
            z=0
            for a, b in zip(x[nnarray[neighbour]], x[i]):
                if str(a).isdigit():
                    # print(a, b)
                    if a-b==0:
                        new_sample.append(b + np.multiply(b + random.choice([-1, 1]), random.uniform(0, 1)))
                    else: new_sample.append(b + np.multiply(a-b, random.uniform(0, 1)))
                else:
                    unique_values = self.samples.iloc[:,z].unique()
                    nn_unique = self.samples.iloc[nnarray[1:self.k+1],z].unique()
                    if len(nn_unique) == 1 and nn_unique[0] == b:
                        new_sample.append(random.choice(unique_values))
                    else: new_sample.append(random.choice(nn_unique))
                z+=1

            # assign intact target variable
            new_sample.append(self.y[i])
            # assign interpolated values
            self.synthetic.append(new_sample)


# %% privateSMOTE with "force" - add 1 or -1 when difference is 0
def PrivateSMOTE_force(original_folder, file):
    """Generate several interpolated data sets considering all classes.

    Args:
        original_folder (string): path of original folder
        file (string): name of file
    """

    output_interpolation_folder = '../output/oversampled/PrivateSMOTE_force'
    data = pd.read_csv(f'{original_folder}/{file}')
    print(len(data))
    # get 80% of data to synthesise
    indexes = np.load('../indexes.npy', allow_pickle=True).item()
    indexes = pd.DataFrame.from_dict(indexes)

    f = list(map(int, re.findall(r'\d+', file.split('_')[0])))
    index = indexes.loc[indexes['ds']==str(f[0]), 'indexes'].values[0]
    data_idx = list(set(list(data.index)) - set(index))
    data = data.iloc[data_idx, :]
    print(len(data))
    
    # encode string with numbers to numeric
    data = encode(data) 
    
    list_key_vars = pd.read_csv('../list_key_vars.csv')
    set_key_vars = ast.literal_eval(
        list_key_vars.loc[list_key_vars['ds']==f[0], 'set_key_vars'].values[0])

    for idx, keys in enumerate(set_key_vars):
        data = aux_singleouts(keys, data)
        X_train = data.loc[data['single_out']==1, data.columns[:-2]]
        Y_train = data.loc[data['single_out']==1, data.columns[-1]]
        y = data.loc[data['single_out']==1, data.columns[-2]]

        knn = [1,3,5]
        per = [1,2,3]
        for k in knn:
            for p in per:
                new = Smote(X_train, Y_train, y, p, k).over_sampling()
                newDf = pd.DataFrame(new)
                # restore feature name 
                newDf.columns = data.columns[:-1]
                # assign singleout
                newDf[data.columns[-1]] = 1
                # add non single outs
                newDf = pd.concat(
                    [newDf, data.loc[data['single_out']==0]])
                
                for col in newDf.columns:
                    if data[col].dtype == np.int64:
                        newDf[col] = round(newDf[col], 0).astype(int)
                    elif data[col].dtype == np.float64:
                        # get decimal places in float
                        dec = str(data[col].values[0])[::-1].find('.')
                        newDf[col] = round(newDf[col], dec)
                    else:    
                        newDf[col] = newDf[col].astype(data[col].dtype)
                    
                    # save oversampled data
                    newDf.to_csv(
                        f'{output_interpolation_folder}{sep}ds{file.split(".csv")[0]}_smote_QI{idx}_knn{k}_per{p}.csv',
                        index=False)

# %%
original_folder = '../original'
_, _, input_files = next(walk(f'{original_folder}'))

not_considered_files = [0,1,3,13,23,28,34,36,40,48,54,66,87]
# %%
for idx,file in enumerate(input_files):
    if int(file.split(".csv")[0]) not in not_considered_files:
        print(idx)
        print(file)
        PrivateSMOTE_force(original_folder, file)

# %%
