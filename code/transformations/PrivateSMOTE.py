"""Apply interpolation with SMOTE
This script will apply SMOTE technique in the single out cases.
"""
# %%
from os import sep, walk
import re
import ast
import pandas as pd
import numpy as np
from collections import defaultdict
from sklearn.preprocessing import LabelEncoder
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
    dt['single_out'] = None
    dt['single_out'] = np.where(k == 1, 1, 0)
    return dt


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

            difference = self.samples[nnarray[neighbour]] - self.samples[i]
            # multiply with a weight
            weight = random.uniform(0, 1)
            additive = np.multiply(difference,weight)

            # assign interpolated values
            self.synthetic[self.newindex, 0:len(self.synthetic[self.newindex])-1] = self.samples[i]+additive
            # assign intact target variable
            self.synthetic[self.newindex, len(self.synthetic[self.newindex])-1] = self.y[i]
            self.newindex+=1

# %%
def interpolation_singleouts_A(original_folder, file):
    """Generate several interpolated data sets considering all classes.

    Args:
        original_folder (string): path of original folder
        file (string): name of file
    """

    output_interpolation_folder = '../output/oversampled/smote_singleouts/'
    data = pd.read_csv(f'{original_folder}/{file}')

    # get 80% of data to synthesise
    indexes = np.load('../indexes.npy', allow_pickle=True).item()
    indexes = pd.DataFrame.from_dict(indexes)

    f = list(map(int, re.findall(r'\d+', file.split('_')[0])))
    index = indexes.loc[indexes['ds']==str(f[0]), 'indexes'].values[0]
    data_idx = list(set(list(data.index)) - set(index))
    data = data.iloc[data_idx, :]

    # encode string with numbers to numeric
    data = encode(data) 
    # apply LabelEncoder to categorical attributes
    label_encoder_dict = defaultdict(LabelEncoder)
    data_encoded = data.apply(lambda x: label_encoder_dict[x.name].fit_transform(x) if x.dtype=='object' else x)
    # remove trailing zeros in integers
    data_encoded = data_encoded.apply(lambda x: x.astype(int) if all(x%1==0) else x)

    map_dict = dict()
    for k in data.columns:
        if data[k].dtype=='object':
            keys = data[k]
            values = data_encoded[k]
            sub_dict = dict(zip(keys, values))
            map_dict[k] = sub_dict

    list_key_vars = pd.read_csv('../list_key_vars.csv')
    set_key_vars = ast.literal_eval(
        list_key_vars.loc[list_key_vars['ds']==f[0], 'set_key_vars'].values[0])

    for idx, keys in enumerate(set_key_vars):
        dt = aux_singleouts(keys, data_encoded)
        zero = dt.loc[dt[dt.columns[-2]]==0,:]
        one = dt.loc[dt[dt.columns[-2]]==1,:]

        X_train_zero = zero.loc[zero['single_out']==1, zero.columns[:-2]]
        Y_train_zero = zero.loc[zero['single_out']==1, zero.columns[-1]]
        y_zero = zero.loc[zero['single_out']==1, zero.columns[-2]]

        X_train_one = one.loc[one['single_out']==1, one.columns[:-2]]
        Y_train_one = one.loc[one['single_out']==1, one.columns[-1]]
        y_one = one.loc[one['single_out']==1, one.columns[-2]]

        # getting the number of singleouts in training set
        singleouts_zero = Y_train_zero.shape[0]
        singleouts_one = Y_train_one.shape[0]

        # storing the singleouts instances separately
        x1_zero = np.ones((singleouts_zero, X_train_zero.shape[1]))
        x1_zero=[X_train_zero.iloc[i] for i, v in enumerate(Y_train_zero) if v==1.0]
        x1_zero=np.array(x1_zero)
        x1_one = np.ones((singleouts_one, X_train_one.shape[1]))
        x1_one=[X_train_one.iloc[i] for i, v in enumerate(Y_train_one) if v==1.0]
        x1_one=np.array(x1_one)

        y_zero=np.array(y_zero)
        y_one=np.array(y_one)

        knn = [1,3,5]
        per = [1,2,3]
        for k in knn:
            for p in per:
                if k<len(x1_zero):
                    new_zero = Smote(x1_zero, y_zero, p, k).over_sampling()
                    newDf_zero = pd.DataFrame(new_zero)
                    # restore feature name 
                    newDf_zero.columns = dt.columns[:-1]
                    # assign singleout
                    newDf_zero[dt.columns[-1]] = 1

                if k<len(x1_one):
                    new_one = Smote(x1_one, y_one, p, k).over_sampling()
                    newDf_one = pd.DataFrame(new_one)
                    newDf_one.columns = dt.columns[:-1]
                    newDf_one[dt.columns[-1]] = 1
                
                # concat two classes
                if len(x1_zero)==0:
                    print("class ZERO: zero singleouts - ", idx)
                    new = newDf_one
                elif len(x1_one)==0:
                    print("class ONE: zero singleouts - ", idx)
                    new = newDf_zero
                else:
                    new = pd.concat([newDf_zero, newDf_one])
                
                # add non single outs
                newDf = pd.concat([new, dt.loc[dt['single_out']==0]])

                for col in newDf.columns:
                    if dt[col].dtype == np.int64:
                        newDf[col] = round(newDf[col], 0).astype(int)
                    elif dt[col].dtype == np.float64:
                        # get decimal places in float
                        dec = str(dt[col].values[0])[::-1].find('.')
                        newDf[col] = round(newDf[col], dec)
                    else:    
                        newDf[col] = newDf[col].astype(dt[col].dtype)
                
                # decoded
                for key in map_dict.keys():
                    d = dict(map(reversed, map_dict[key].items()))
                    # get the closest key in dict as the synthetisation may not create exact values
                    newDf[key] = newDf[key].apply(lambda x: d.get(x) or d[min(d.keys(), key = lambda key: abs(key-x))])
                    #newDf[key] = newDf[key].map(d)

                # save oversampled data
                newDf.to_csv(
                    f'{output_interpolation_folder}{sep}ds{file.split(".csv")[0]}_smote_QI{idx}_knn{k}_per{p}.csv',
                    index=False)

# %% privateSMOTE regardless of the class
def interpolation_singleouts_B(original_folder, file):
    """Generate several interpolated data sets considering all classes.

    Args:
        original_folder (string): path of original folder
        file (string): name of file
    """

    output_interpolation_folder = '../output/'
    data = pd.read_csv(f'{original_folder}/{file}')

    # get 80% of data to synthesise
    indexes = np.load('../indexes.npy', allow_pickle=True).item()
    indexes = pd.DataFrame.from_dict(indexes)

    f = list(map(int, re.findall(r'\d+', file.split('_')[0])))
    index = indexes.loc[indexes['ds']==str(f[0]), 'indexes'].values[0]
    data_idx = list(set(list(data.index)) - set(index))
    data = data.iloc[data_idx, :].reset_index()
    indexes = data.sample(frac=.015).index.to_list()
    data = data.iloc[indexes,:]
    data.to_csv(f'{output_interpolation_folder}{sep}ds{file}',
                        index=False)
    # encode string with numbers to numeric
    data = encode(data) 
    # apply LabelEncoder to categorical attributes
    label_encoder_dict = defaultdict(LabelEncoder)
    data_encoded = data.apply(lambda x: label_encoder_dict[x.name].fit_transform(x) if x.dtype=='object' else x)
    # remove trailing zeros in integers
    data_encoded = data_encoded.apply(lambda x: x.astype(int) if all(x%1==0) else x)

    map_dict = dict()
    for k in data.columns:
        if data[k].dtype=='object':
            keys = data[k]
            values = data_encoded[k]
            sub_dict = dict(zip(keys, values))
            map_dict[k] = sub_dict

    list_key_vars = pd.read_csv('../list_key_vars.csv')
    set_key_vars = ast.literal_eval(
        list_key_vars.loc[list_key_vars['ds']==f[0], 'set_key_vars'].values[0])

    for idx, keys in enumerate(set_key_vars):
        dt = aux_singleouts(keys, data_encoded)
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
                        elif dt[col].dtype == np.float64:
                            # get decimal places in float
                            dec = str(dt[col].values[0])[::-1].find('.')
                            newDf[col] = round(newDf[col], dec)
                        else:    
                            newDf[col] = newDf[col].astype(dt[col].dtype)
                    
                    # decoded
                    for key in map_dict.keys():
                        d = dict(map(reversed, map_dict[key].items()))
                        newDf[key] = newDf[key].apply(lambda x: d.get(x) or d[min(d.keys(), key = lambda key: abs(key-x))])
                        # newDf[key] = newDf[key].map(d)

                    # save oversampled data
                    newDf.to_csv(
                        f'{output_interpolation_folder}{sep}ds{file.split(".csv")[0]}_smote_QI{idx}_knn{k}_per{p}.csv',
                        index=False)

                except: # no singleouts
                    pass
# %%
original_folder = '../original'
_, _, input_files = next(walk(f'{original_folder}'))

not_considered_files = [0,1,3,13,23,28,34,36,40,48,54,66,87]
# %%
for idx,file in enumerate(input_files):
    if int(file.split(".csv")[0]) not in not_considered_files:
        print(idx)
        print(file)
        interpolation_singleouts_A(original_folder, file)

# %%
for idx,file in enumerate(input_files):
    if int(file.split(".csv")[0]) in [33]:
        print(idx)
        print(file)
        interpolation_singleouts_B(original_folder, file)

# %%
