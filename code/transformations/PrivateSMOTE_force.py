"""Apply interpolation with SMOTE
This script will apply SMOTE technique in the single out cases.
"""
# %%
import psutil
from os import sep
import re
import ast
import pandas as pd
import numpy as np
import random
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
import random
from random import randrange
import time


def keep_numbers(data):
    """mantain correct data types according to the data"""
    data_types = data.copy()
    for col in data.columns:
        # transform strings to digits
        if isinstance(data[col].iloc[0], str) and data[col].iloc[0].isdigit():
            data[col] = data[col].astype(float)
        # remove trailing zeros
        if isinstance(data[col].iloc[0], (int, float)):
            if int(data[col].iloc[0]) == float(data[col].iloc[0]):
                data_types[col] = data[col].astype(int)
            else: data[col] = data_types[col].astype(float)
    return data, data_types


def aux_singleouts(key_vars, dt):
    """create single out variable based on k-anonymity"""
    k = dt.groupby(key_vars)[key_vars[0]].transform(len)
    dt['single_out'] = np.where(k == 1, 1, 0)
    return dt


class Smote:
    """Apply Smote
    """
    def __init__(self, samples, singleout, y, N, k):
        """Initiate arguments

        Args:
            samples (pd.Dataframe): training samples
            singleout (pd.Series): singleout variable
            y (pd.Series): target sample
            N (int): number of interpolations per observation
            k (int): number of nearest neighbours
        """
        self.samples = samples.reset_index(drop=True)
        self.n_samples = samples.shape[0]
        self.n_attrs = samples.shape[1]
        self.singleout = singleout.reset_index(drop=True)
        self.y = y
        self.N = N
        self.k = k
        self.newindex=0
        # transform all data in ndarray with the same dtypes
        self.x = np.array(self.samples, dtype=self.samples.dtypes)

    def over_sampling(self):
        """find the nearest neighbors and populate with new data

        Returns:
            pd.DataFrame: synthetic data
        """
        N = int(self.N)

        # OHE + standardization for nearest neighbor
        encoded_data = pd.get_dummies(self.samples, drop_first=True).astype(int)
        standardized_data = StandardScaler().fit_transform(encoded_data)
        neighbors = NearestNeighbors(n_neighbors=self.k+1).fit(standardized_data)

        # transform the tager in 1D-array
        self.y = np.array(self.y)

        # inicialize the synthetic samples
        self.synthetic = np.empty(shape=(self.n_samples * N, self.n_attrs+1), dtype=self.samples.dtypes)
        
        # find the categories for each categorical column in all sample
        self.unique_values = [self.samples.loc[:,col].unique() for col in self.samples.select_dtypes(np.object)]

        # find the minimun value for each numerical column
        self.min_values = [self.samples.loc[:,col].min() if isinstance(self.samples[col].iloc[0], (int, float)) else np.nan for col in self.samples.columns]

        # for each observation find nearest neighbours
        for i in range(len(self.samples)):
            nnarray = neighbors.kneighbors(
                standardized_data[i].reshape(1, -1), return_distance=False)[0]
            self._populate(N, i, nnarray)

        return self.synthetic

    def _populate(self, N, i, nnarray):
        # populate N times
        while N!=0:
            # find index of nearest neighbour excluding the observation in comparison
            neighbour = randrange(1, self.k+1)

            # generate new numerical value for each column
            new_nums_values = [orig_val + np.multiply(random.choice([-1, 1]), random.uniform(0, 1)) \
                               if neighbor_val==orig_val and isinstance(orig_val, (int, float)) and orig_val!=self.min_values[j] \
                               else orig_val + random.uniform(0, 1) if neighbor_val==orig_val and isinstance(orig_val, (int, float)) and orig_val==self.min_values[j] \
                                else (orig_val + np.multiply(neighbor_val-orig_val, random.uniform(0, 1)) if isinstance(orig_val, (int, float)) and neighbor_val!=orig_val \
                                      else orig_val) \
                                    for j, (neighbor_val, orig_val) in enumerate(zip(self.x[nnarray[neighbour]], self.x[i]))]

            if len(self.unique_values) > 0:
                # find the categories for each categorical column in nearest neighbors sample
                nn_unique = [self.samples.loc[nnarray[1:self.k+1],col].unique() for col in self.samples.select_dtypes(np.object)]

                # randomly select a category
                new_cats_values = [random.choice(self.unique_values[u]) if len(nn_unique[u]) == 1 else random.choice(nn_unique[u]) for u in range(len(self.unique_values))]
        
                # replace the old categories
                iter_cat_calues = iter(new_cats_values)
                new_nums_values = [next(iter_cat_calues) if isinstance(val, str) else val for val in new_nums_values]

            # assign interpolated values
            self.synthetic[self.newindex, 0:len(new_nums_values)] = np.array(new_nums_values)

            # assign intact target variable
            self.synthetic[self.newindex, len(new_nums_values)] = self.y[i]
            self.newindex+=1

            N-=1

# %% privateSMOTE with "force" - add 1 or -1 when difference is 0
def PrivateSMOTE_force_(msg):
    """Generate several interpolated data sets considering all classes.

    Args:
        msg (str): name of the original file and respective PrivateSMOTE parameters
    """
    print(msg)

    start= time.time()
    output_interpolation_folder = 'output/oversampled/PrivateSMOTE_force'
    
    # get 80% of data to synthesise
    indexes = np.load('indexes.npy', allow_pickle=True).item()
    indexes = pd.DataFrame.from_dict(indexes)

    f = list(map(int, re.findall(r'\d+', msg.split('_')[0])))
    print(str(f[0]))
    data = pd.read_csv(f'original/{str(f[0])}.csv')

    index = indexes.loc[indexes['ds']==str(f[0]), 'indexes'].values[0]
    data_idx = list(set(list(data.index)) - set(index))
    data = data.iloc[data_idx, :]

    # encode string with numbers to numeric and remove trailing zeros
    data, data_types = keep_numbers(data)
    
    list_key_vars = pd.read_csv('list_key_vars.csv')
    set_key_vars = ast.literal_eval(
        list_key_vars.loc[list_key_vars['ds']==f[0], 'set_key_vars'].values[0])

    keys_nr = list(map(int, re.findall(r'\d+', msg.split('_')[2])))[0]
    print(keys_nr)
    keys = set_key_vars[keys_nr]
    data = aux_singleouts(keys, data)

    X_train = data.loc[data['single_out']==1, data.columns[:-2]] # training singleout sample
    Y_train = data.loc[data['single_out']==1, data.columns[-1]] # singleout variable values
    y = data.loc[data['single_out']==1, data.columns[-2]] # target variable values

    knn = list(map(int, re.findall(r'\d+', msg.split('_')[3])))[0]
    per = list(map(int, re.findall(r'\d+', msg.split('_')[4])))[0]

    new = Smote(X_train, Y_train, y, per, knn).over_sampling()

    newDf = pd.DataFrame(new, columns = data.columns[:-1])
    newDf = newDf.astype(dtype = data[data.columns[:-1]].dtypes)

    # assign singleout
    newDf['single_out'] = 1

    # add non single outs
    newDf = pd.concat(
        [newDf, data.loc[data['single_out']==0]])

    for col in newDf.columns[:-1]:
        if data_types[col].dtype == np.int64:
            newDf[col] = round(newDf[col], 0).astype(int)
        if data_types[col].dtype == np.float64:
            # get decimal places in float
            dec = str(data[col].values[0])[::-1].find('.')
            newDf[col] = round(newDf[col], dec)

    # save oversampled data
    newDf.to_csv(
        f'{output_interpolation_folder}{sep}{msg}.csv',
        index=False)

    # Store execution costs  
    process = psutil.Process()
    computational_costs = {'execution_time': time.time()-start,
                            'memory': process.memory_percent(),
                            'cpu_time_user': process.cpu_times()[0],
                            'cpu_time_system': process.cpu_times()[1],
                            'cpu_percent': process.cpu_percent()}
    
    computational_costs_df = pd.DataFrame([computational_costs])
    computational_costs_df.to_csv(
            f'computational_costs{sep}PrivateSMOTE_force{sep}{msg}.csv',
            index=False)

# %%
