"""Apply interpolation with SMOTE
This script will apply SMOTE technique in the single out cases.
"""
# %%
import cupy as cp
from os import sep
import re
import ast
import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler, LabelEncoder
from scipy import stats
from random import randrange
import argparse

parser = argparse.ArgumentParser(description='Master Example')
# parser.add_argument('--input_folder', type=str, help='Input folder', default="./input")
parser.add_argument('--input_file', type=str, default="none")
args = parser.parse_args()


def keep_numbers(data):
    """fix data types according to the data"""
    data_types = data.copy()
    for col in data.columns:
        # transform numerical strings to digits
        if isinstance(data[col].iloc[0], str) and data[col].iloc[0].isdigit():
            data[col] = data[col].astype(float)
        # remove trailing zeros
        if isinstance(data[col].iloc[0], (int, float)):
            if int(data[col].iloc[0]) == float(data[col].iloc[0]):
                data[col] = data[col].astype(int)
            else: data[col] = data_types[col].astype(float)
    return data, data_types


def aux_singleouts(key_vars, dt):
    """create single out variable based on k-anonymity"""
    k = dt.groupby(key_vars)[key_vars[0]].transform(len)
    dt['single_out'] = np.where(k < 3 , 1, 0)
    return dt

def remove_outliers(key_vars, dt):
    newdf = dt[key_vars].select_dtypes(include=np.number)
    newdf = newdf[(np.abs(stats.zscore(newdf)) < 3).all(axis=1)]
    dt = dt[dt.index.isin(newdf.index)]
    return dt

def check_and_adjust_data_types(origDF, newDf, data_types):
    for col in newDf.columns[:-1]:
        if data_types[col].dtype == np.int64:
            newDf[col] = round(newDf[col], 0).astype(int)
        elif data_types[col].dtype == np.float64:
            # Adjust the trailing values for float columns
            dec = str(origDF[col].values[0])[::-1].find('.')
            newDf[col] = round(newDf[col], dec)
    return newDf


class PrivSmote:
    """Apply Smote
    """
    def __init__(self, samples, N, k, ep):
        """Initiate arguments

        Args:
            data (pd.Dataframe): all data
            y (pd.Series): target sample
            N (int): number of interpolations per observation
            k (int): number of nearest neighbours
            ep (float): privacy budget (epsilon)
        """
        self.samples = samples.reset_index(drop=True)
        self.N = N
        self.k = k
        self.ep = ep
        self.newindex = 0

        # singleout samples that will be replaced
        self.X_train = self.samples.loc[self.samples['single_out'] == 1, self.samples.columns[:-2]]
        print("n singleouts: ", self.X_train.shape)
        # target variable values
        # self.y = self.prepare_target_variable(self.samples.loc[:, self.samples.columns[-2]])
        self.y = cp.array(self.samples.loc[:, self.samples.columns[-2]])
        print(self.y)
        # drop singlout and target variables to knn
        self.data_knn = self.samples.loc[:, self.samples.columns[:-2]]
        print("all data: ", self.data_knn.shape)
        print(set(self.data_knn.dtypes))
        # nr of samples and attributs to synthetize
        self.n_samples = self.X_train.shape[0]
        self.n_attrs = self.X_train.shape[1]
        # transform singleout samples in ndarray
        self.x = cp.array(self.data_knn)

        
    def prepare_target_variable(self, target_variable):
        # Convert non-numeric values to numeric
        if isinstance(self.samples[target_variable], object):
            label_encoder = LabelEncoder()
            self.samples[target_variable] = label_encoder.fit_transform(self.samples[target_variable])
        return cp.array(self.samples[target_variable])

    def prepare_categorical_variables(self, data):
        # One-hot encode categorical columns
        data_encoded = pd.get_dummies(data, drop_first=True)
        return cp.array(data_encoded)

    def over_sampling(self):
        """find the nearest neighbors and populate with new data

        Returns:
            pd.DataFrame: synthetic data
        """
        N = int(self.N)
        # OHE + standardization for nearest neighbor using all data
        encoded_data = self.prepare_categorical_variables(self.data_knn)

        # Convert Cupy array to NumPy array
        encoded_data_numpy = cp.asnumpy(encoded_data)

        # Standardize the data using NumPy array
        standardized_data = StandardScaler().fit_transform(encoded_data_numpy)

        neighbors = NearestNeighbors(n_neighbors=self.k + 1).fit(standardized_data)

        # inicialize the synthetic samples
        self.synthetic = cp.empty(shape=(self.n_samples * N, self.n_attrs + 1), dtype='float32')

        # find the categories for each categorical column in all sample
        self.unique_values = [cp.unique(self.data_knn.loc[:, col]) for col in self.data_knn.select_dtypes(object)]
        #print(self.unique_values)
        # find the minimun value for each numerical column
        self.min_values = [self.data_knn[col].min() if not isinstance(self.data_knn[col].iloc[0], str) else cp.nan for col in
                           self.data_knn.columns]
        #print(self.min_values)
        # find the maximum value for each numerical column
        self.max_values = [self.data_knn[col].max() if not isinstance(self.data_knn[col].iloc[0], str) else cp.nan for col
                           in self.data_knn.columns]
        #print(self.max_values)
        # find the standard deviation value for each numerical column
        self.std_values = [cp.std(self.data_knn[col]) if not isinstance(self.data_knn[col].iloc[0], str) else cp.nan for
                           col in self.data_knn.columns]

        # for each observation find nearest neighbours
        for i, _ in enumerate(cp.asnumpy(standardized_data)):
            if i in self.X_train.index:
                #print(i)
                nnarray = neighbors.kneighbors(cp.asnumpy(standardized_data[i].reshape(1, -1)),
                                               return_distance=False)[0]  # Convert to NumPy array before querying
                #print(nnarray)
                self._populate(N, i, nnarray)
        return cp.asnumpy(self.synthetic)  # Convert back to NumPy array before returning

    def _populate(self, N, i, nnarray):
        # populate N times
        while N != 0:
            # find index of nearest neighbour excluding the observation in comparison
            neighbour = randrange(1, self.k + 1)
            # print(len(self.x[i]))
            # print(self.x[i])
            # print(self.x[nnarray[neighbour]])
            # print(neighbour)
            # print(cp.random.laplace(0, 1 / self.ep, size=None))
            # control flip (with standard deviation) for equal neighbor and original values 
            flip = [cp.multiply(cp.multiply(
                            cp.random.choice([-1, 1], size=1), std),
                            cp.random.laplace(0, 1 / self.ep, size=None))
                            if cp.issubdtype(orig_val, (cp.floating, cp.integer))
                            else orig_val
                            for std, orig_val in zip(self.std_values, self.x[i])]
            #print(flip)

            # without flip when neighbour is different from original
            noise = [cp.multiply(
                            neighbor_val - orig_val,
                            cp.random.laplace(0, 1 / self.ep, size=None))
                            if cp.issubdtype(orig_val, (cp.float64, cp.int64))
                            else orig_val
                            for (neighbor_val, orig_val) in zip(self.x[nnarray[neighbour]], self.x[i])]
            
            #print(noise)
            # generate new numerical value for each column
            new_nums_values = [orig_val + flip[j]
                    if neighbor_val == orig_val 
                    and cp.issubdtype(orig_val, (cp.floating, cp.integer)) 
                    and self.min_values[j] <= orig_val + flip[j] <= self.max_values[j]
                    else orig_val - flip[j]
                    if neighbor_val == orig_val and cp.issubdtype( orig_val, (cp.floating, cp.integer)) 
                    and (self.min_values[j] > orig_val + flip[j]
                    or orig_val + flip[j] > self.max_values[j])
                    else orig_val + noise[j]
                    if neighbor_val != orig_val and cp.issubdtype(orig_val, (cp.floating, cp.integer)) 
                    and self.min_values[j] <= orig_val + noise[j] <= self.max_values[j]
                    else orig_val - noise[j]
                    if neighbor_val != orig_val and cp.issubdtype(orig_val, (cp.floating, cp.integer)) 
                    and (self.min_values[j] > orig_val + noise[j] > self.max_values[j])
                    else orig_val
                    for j, (neighbor_val, orig_val) in enumerate(zip(self.x[nnarray[neighbour]], self.x[i]))]

            #print(new_nums_values)
            if len(self.unique_values) > 0:
                # find the categories for each categorical column in nearest neighbors sample
                nn_unique = [cp.unique( self.samples.loc[nnarray[1 : self.k + 1], col]) 
                            for col in self.data_knn.select_dtypes("object")]

                # randomly select a category
                new_cats_values = [cp.random.choice(self.unique_values[u], size=None)
                                if len(nn_unique[u]) == 1
                                else cp.random.choice(nn_unique[u], size=None)
                                for u in range(len(self.unique_values))]

                # replace the old categories
                iter_cat_calues = iter(new_cats_values)
                new_nums_values = [next(iter_cat_calues)
                                if cp.issubdtype(val, cp.character)
                                else val
                                for val in new_nums_values]
                        
            # Concatenate the arrays along axis=0
            synthetic_array = cp.hstack(new_nums_values)
                    
            # assign interpolated values
            self.synthetic[self.newindex, 0 : synthetic_array.shape[0]] = synthetic_array
            
            # assign intact target variable
            self.synthetic[self.newindex, synthetic_array.shape[0]] = self.y[i]
            self.newindex += 1
            N -= 1

# %% 
def PrivateSMOTE_force_laplace_(input_file):
    """Generate several interpolated data sets considering all classes.

    Args:
        msg (str): name of the original file and respective PrivateSMOTE parameters
    """
    print(input_file)

    output_interpolation_folder = 'output/oversampled/5-PrivateSMOTE'
    
    # get 80% of data to synthesise
    indexes = np.load('indexes.npy', allow_pickle=True).item()
    indexes = pd.DataFrame.from_dict(indexes)

    f = list(map(int, re.findall(r'\d+', input_file.split('_')[0])))
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

    keys_nr = list(map(int, re.findall(r'\d+', input_file.split('_')[2])))[0]
    print(keys_nr)
    keys = set_key_vars[keys_nr]

    # print(data.shape)
    data = aux_singleouts(keys, data)

    # encoded target
    label_encoder = LabelEncoder()
    def enc_dec(target, action):
        if isinstance(target, object):
            target = label_encoder.fit_transform(target) if action=='encode' else label_encoder.inverse_transform(target)
        return target

    data[data.columns[-2]] = enc_dec(data[data.columns[-2]], 'encode')

    knn = list(map(int, re.findall(r'\d+', input_file.split('_')[3])))[0]
    per = list(map(int, re.findall(r'\d+', input_file.split('_')[4])))[0]
    ep = list(map(float, re.findall(r'\d+', input_file.split('_')[1])))[0]

    new = PrivSmote(data, per, knn, ep).over_sampling()
    # Convert synthetic data back to a Pandas DataFrame
    newDf = pd.DataFrame(cp.asnumpy(new), columns=data.columns[:-1])
    newDf = newDf.astype(dtype=data[data.columns[:-1]].dtypes)
    newDf = pd.concat([newDf, pd.DataFrame({'single_out': [1] * newDf.shape[0]})], axis=1)
    newDf[newDf.columns[-2]] = enc_dec(newDf[newDf.columns[-2]], 'decode')

    if newDf.shape[0] != data.shape[0]:
        newDf = pd.concat([newDf, data.loc[data['single_out'] == 0]])

    # Check and adjust data types and trailing values
    newDf = check_and_adjust_data_types(data, newDf, data_types)
    
    newDf.to_csv(f'{output_interpolation_folder}{sep}{input_file}.csv', index=False)

PrivateSMOTE_force_laplace_(args.input_file)