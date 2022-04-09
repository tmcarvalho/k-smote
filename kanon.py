"""Find singleouts to synthetise
This script will create a new varible indicating which observations
correspond to a unique signature.
Such a variable is similar to a binary target variable, where 1 is a singleout
and 0 otherwise.
"""

from os import sep, getcwd
import pandas as pd
import numpy as np

def single_outs():
    """Creates a new binary variable with singleouts information based on selected quasi-identifiers

    Returns:
        dataframe: Dataframe with single outs variable
    """
    data = pd.read_csv(f'{getcwd()}{sep}dataset.csv')

    key_vars = ['age', 'det_ind_code', 'det_occ_code', 'weeks_worked', 'year']

    data_copy = data.copy()

    k = data.groupby(key_vars)[key_vars[0]].transform(len)

    data_copy['single_out'] = None
    data_copy['single_out'] = np.where(k == 1, 1, 0)

    return data_copy, key_vars
