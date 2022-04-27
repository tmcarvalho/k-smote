"""Find singleouts to synthetise
This script will create a new varible indicating which observations
correspond to a unique signature.
Such a variable is similar to a binary target variable, where 1 is a singleout
and 0 otherwise.
"""
import random
import pandas as pd
import numpy as np

def single_outs(data: pd.DataFrame) -> tuple[pd.DataFrame, list]:
    """It takes a dataframe and returns a new dataframe with a new column called 'single_out'
    that is 1 if the row is a single out and 0 otherwise, based on select quasi-identifiers

    :param data: the dataframe you want to create the single out variable

    Returns:
        dataframe: Dataframe with single outs variable
        list: selected quasi-identifiers
    """
    set_data = []
    set_key_vars = []
    # select 10% of attributes as quasi-identifiers
    random.seed(42)
    for i in range(0, 5):
        key_vars = random.choices(data.columns[:-1], k=int(0.1*len(data.columns)))
        set_key_vars.append(key_vars)

        if (i > 0) & len(set_key_vars) != i+1:
            set_key_vars = [x for x in set_key_vars if x != key_vars]
            key_vars = random.choices(data.columns, k=int(0.1*len(data.columns)))
            set_key_vars.append(key_vars)
            i -= 1

    for key_vars in set_key_vars:
        k = data.groupby(key_vars)[key_vars[0]].transform(len)

        data_copy = data.copy()
        data_copy['single_out'] = None
        data_copy['single_out'] = np.where(k == 1, 1, 0)

        set_data.append(data_copy)


    return set_data, set_key_vars
