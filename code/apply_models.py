""" 
This script will modeling data
"""
import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from modeling import evaluate_model


def save_results(file, args, validation, test):
    """Create a folder if dooes't exist and save results

    Args:
        file (string): file name
        args (args): command line arguments
        validation (dictionary): validation of training results
        test (dictionary): test results
    """
    output_folder_val = (
        f'{args.output_folder}/validation')
    output_folder_test = (
        f'{args.output_folder}/test')
    if not os.path.exists(output_folder_val) |\
            os.path.exists(output_folder_test):
        os.makedirs(output_folder_val)
        os.makedirs(output_folder_test)
    np.save(
        f'{output_folder_val}/{file.replace("csv", "npy")}', validation)
    np.save(
        f'{output_folder_test}/{file.replace("csv", "npy")}', test)


def simple_modeling(file, args):
    """Apply predictive performance.

    Args:
        file (string): input file

    Raises:
        Exception: failed to apply smote when single outs
        class is great than non single outs.
        exc: failed to writing the results.
    """
    try:
        print(f'{args.input_folder}/{file}')
        data = pd.read_csv(f'{args.input_folder}/{file}')
        data = data.apply(LabelEncoder().fit_transform)
        # prepare data to modeling
        X, y = data.iloc[:, :-1], data.iloc[:, -1]

        # predictive performance
        validation, test = evaluate_model(X, y)
    except:
        return False

    # save validation and test results
    try:
       save_results(file, args, validation, test)

    except Exception as exc:
        raise exc


def modeling_singleouts(file, args):
    """Apply predictive performance.

    Args:
        file (string): input file

    Raises:
        Exception: failed to apply smote when single outs
        class is great than non single outs.
        exc: failed to writing the results.
    """
    print(f'{args.input_folder}/{file}')
    data = pd.read_csv(f'{args.input_folder}/{file}')
    data = data.apply(LabelEncoder().fit_transform)
    # prepare data to modeling
    X, y = data.iloc[:, :-2], data.iloc[:, -2]
    if y.value_counts().nunique() != 1:
        # predictive performance
        validation, test = evaluate_model(X, y)
        # save validation and test results
        try:
            if validation:
                save_results(file, args, validation, test)

        except Exception as exc:
            raise exc


    


