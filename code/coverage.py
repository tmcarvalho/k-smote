import pandas as pd
import numpy as np
import re
import ast
import os
from sdmetrics.single_column import StatisticSimilarity, RangeCoverage, CategoryCoverage, BoundaryAdherence
from sdmetrics.column_pairs import CorrelationSimilarity
import argparse

# Parse command-line arguments
parser = argparse.ArgumentParser(description='Master Example')
parser.add_argument('--input_folder', type=str, default="none")
args = parser.parse_args()

orig_files = next(os.walk('original'))[2]

# Extract lower bound from range string and convert to number (PPT)
def is_range_format(value):
    if not isinstance(value, str):
        return False
    return '[' in value

# Function to extract lower bound from range string and convert to number
def extract_lower_bound(range_string):
    if is_range_format(range_string):
        try:
            lower_bound = range_string[1:-1].split(',')[0].strip()
            return float(lower_bound) if '.' in lower_bound else int(lower_bound)
        except (ValueError, TypeError):
            return None
    elif range_string.strip() == '*':
        return np.nan
    else:
        return None

# Function to transform columns with ranges into numbers
def transform_columns_with_ranges(df):
    for col in df.select_dtypes(include=object).columns:
        df[col] = df[col].apply(extract_lower_bound)
        df[col] = pd.to_numeric(df[col], errors='coerce')  # 'coerce' to handle non-numeric values gracefully
    return df

def coverage_(technique):
    all_stats = []
    if technique=='PPT':
        path_ppt="PPT_transformed/PPT_train/"
        transf_files = next(os.walk(path_ppt))[2]
    else:
        path="output/oversampled/"
        transf_files = next(os.walk(f'{path}{technique}'))[2]

    for transf_file in transf_files:
        for orig in orig_files:
            f = int(orig.split('.csv')[0])
            if f not in [0,1,3,13,23,28,34,36,40,48,54,66,87, 100,43]:
                if f'ds{f}' == transf_file.split('_')[0]:
                    if technique=='PPT':
                        transf_data = pd.read_csv(f'{path_ppt}{transf_file}')
                        # Transforming columns with ranges into numbers
                        transf_data = transform_columns_with_ranges(transf_data)
                    else:
                        transf_data = pd.read_csv(f'{path}{technique}/{transf_file}')
                    if technique == 'PrivateSMOTE':
                        transf_data = transf_data.loc[:, transf_data.columns[:-1]]
                    orig_data = pd.read_csv(f'original/{orig}')
                    num_columns = orig_data.select_dtypes(include=np.number).columns
                    if f == 37: # because SDV models returns real states instead of numbers as in the original data
                        transf_data.rename(columns = {'state':'code_number','phone_number':'number', 'voice_mail_plan':'voice_plan'}, inplace = True) 
                    if f == 55:
                        transf_data.rename(columns = {'state':'code_number'}, inplace = True) 

                    try:
                        corr = CorrelationSimilarity.compute(
                                real_data=orig_data[num_columns],
                                synthetic_data=transf_data[num_columns],
                                coefficient='Pearson'
                            )
                    except: # fails for columns with constant values
                        corr=np.nan
                    # print(num_columns)
                    for col in num_columns:
                        
                        stats_mean = StatisticSimilarity.compute(
                            real_data=orig_data[col],
                            synthetic_data=transf_data[col],
                            statistic='mean'
                        )
                        stats_med = StatisticSimilarity.compute(
                            real_data=orig_data[col],
                            synthetic_data=transf_data[col],
                            statistic='median'
                        )
                        stats_std = StatisticSimilarity.compute(
                            real_data=orig_data[col],
                            synthetic_data=transf_data[col],
                            statistic='std'
                        )
                        range = RangeCoverage.compute(
                            real_data=orig_data[col],
                            synthetic_data=transf_data[col]
                        )
                        boundary = BoundaryAdherence.compute(
                            real_data=orig_data[col],
                            synthetic_data=transf_data[col]
                        )
                        # print(stats)
                        stats_ = {'ds':transf_file, 
                                  'col': col,
                                  'Correlation': corr,
                                  'Statistic Similarity (Mean)': stats_mean,
                                  'Statistic Similarity (Median)': stats_med,
                                  'Statistic Similarity (Standard Deviation)': stats_std,
                                  'Boundary Adherence': boundary,
                                  'Range Coverage': range, 
                                  'Category Coverage': np.nan}
                        all_stats.append(stats_)

                    cat_columns = orig_data.select_dtypes(include=object).columns
                    for col in cat_columns:
                        cat_range = CategoryCoverage.compute(
                            real_data=orig_data[col],
                            synthetic_data=transf_data[col]
                        )
                        stats_ = {'ds':transf_file, 'col': col,
                                  'Correlation': corr,
                                  'Statistic Similarity (Mean)': stats_mean,
                                  'Statistic Similarity (Median)': stats_med,
                                  'Statistic Similarity (Standard Deviation)': stats_std,
                                  'Boundary Adherence': boundary,
                                  'Range Coverage': range, 
                                  'Category Coverage': cat_range}

                        all_stats.append(stats_)
    return all_stats

stats = coverage_(args.input_folder)       
df = pd.DataFrame(stats)
# remove wrong results (dpgan in deep learning folders) 
df = df.loc[~df.ds.str.contains('dpgan')].reset_index(drop=True)

df.to_csv(f'output_analysis/coverage_{args.input_folder}.csv', index=False)

# python3 code/coverage.py --input_folder "PPT"