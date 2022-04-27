"""Apply sinthetisation with SMOTE
This script will apply SMOTE technique in the single out cases.
"""
# %%
#!/usr/bin/env python
import functools
import os
from os import sep
import threading
import argparse
import pika
import pandas as pd
import numpy as np
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import LabelEncoder
from kanon import single_outs
from record_linkage import record_linkage
from modeling import evaluate_model


parser = argparse.ArgumentParser(description='Master Example')
parser.add_argument('--input_folder', type=str, help='Input folder', default="./input")
args = parser.parse_args()

def ack_message(ch, delivery_tag, work_sucess):
    if ch.is_open:
        if work_sucess:
            print("[x] Done")
            ch.basic_ack(delivery_tag)
        else:
            ch.basic_reject(delivery_tag, requeue=False)
            print("[x] Rejected")
    else:
        # Channel is already closed, so we can't ACK this message;
        # log and/or do something that makes sense for your app in this case.
        pass


def apply_record_linkage(oversample_data, original_data, keys):
    """Apply record linkage and calculate the percentage of re-identification

    Args:
        oversample_data (pd.Dataframe): oversampled data
        original_data (pd.Dataframe): original dataframe
        keys (list): list of quasi-identifiers

    Returns:
        _type_: _description_
    """
    oversample_singleouts = oversample_data[oversample_data['single_out']==1]
    original_singleouts = original_data[original_data['single_out']==1]
    potential_matches = record_linkage(oversample_singleouts, original_singleouts, keys)

    # get acceptable score (QIs match at least 50%)
    acceptable_score = potential_matches[potential_matches['Score'] >= \
        0.5*potential_matches['Score'].max()]
    level_1_acceptable_score = acceptable_score.groupby(['level_1'])['level_0'].size()
    per = ((1/level_1_acceptable_score.min()) * 100) / len(oversample_data)

    # get max score (all QIs match)
    max_score = potential_matches[potential_matches['Score'] == len(keys)]
    # find original single outs with an unique match in oversampled data - 100% match
    level_1_max_score = max_score.groupby(['level_1'])['level_0'].size()
    per_100 = (len(level_1_max_score[level_1_max_score == 1]) * 100) / len(oversample_data)

    return potential_matches, per, per_100

# 
def all_work(file):
    print(f'{args.input_folder}/{file}')
    data = pd.read_csv(f'{args.input_folder}/{file}')
    # data = data.apply(lambda x: x.apply(lambda y: round(y, 3) if abs(decimal.Decimal(f'{y}').as_tuple().exponent) >= 3 else y))
    data = data.apply(lambda x: round(x, 3) if x.dtype=='float64' else x)
    # apply LabelEncoder beacause of smote
    data = data.apply(LabelEncoder().fit_transform)
    set_data, set_key_vars = single_outs(data)
    # key_vars_idx = [data.columns.get_loc(c) for c in key_vars if c in data]
    knn = [1, 2, 3, 4, 5]
    # percentage of majority class
    ratios = [0.1, 0.2, 0.3, 0.5, 0.7]
    for idx, key_vars in enumerate(set_key_vars):
        dt = set_data[idx]
        print(f'QIS ITERATIONS: {idx}')
        if len(set_key_vars) > 5:
            raise Exception("ARDEU!!!!!")
        # interpolation of cases where nr of single outs is less than 40% 
        thr_condition = len(dt[dt['single_out']==1]) <= 0.4 * len(dt)   
        if thr_condition and len(dt[dt['single_out']!=0]):   
            for nn in knn:
                print(f'NUMBER OF KNN: {nn}')
                for ratio in ratios:
                    print(f'NUMER OF RATIO: {ratio}')
                    smote = SMOTE(random_state=42,
                                k_neighbors=nn,
                                sampling_strategy=ratio)
                    # fit predictor and target variable
                    X = dt[dt.columns[:-1]]
                    y = dt.iloc[:, -1]
                    x_smote, y_smote = smote.fit_resample(X, y)

                    # add single out to apply record linkage
                    x_smote['single_out'] = y_smote
                    # remove original single outs from oversample
                    oversample = x_smote.copy()
                    oversample = oversample.drop(dt[dt['single_out']==1].index).reset_index(drop=True)

                    matches, percentage, percentage_100 = apply_record_linkage(
                        oversample,
                        dt,
                        key_vars)

                    # prepare data to modeling
                    X, y = oversample.iloc[:, :-2], oversample.iloc[:, -2]
                    # predictive performance
                    validation, test = evaluate_model(
                        X,
                        y,
                        nn,
                        percentage,
                        percentage_100)

                    # save oversample, potential matches, validation and test results
                    try:
                        output_folder_ov = (
                            f'{os.path.dirname(os.getcwd())}{sep}output{sep}'
                            f'oversampled{sep}{file.split(".")[0]}')
                        output_folder_rl = (
                            f'{os.path.dirname(os.getcwd())}{sep}output{sep}'
                            f'record_linkage{sep}{file.split(".")[0]}')
                        output_folder_val = (
                            f'{os.path.dirname(os.getcwd())}{sep}output{sep}modeling'
                            f'{sep}oversampled{sep}validation{sep}{file.split(".")[0]}')
                        output_folder_test = (
                            f'{os.path.dirname(os.getcwd())}{sep}output{sep}modeling{sep}'
                            f'oversampled{sep}test{sep}{file.split(".")[0]}')    
                        if not os.path.exists(output_folder_ov) | os.path.exists(output_folder_rl) | \
                            os.path.exists(output_folder_val) | os.path.exists(output_folder_test):
                            os.makedirs(output_folder_ov)
                            os.makedirs(output_folder_rl)
                            os.makedirs(output_folder_val)
                            os.makedirs(output_folder_test)

                        # save oversampled data
                        oversample.to_csv(
                            f'{output_folder_ov}{sep}oversample_QI{idx}_knn{nn}_per{ratio}.csv',
                            index=False)
                        # save record linkage results
                        matches.to_csv(
                            f'{output_folder_rl}{sep}potential_matches_QI{idx}_knn{nn}_per{ratio}.csv',
                            index=False)

                        np.save(
                            f'{output_folder_val}{sep}validation_QI{idx}_knn{nn}_per{ratio}.npy',
                                validation)
                        np.save(
                            f'{output_folder_test}{sep}test_QI{idx}_knn{nn}_per{ratio}.npy', test)

                    except Exception as exc:
                        raise exc


def do_work(conn, ch, delivery_tag, body):
    msg = body.decode('utf-8')
    work_sucess = all_work(msg)
    cb = functools.partial(ack_message, ch, delivery_tag, work_sucess)
    conn.add_callback_threadsafe(cb)


def on_message(ch, method_frame, _header_frame, body, args):
    (conn, thrds) = args
    delivery_tag = method_frame.delivery_tag
    t = threading.Thread(target=do_work, args=(conn, ch, delivery_tag, body))
    t.start()
    thrds.append(t)


#credentials = pika.PlainCredentials('guest', 'guest')
connection = pika.BlockingConnection(pika.ConnectionParameters(host='localhost', heartbeat=5))
channel = connection.channel()
channel.queue_declare(queue='task_queue', durable=True, arguments={"dead-letter-exchange":"dlx"})
print(' [*] Waiting for messages. To exit press CTRL+C')

channel.basic_qos(prefetch_count=1)

threads = []
on_message_callback = functools.partial(on_message, args=(connection, threads))
channel.basic_consume('task_queue', on_message_callback)

try:
    channel.start_consuming()
except KeyboardInterrupt:
    channel.stop_consuming()

# Wait for all to complete
for thread in threads:
    thread.join()

connection.close()


# find . -name ".DS_Store" -delete
# python3 code/task.py  --input_folder "input"
# python3 code/worker_smote.py  --input_folder "input"
