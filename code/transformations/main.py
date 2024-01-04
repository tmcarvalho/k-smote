import subprocess
import time
import re
import pandas as pd
import ast
import os
import functools
import threading
import argparse
import json
import pika
import psutil
import GPUtil

# Argument parsing
parser = argparse.ArgumentParser(description='Master Example')
parser.add_argument('--type', type=str, help='transformation type', default="none")
args = parser.parse_args()

# List to store resource usage information
# resource_usage_data = Manager().list()
# print(resource_usage_data)
# Record start time
start_time = time.time()

# List to store resource usage information
resource_usage_data = []

def ack_message(ch, delivery_tag, work_success):
    """Acknowledge message

    Args:
        ch (_type_): channel of the queue
        delivery_tag (_type_): _description_
        work_success (_type_): _description_
    """
    if ch.is_open:
        if work_success:
            print("[x] Done")
            ch.basic_ack(delivery_tag)
        else:
            ch.basic_reject(delivery_tag, requeue=False)
            print("[x] Rejected")
    else:
        # Channel is already closed, so we can't ACK this message;
        # log and/or do something that makes sense for your app in this case.
        pass

def measure_resource_consumption(file):
    # Measure CPU usage
    cpu_percent = psutil.cpu_percent(interval=1)
    # Measure GPU usage
    gpu_percent = GPUtil.getGPUs()[0].load * 100

    # Measure RAM usage
    ram_percent = psutil.virtual_memory().percent

    # Measure temperature
    # cpu_temperature = psutil.sensors_temperatures()["cpu_thermal"][0]

    # Measure GPU temperature
    gpu_temperature = GPUtil.getGPUs()[0].temperature

    # Measure elapsed time
    elapsed_time = time.time() - start_time

    # Store resource usage in a dictionary
    resource_usage = {
        'file': file,
        'elapsed_time': elapsed_time,
        'cpu_percent': cpu_percent,
        'gpu_percent': gpu_percent,
        'ram_percent': ram_percent,
        # 'cpu_temperature': cpu_temperature.current,
        'gpu_temperature': gpu_temperature
    }

    # Print and store resource usage
    print(resource_usage)
    resource_usage_data.append(resource_usage)


def do_work(conn, ch, delivery_tag, body):
    msg = body.decode('utf-8')

    # Measure resource consumption before running the script
    measure_resource_consumption(msg)

    if 'PrivateSMOTE' in args.type:
        f = list(map(int, re.findall(r'\d+', msg.split('_')[0])))
        list_key_vars = pd.read_csv('list_key_vars.csv')
        set_key_vars = ast.literal_eval(
            list_key_vars.loc[list_key_vars['ds']==f[0], 'set_key_vars'].values[0])

        keys_nr = list(map(int, re.findall(r'\d+', msg.split('_')[2])))[0]
        print(keys_nr)
        keys = set_key_vars[keys_nr]
        keys_str = ','.join(keys)
        
        # Use subprocess.Popen to run the script asynchronously
        process = subprocess.Popen(['python3', 'code/transformations/PrivateSMOTE_force_laplace.py', '--input_file', msg, '--key_vars', keys_str], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, shell=False)
    
    if args.type == 'SDV':
        process = subprocess.Popen(['python3', 'code/transformations/deep_learning.py', '--input_file', msg], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    
    if args.type == 'Synthcity':
        process = subprocess.Popen(['python3', 'code/transformations/deep_learning.py', '--input_file', msg], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

    # Poll the subprocess while it's running
    while process.poll() is None:
        # Measure resource consumption during the script execution
        measure_resource_consumption(msg)
        #time.sleep(1)  # Adjust the interval as needed
        # The subprocess is still running
        print("Subprocess is still running...")

    # Wait for the process to finish and capture output
    stdout, stderr = process.communicate()

    # Print captured output and return code
    print("Standard Output:")
    print(stdout)

    print("\nStandard Error:")
    print(stderr)

    # After subprocess completion, store resource usage data in a JSON file
    with open(f'output/comp_costs/{msg}.json', 'w') as json_file:
        json.dump(resource_usage_data, json_file, indent=2)

    os.system('find . -name "__pycache__" -type d -exec rm -rf "{}" +')
    os.system('find . -name "*.pyc"| xargs rm -f "{}"')
    cb = functools.partial(ack_message, ch, delivery_tag, process)
    conn.add_callback_threadsafe(cb)

def on_message(ch, method_frame, _header_frame, body, args):
    (conn, thrds) = args
    delivery_tag = method_frame.delivery_tag
    t = threading.Thread(target=do_work, args=(conn, ch, delivery_tag, body))
    t.start()
    thrds.append(t)


# Connection setup
connection = pika.BlockingConnection(pika.ConnectionParameters(host='localhost', heartbeat=0))
channel = connection.channel()
channel.queue_declare(queue='task_queue_transf', durable=True, arguments={"dead-letter-exchange": "dlx"})
print(' [*] Waiting for messages. To exit press CTRL+C')

channel.basic_qos(prefetch_count=1)

threads = []
on_message_callback = functools.partial(on_message, args=(connection, threads))
channel.basic_consume('task_queue_transf', on_message_callback)

try:
    channel.start_consuming()
except KeyboardInterrupt:
    channel.stop_consuming()

# Wait for all to complete
for thread in threads:
    thread.join()

connection.close()

# find . -name ".DS_Store" -delete
# python3 code/transformations/task_transf.py  --input_folder "original" --type "PrivateSMOTE"
# python3 code/transformations/main.py --type "PrivateSMOTE"