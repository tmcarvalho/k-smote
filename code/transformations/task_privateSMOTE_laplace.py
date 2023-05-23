"""Task manager
This script will add the tasks in the queue
"""
#!/usr/bin/env python
import argparse
import os
import pika

parser = argparse.ArgumentParser(description='Master Example')
parser.add_argument('--input_folder', type=str, help='Input folder', default="./input")
args = parser.parse_args()


def put_file_queue(ch, file_name):
    """Add files to the queue

    Args:
        ch (_type_): channel of the queue
        file_name (string): name of file
    """
    ch.basic_publish(
        exchange='',
        routing_key='task_queue_privatesmote_laplace',
        body=file_name,
        properties=pika.BasicProperties(
            delivery_mode=2,  # make message persistent
        ))


credentials = pika.PlainCredentials('guest', 'guest')
connection = pika.BlockingConnection(
    pika.ConnectionParameters(host='localhost', credentials=credentials, heartbeat=5))
channel = connection.channel()

channel.exchange_declare(exchange='dlx', exchange_type='direct')

channel.queue_declare(queue='task_queue_privatesmote_laplace', durable=True, arguments={"dead-letter-exchange": "dlx"})
dl_queue = channel.queue_declare(queue='dl')

channel.queue_bind(exchange='dlx', routing_key='task_queue_privatesmote_laplace', queue=dl_queue.method.queue)

knn = [1,3,5]
per = [1,2,3]
# epislon = [1, 2, 3, 5]

files = files = next(os.walk(args.input_folder))[2]
for file in files:
    f = int(file.split('.')[0])
    if f not in [0,1,3,13,23,28,34,36,40,48,54,66,87]:
        if f in [68]:
            print(file)
            for idx in range(5):
                for k in knn:
                    for p in per:
                        # for ep in epislon:
                        # files.append(f'{file.split(".")[0]}_privateSMOTE_QI{idx}_knn{k}_per{p}')
                        print(f'ds{file.split(".")[0]}_privateSMOTE_QI{idx}_knn{k}_per{p}')
                        put_file_queue(channel, f'ds{file.split(".")[0]}_privateSMOTE_QI{idx}_knn{k}_per{p}')
                        
connection.close()