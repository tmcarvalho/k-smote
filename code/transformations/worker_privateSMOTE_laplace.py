"""Apply sinthetisation with SMOTE
This script will apply SMOTE technique in the single out cases.
"""
# %%
#!/usr/bin/env python
import os
import functools
import threading
import argparse
import gc
import pika
from PrivateSMOTE_laplace import PrivateSMOTE_laplace_
from PrivateSMOTE_force_laplace import PrivateSMOTE_force_laplace_

#%%
parser = argparse.ArgumentParser(description='Master Example')
# parser.add_argument('--input_folder', type=str, help='Input folder', default="./input")
parser.add_argument('--type', type=str, help='PrivateSMOTE type', default="none")
args = parser.parse_args()

def ack_message(ch, delivery_tag, work_sucess):
    """Acknowledge message

    Args:
        ch (_type_): channel of the queue
        delivery_tag (_type_): _description_
        work_sucess (_type_): _description_
    """
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
        
   
# %%
def do_work(conn, ch, delivery_tag, body):
    msg = body.decode('utf-8')
    if args.type == 'PrivateSMOTE_laplace':
        work_sucess = PrivateSMOTE_laplace_(msg)
    if args.type == 'PrivateSMOTE_force_laplace':
        work_sucess = PrivateSMOTE_force_laplace_(msg)
    gc.collect()
    os.system('find . -name "__pycache__" -type d -exec rm -rf "{}" +')
    os.system('find . -name "*.pyc"| xargs rm -f "{}"')
    cb = functools.partial(ack_message, ch, delivery_tag, work_sucess)
    conn.add_callback_threadsafe(cb)


def on_message(ch, method_frame, _header_frame, body, args):
    (conn, thrds) = args
    delivery_tag = method_frame.delivery_tag
    t = threading.Thread(target=do_work, args=(conn, ch, delivery_tag, body))
    t.start()
    thrds.append(t)


#credentials = pika.PlainCredentials('guest', 'guest')
connection = pika.BlockingConnection(pika.ConnectionParameters(host='localhost', heartbeat=0
))
channel = connection.channel()
channel.queue_declare(queue='task_queue_privatesmote_force_laplace', durable=True, arguments={"dead-letter-exchange":"dlx"})
print(' [*] Waiting for messages. To exit press CTRL+C')

channel.basic_qos(prefetch_count=1)

threads = []
on_message_callback = functools.partial(on_message, args=(connection, threads))
channel.basic_consume('task_queue_privatesmote_force_laplace', on_message_callback)

try:
    channel.start_consuming()
except KeyboardInterrupt:
    channel.stop_consuming()

# Wait for all to complete
for thread in threads:
    thread.join()

connection.close()


# find . -name ".DS_Store" -delete
# python3 code/transformations/task_privateSMOTE_laplace.py  --input_folder "original" 
# python3 code/transformations/worker_privateSMOTE_laplace.py --type "PrivateSMOTE_laplace"
# python3 code/transformations/worker_privateSMOTE_laplace.py --type "PrivateSMOTE_force_laplace"