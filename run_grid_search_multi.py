import itertools
import random
import subprocess
import os
from absl import logging, flags, app
from multiprocessing import Queue, Manager
from pathos import multiprocessing
import traceback
import time
import sys
log_dir = sys.argv[1]
COMMAND = "OMP_NUM_THREADS=4 python3 sandbox/finetuning/runs/pg_test.py --env_name GoalTask --algo hippo_random_p -trainsnn -random_init -d 0.99 -mb -sbl -msb -lr 3e-4 --logdir " + log_dir

num_gpus = 2
max_worker_num = num_gpus * 1
skill_dims = (4, 8)
clippings = (0.05, 0.1)
time_commitment_ranges = ((2, 5), (3, 7))
train_pi_iters = (10, 40, 80)
product = itertools.product(*(skill_dims, clippings, time_commitment_ranges, train_pi_iters))
def _init_device_queue(max_worker_num):
    m = Manager()
    device_queue = m.Queue()
    for i in range(max_worker_num):
        idx = i % num_gpus
        device_queue.put(idx)
    return device_queue

def run():
    """Run trainings with all possible parameter combinations in
    the configured space.
    """

    process_pool = multiprocessing.Pool(
        processes=max_worker_num, maxtasksperchild=1)
    device_queue = _init_device_queue(max_worker_num)

    product = itertools.product(*(skill_dims, clippings, time_commitment_ranges, train_pi_iters))
    for i in range(3):
        for task_count, values in enumerate(product):
            skill_dim, epsilon, time_commitment, tpi = values
            command = "%s -p %d -minp %d -latdim %d -eps %0.2f -tpi %d" % (COMMAND, time_commitment[1], time_commitment[0], skill_dim, epsilon, tpi)
            process_pool.apply_async(
                func=_worker,
                args=[command, device_queue],
                error_callback=lambda e: logging.error(e))

    process_pool.close()
    process_pool.join()

def _worker(command, device_queue):
    # sleep for random seconds to avoid crowded launching
    try:
        time.sleep(random.uniform(0, 60))

        device = device_queue.get()

        logging.set_verbosity(logging.INFO)

        logging.info("command %s" % command)
        os.system(command)

        device_queue.put(device)
    except Exception as e:
        logging.info(traceback.format_exc())
        raise e

run()
