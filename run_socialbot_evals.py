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
COMMAND = "OMP_NUM_THREADS=4 python3 sandbox/finetuning/runs/pg_test.py --algo hippo_random_p -trainsnn -random_init -d 0.99 -tpi 10 -p 5 -eps 0.1 -minp 2 -latdim 6 -mb -sbl -msb -lr 3e-4 --logdir " + log_dir

num_gpus = 2
max_worker_num = num_gpus * 2
itrs = (1, 2, 3) # so that it's repeated 3 times each
#env_names = ("FetchReach-v1", "FetchPush-v1", "FetchSlide-v1", "FetchPickAndPlace-v1")
envs = ["GoalTask", "KickBall"]#"FetchPush-v1", "FetchSlide-v1", "FetchPickAndPlace-v1"]
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

    #product = itertools.product(*(env_names, itrs))
    for i in range(3):
        for env in envs:
    #for task_count, values in enumerate(product):
        #env_name, _ = values
            command = "%s --env_name %s" % (COMMAND, env)
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
