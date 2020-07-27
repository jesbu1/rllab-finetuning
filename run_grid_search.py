import itertools
import subprocess
import os
COMMAND = "OMP_NUM_THREADS=2 python3 sandbox/finetuning/runs/pg_test.py --env_name FetchReach-v1 --algo hippo_random_p -trainsnn -random_init -d 0.98 -mb -sbl -msb -lr 3e-4"

skill_dims = (6, 9)
clippings = (0.05, 0.1, 0.15)
time_commitment_ranges = ((5, 15), (2, 10), (3, 7))
train_pi_iters = (10, 30)
product = itertools.product(*(skill_dims, clippings, time_commitment_ranges, train_pi_iters))
for param_config in product:
    skill_dim, epsilon, time_commitment, tpi = param_config
    command = "%s -p %d -minp %d -latdim %d -eps %0.2f -tpi %d" % (COMMAND, time_commitment[1], time_commitment[0], skill_dim, epsilon, tpi)
    print(command + " &&")
    #process = subprocess.Popen(command.split(), stdout=subprocess.PIPE)
    #output, error = process.communicate()
    #stream = os.popen(command)
    #print(stream.read())
    #print(output, error)
