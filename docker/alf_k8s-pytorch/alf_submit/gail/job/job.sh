#!/bin/bash

# init envrionment, should always keep this
source /opt/set_env.sh

# set PYTHONPATH
export PYTHONPATH=$WORKING_PATH/hippo:$WORKING_PATH/hippo/SocialRobotCustom/python

# mujoco_py path
export PYTHONPATH=/opt/usr/local/lib/python3.6/dist-packages:${PYTHONPATH}
export MUJOCO_PY_MUJOCO_PATH=/opt/.mujoco/mujoco200_linux
export MUJOCO_PY_MJKEY_PATH=/opt/.mujoco/mjkey.txt
export LD_LIBRARY_PATH=/opt/.mujoco/mujoco200_linux/bin:${LD_LIBRARY_PATH}

# do not need a xserver to render, just disable it
unset DISPLAY

# run training, headless
# sometimes the network is unstable so we customize timeout
pip install --upgrade pip
#cd $WORKING_PATH/hippo; pip install --default-timeout=100 -e .; pip install --default-timeout=100 -e gym_0.12.5; pip install --default-timeout=500 -e SocialRobotCustom; pip install tensorboard-logger; source ~/.bashrc;
cd $WORKING_PATH/hippo; pip install --default-timeout=500 -e .; pip install --default-timeout=500 -e gym_0.12.5; pip install tensorboard-logger; source ~/.bashrc;
python3 run_gpu_idler.py & xvfb-run python3 run_fetch_evals.py /job_data/hippo_run_logs
#python3 run_gpu_idler.py & xvfb-run python3 run_grid_search_multi.py /job_data/hippo_run_logs
