from social_bot.tasks import GoalTask, ICubAuxiliaryTask, KickingBallTask, Reaching3D
import gym
import numpy as np
import contextlib
import socket
from gym import utils
from fasteners.process_lock import InterProcessLock
import time
DEFAULT_SOCIALBOT_PORT = 11345
class SimpleSocialBotWrapper(gym.Wrapper, utils.EzPickle):
    def __init__(self,
                 env_id,
                 task,
                 horizon=100):
        self.env_id = env_id
        self.horizon = horizon
        self.step_number = 0
        self.task = task
        self.task_map = dict(KickingBall=KickingBallTask, Goal=GoalTask) 
        #port_range = [DEFAULT_SOCIALBOT_PORT]
        #with SimpleSocialBotWrapper._get_unused_port(*port_range) as port:
        #    gym_spec = gym.spec(env_id)
        #    assert task in self.task_map.keys()
        #    if self.task_map[task] == KickingBallTask:
        #        action_cost = 0.01
        #    elif self.task_map[task] == GoalTask:
        #        action_cost = 0.01
        #    self.env = gym_spec.make(port=port, tasks=[self.task_map[task]], action_cost=action_cost)
        #self.action_space = self.env.action_space
        #self.observation_space = self.env.observation_space
        utils.EzPickle.__init__(self, env_id, task, horizon)
        
        
    @staticmethod
    @contextlib.contextmanager
    def _get_unused_port(start, end=65536, n=1):
        process_locks = []
        unused_ports = []
        #time.sleep(np.random.uniform(0, 20))
        try:
            for port in range(start, end):
                process_locks.append(
                    InterProcessLock(path='/tmp/socialbot/{}.lock'.format(port)))
                if not process_locks[-1].acquire(blocking=False):
                    process_locks[-1].lockfile.close()
                    process_locks.pop()
                    for process_lock in process_locks:
                        process_lock.release()
                    process_locks = []
                    continue
                try:
                    with contextlib.closing(socket.socket()) as sock:
                        sock.bind(('', port))
                        unused_ports.append(port)
                        if len(unused_ports) == 2:
                            break
                except socket.error:
                    for process_lock in process_locks:
                        process_lock.release()
                    process_locks = []
            if len(unused_ports) < n:
                raise socket.error("No unused port in [{}, {})".format(start, end))
            if n == 1:
                yield unused_ports[0]
            else:
                yield unused_ports
        finally:
            if process_locks:
                for process_lock in process_locks:
                    process_lock.release()
    
    def step(self, action):
        """Advance the environment by one simulation step.

        If the environment is using the contextual setting, an "is_success"
        term is added to the info_dict to specify whether the objective has
        been met.

        Parameters
        ----------
        action : array_like
            actions to be performed by the agent

        Returns
        -------
        array_like
            next observation
        float
            environmental reward
        bool
            done mask
        dict
            extra information dictionary
        """
        # Run environment update.
        obs, rew, done, info = self.env.step(action)

        # Check if the time horizon has been met.
        self.step_number += 1
        if self.step_number >= self.horizon:
            done = True
        return obs, rew, done, info

    def reset(self):
        """Reset the environment.

        If the environment is using the contextual setting, a new context is
        issued.

        Returns
        -------
        array_like
            initial observation
        """
        # Reset the step counter.
        self.step_number = 0
        ob = self.env.reset()
        return ob
    
    def __setstate__(self, d):
        out = type(self)(*d["_ezpickle_args"], **d["_ezpickle_kwargs"])
        self.__dict__.update(out.__dict__)
        self.task_map = dict(KickingBall=KickingBallTask, Goal=GoalTask) 
        port_range = [DEFAULT_SOCIALBOT_PORT]
        with SimpleSocialBotWrapper._get_unused_port(*port_range) as port:
            gym_spec = gym.spec(self.env_id)
            assert self.task in self.task_map.keys()
            if self.task_map[self.task] == KickingBallTask:
                action_cost = 0.01
            elif self.task_map[self.task] == GoalTask:
                action_cost = 0.01
            self.env = gym_spec.make(port=port, tasks=[self.task_map[self.task]], action_cost=action_cost)
        self.action_space = self.env.action_space
        self.observation_space = self.env.observation_space