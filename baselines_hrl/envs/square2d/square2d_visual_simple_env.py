from mujoco_py import load_model_from_path, MjSim, MjViewer
from gym import spaces
from gym import Env, GoalEnv
from gym.utils import seeding
import os
import numpy as np
from numpy.random import random
from os import path
from envs.square2d.square2d_visual_env import Square2dVisualEnv


class Square2dVisualSimpleEnv(Square2dVisualEnv):
    def __init__(self, *args, **kwargs):
        super(Square2dVisualSimpleEnv, self).__init__(*args, **kwargs)
        if 'horizon' not in kwargs:
            self.horizon = 100

    '''        
    def reset(self):
        self.set_ball_location([0., 0.])
        self.set_goal_location(self._sample_goal())
        self.sim.forward()
        self.time_step = 0
        return self.get_current_observation()
    '''
    def step(self, ctrl):
        ctrl = np.clip(ctrl, -1., 1.)
        ctrl = ctrl/100
        ballPos = self.get_ball_location()
        self.set_ball_location(np.asarray(ballPos) + np.asarray(ctrl))
        self.sim.forward()
        self.take_step()
        obs = self.get_current_observation()
        info = {
            'is_success': self._is_success(obs['achieved_goal'], obs['desired_goal']),
        }
        reward = self.compute_reward(obs['achieved_goal'], obs['desired_goal'], {})
        done = (reward == 1.0)
        self.time_step += 1
        if self.time_step >= self.horizon:
            done = True
        return obs, reward, done, info


    @staticmethod
    def _sample_goal():
        return (random((2,)) - 0.5)/3