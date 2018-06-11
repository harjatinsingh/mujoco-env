# Created by Xingyu Lin, 03/06/2018
import gym
import envs
import scipy.misc

from envs.square2d import Square2dEnv
import time
import numpy as np
if __name__ == '__main__':
    test_env = 'Square2d-v0'
    # env = gym.make(test_env)
    env = Square2dEnv(horizon=1000)
    for i in range(5):
        env.reset()
        done = False
        time_count = 0
        while not done:
            time_count += 1
            action = env.action_space.sample()
            obs, reward, done, _ = env.step(action)
            env.render()
        time.sleep(3)