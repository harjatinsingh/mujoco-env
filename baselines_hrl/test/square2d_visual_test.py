# Created by Xingyu Lin, 03/06/2018
import gym
import envs
import cv2 as cv
import scipy.misc
from pprint import pprint

from envs.square2d import Square2dVisualSimpleEnv
import time
import numpy as np
if __name__ == '__main__':
    test_env = 'Square2dVisualSimple-v0'
    #env = gym.make(test_env)
    env = Square2dVisualSimpleEnv(horizon=10000)
    for i in range(5):
        env.reset()
        done = False
        time_count = 0
        while not done:
            time_count += 1
            action = env.action_space.sample()
            #env.set_goal_location([0.3,0.3])
            obs, reward, done, _ = env.step(action)
            #print(env.get_goal_location())
            #print(len(obs))
            #print(reward.shape)
            #print(done.shape)
            #print(reward)
            img = env.render()
            #print(img.shape)
            cv.imshow('display', obs['observation'])
            #cv.imshow('display', obs['desired_goal'])
            cv.waitKey(1)
            #time.sleep(1000000)
        time.sleep(3)