# Created by Xingyu Lin, 03/06/2018
import gym
import envs
import cv2 as cv
import scipy.misc
from pprint import pprint

from envs.square2d import Square2dVisualEnv
import time
import numpy as np
if __name__ == '__main__':
    test_env = 'Square2dVisual-v0'
    #env = gym.make(test_env)
    env = Square2dVisualEnv(horizon=1000)
    for i in range(5):
        env.reset()
        done = False
        time_count = 0
        while not done:
            time_count += 1
            action = env.action_space.sample()
            obs, reward, done, _ = env.step(action)
            img = env.render()
            #print(img.shape)
            cv.imshow('display', img)
            cv.waitKey(1)
            #time.sleep(1000000)
        time.sleep(3)