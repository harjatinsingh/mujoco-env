# Created by Xingyu Lin, 03/06/2018
import gym
import envs
import cv2 as cv
import scipy.misc
import pickle
from pprint import pprint


from envs.square2d import Square2dVisualEnv
import time
import numpy as np
if __name__ == '__main__':
    test_env = 'Square2dVisual-v0'
    #env = gym.make(test_env)
    policy_file = '/media/part/cmu_ri/deep/deep_RL/data/local/square2d-debug/square2d_debug_2018_06_15/policy_0.pkl'
    with open(policy_file, 'rb') as f:
        policy = pickle.load(f)
    #env_name = policy.info['env_name']


    env = Square2dVisualEnv(horizon=10000)
    for i in range(5):
        obs = env.reset()
        done = False
        time_count = 0
        while not done:
            o = obs['observation']
            ag = obs['achieved_goal']
            g = obs['desired_goal']
            
            action = policy.get_actions(o,ag,g)
            #action = env.action_space.sample()
            #print(action)
            
            time_count += 1
            #action = env.action_space.sample()
            #env.set_goal_location([0.3,0.3])
            obs, reward, done, info = env.step(action)
            #print(info['is_success'])
            print(reward)
            if reward == -0.0:
                print(reward)
                input("-----------------")
            
            print(reward)    
            #print(env.get_goal_location())
            #print(len(obs))
            #print(reward.shape)
            #print(done.shape)
            #print(reward)
            img = env.render()
            #print(img.shape)
            cv.imshow('display', img)
            #cv.imshow('display', obs['desired_goal'])
            cv.waitKey(100)
            #time.sleep(1000000)
        time.sleep(3)