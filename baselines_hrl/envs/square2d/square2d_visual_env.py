from mujoco_py import load_model_from_path, MjSim, MjViewer
from gym import spaces
from gym import Env, GoalEnv
from gym.utils import seeding
import os
import numpy as np
from numpy.random import random
from os import path
import cv2 as cv

class Square2dVisualEnv(GoalEnv):
    # TODO make this into GoalEnv
    def __init__(self, model_path='./square2d.xml', distance_threshold=2200, frame_skip=2,
                 horizon=100):

        if model_path.startswith("/"):
            fullpath = model_path
        else:
            fullpath = os.path.join(os.path.dirname(__file__), model_path)
        if not path.exists(fullpath):
            raise IOError("File %s does not exist" % fullpath)

        self.model = load_model_from_path(fullpath)
        self.seed()

        self.sim = MjSim(self.model)
        self.data = self.sim.data
        self.viewer = None
        self.distance_threshold = distance_threshold
        self.frame_skip = frame_skip
        self.reward_type = 'sparse'
        self.horizon = horizon
        self.time_step = 0
        obs = self.reset()
        self.goal_observation = None
        self.action_space = spaces.Box(-1., 1., shape=(2,), dtype='float32')
        self.observation_space = spaces.Dict(dict(
            desired_goal=spaces.Box(-np.inf, np.inf, shape=obs['achieved_goal'].shape, dtype='float32'),
            achieved_goal=spaces.Box(-np.inf, np.inf, shape=obs['achieved_goal'].shape, dtype='float32'),
            observation=spaces.Box(-np.inf, np.inf, shape=obs['observation'].shape, dtype='float32'),
        ))

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def reset(self):
        goal_location = self._sample_goal().copy()
        #goal_location = np.array([0.3,0.3])
        self.get_image_of_goal_observation(goal_location)
        self.sim.forward()
        data = self.render()
        #data = data[::-1, :, :].flatten()
        self.goal_observation = data

        #print(self.get_goal_location())
        #print(self.get_ball_location())
        #cv.imshow('display', data)
        #v.waitKey(1)
        self.set_ball_location([0., 0.])
        self.set_goal_location(goal_location)
        self.sim.forward()
        self.time_step = 0
        return self.get_current_observation()

    def _get_viewer(self):
        if self.viewer is None:
            self.viewer = MjViewer(self.sim)
            self.viewer_setup()
        return self.viewer

    def viewer_setup(self):

        self.viewer.cam.lookat[0] = 0.0  # x,y,z offset from the object (works if trackbodyid=-1)
        self.viewer.cam.lookat[1] = 0.0
        self.viewer.cam.lookat[2] = 0.0
        self.viewer.cam.elevation = -90  # camera rotation around the axis in the plane going through the frame origin (if 0 you just see a line)
        self.viewer.cam.azimuth = 90
        self.viewer.cam.distance = 1.5

    def set_goal_location(self, goalPos):
        # goal = [xLoc, yLoc]
        self.sim.data.qpos[0] = goalPos[0]
        self.sim.data.qpos[1] = goalPos[1]

    def set_ball_location(self, ballPos):

        self.sim.data.qpos[2] = ballPos[0]
        self.sim.data.qpos[3] = ballPos[1]

    def set_distance_threshold(self, distance_threshold):
        self.distance_threshold = distance_threshold

    def set_frame_skip(self, frame_skip):
        self.frame_skip = frame_skip

    def get_frame_skip(self):
        return self.frame_skip

    def get_distance_threshold(self):
        return self.distance_threshold

    def get_ball_location(self):
        return self.sim.data.qpos[2:4]

    def get_goal_location(self):
        return self.sim.data.qpos[0:2]

    def get_ball_velocity(self):
        return self.sim.data.qvel[2:4]

    def send_control_command(self, xDirectionControl, yDirectionControl):

        self.sim.data.ctrl[0] = xDirectionControl
        self.sim.data.ctrl[1] = yDirectionControl

    def get_current_observation(self):
        # obs = np.concatenate([self.get_goal_location(), self.get_ball_location(), self.get_ball_velocity()]).ravel()
        data = self.render()
        #data = data[::-1, :, :]
        #data = data[::-1, :, :].flatten()
        #print(data[::-1, :, :].flatten().shape)
        #obs = np.concatenate([data, self.get_ball_velocity()]).ravel()
        #bs = data.flatten()
        #desired_goal = self.goal_observation.flatten()
        #achieved_goal = data.flatten()
        obs = data
        desired_goal = self.goal_observation
        achieved_goal = data
        #print(achieved_goal.shape)
        #print(desired_goal.shape)
        #print(obs.shape)
        #input("--------------------")
        
        return {
            'observation': obs.copy(),
            'achieved_goal': achieved_goal.copy(),
            'desired_goal': desired_goal.copy()

        }

    def get_image_of_goal_observation(self, goalPos):

        if not goalPos[0]:
            goalPos[0] = self.sim.data.qpos[0]

        if not goalPos[1]:
            goalPos[1] = self.sim.data.qpos[1]

        self.sim.data.qpos[0] = goalPos[0]
        self.sim.data.qpos[1] = goalPos[1]
        self.sim.data.qpos[2] = goalPos[0]
        self.sim.data.qpos[3] = goalPos[1]
        #image = self.sim.render(camera_name='camera1', width=200, height=200, depth=False)
        #self.render()

    def do_simulation(self, ctrl, n_frames):
        self.send_control_command(ctrl[0], ctrl[1])
        for _ in range(n_frames):
            self.take_step()

    def step(self, ctrl):
        ctrl = np.clip(ctrl, -1., 1.)
        self.do_simulation(ctrl, self.frame_skip)
        obs = self.get_current_observation()
        info = {
            'is_success': self._is_success(obs['achieved_goal'], self.goal_observation),
        }
        reward = self.compute_reward(obs['achieved_goal'], obs['desired_goal'], {})
        done = (reward == 1.0)
        self.time_step += 1
        if self.time_step >= self.horizon:
            done = True
        return obs, reward, done, info

    def take_step(self):
        self.sim.step()

    def render(self, mode='human'):
        #self._get_viewer().render() 
        #print(dir(self.sim))
        #image = self.get_image_of_goal_observation(self._sample_goal())
        image = self.sim.render(camera_name='camera1', width=100, height=100, depth=False)
        #print(dir(self.sim.render))
        #input("-------------")
        image = image[::-1, :, :]
        return image

    def compute_reward(self, achieved_goal, desired_goal, info):
        # Compute distance between goal and the achieved goal.
        #print(achieved_goal.shape)
        #print(desired_goal.shape)
        d = np.linalg.norm(achieved_goal - desired_goal, axis=-1)
        #return d
        #d = np.array([d])
        #print(d.shape)
        if self.reward_type == 'sparse':
            return -(d > self.distance_threshold).astype(np.float32)
        else:
            return -d

    def _is_success(self, achieved_goal, desired_goal):
        d = np.linalg.norm(achieved_goal.flatten() - desired_goal.flatten(), axis=-1)
        return (d < self.distance_threshold).astype(np.float32)

    def _sample_goal(self):
        return random((2,)) - 0.5
