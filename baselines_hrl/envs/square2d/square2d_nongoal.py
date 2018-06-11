# Created by Xingyu Lin, 10/06/2018                                                                                  
from mujoco_py import load_model_from_path, MjSim, MjViewer
# from gym import spaces
from rllab import spaces
from gym import Env, GoalEnv
from gym.utils import seeding
import os
import numpy as np
from os import path


class Square2dEnv(Env):
    # TODO make this into GoalEnv
    def __init__(self, model_path='./square2d.xml', distance_threshold=1e-1, frame_skip=2, goal=[0.3, 0.3],
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
        self.set_goal_location(goal)
        self.reward_type = 'dense'
        self.horizon = horizon
        self.time_step = 0
        obs = self.get_current_observation()
        self.action_space = spaces.Box(-1., 1., shape=(2,))
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=obs.shape)

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def reset(self):
        self.set_ball_location([0., 0.])
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
        assert np.linalg.norm(np.asarray(goalPos) - np.asarray([0.3, 0.3]), axis=-1) < 0.1
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
        obs = np.concatenate([self.get_goal_location(), self.get_ball_location(), self.get_ball_velocity()]).ravel()
        return obs.copy()
        # obs = np.concatenate([self.get_ball_location(), self.get_ball_velocity()]).ravel()
        # desired_goal = self.get_goal_location()
        # achieved_goal = self.get_ball_location()
        # return {
        #     'observation': obs.copy(),
        #     'achieved_goal': achieved_goal.copy(),
        #     'desired_goal': desired_goal.copy()
        #
        # }

    def get_image_of_goal_observation(self, xLoc=None, yLoc=None):

        if not xLoc:
            xLoc = self.sim.data.qpos[0]

        if not yLoc:
            yLoc = self.sim.data.qpos[1]

        self.sim.data.qpos[0] = xLoc
        self.sim.data.qpos[1] = yLoc
        self.sim.data.qpos[2] = xLoc
        self.sim.data.qpos[3] = yLoc

        self.render()

    def do_simulation(self, ctrl, n_frames):
        self.send_control_command(ctrl[0], ctrl[1])
        for _ in range(n_frames):
            self.take_step()

    def step(self, ctrl):

        if np.linalg.norm(self.get_goal_location() - [0.3, 0.3], axis=-1) > 0.1:
            print(self.get_goal_location())
            # assert False
        ctrl = np.clip(ctrl, -1., 1.)
        self.do_simulation(ctrl, self.frame_skip)
        obs = self.get_current_observation()
        info = {
        }
        reward = self.compute_reward(self.get_ball_location(), self.get_goal_location(), {})
        done = (reward == 1.0)
        self.time_step += 1
        if self.time_step >= self.horizon:
            done = True
        return obs, reward, done, info

    def take_step(self):
        self.sim.step()

    def render(self, mode='human'):
        self._get_viewer().render()

    def compute_reward(self, achieved_goal, desired_goal, info):
        # Compute distance between goal and the achieved goal.
        d = np.linalg.norm(achieved_goal - desired_goal, axis=-1)
        if self.reward_type == 'sparse':
            return -(d > self.distance_threshold).astype(np.float32)
        else:
            return -d

    def _is_success(self, achieved_goal, desired_goal):
        d = np.linalg.norm(achieved_goal - desired_goal, axis=-1)
        return (d < self.distance_threshold).astype(np.float32)

    def log_diagnostics(self, paths):
        pass

    def terminate(self):
        pass