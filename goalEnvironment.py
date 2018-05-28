from mujoco_py import load_model_from_path, MjSim, MjViewer
import os
import numpy as np
from pprint import pprint
from os import path



class goalEnvironment():

	 def __init__(self, model_path, distance_threshold = 1e-1, frame_skip = 1):

	 	if model_path.startswith("/"):
	 		fullpath = model_path
	 	else:
	 		fullpath = os.path.join(os.path.dirname(__file__), "assets", model_path)
	 	if not path.exists(fullpath):
	 		raise IOError("File %s does not exist" % fullpath)

	 	self.model = load_model_from_path(fullpath)
	 	self.sim = MjSim(self.model)
	 	self.data = self.sim.data
	 	self.viewer = MjViewer(self.sim)
	 	self.distance_threshold = distance_threshold
	 	self.frame_skip = frame_skip

	 def viewer_setup(self):

	    self.viewer.cam.lookat[0] = 0.0         # x,y,z offset from the object (works if trackbodyid=-1)
	    self.viewer.cam.lookat[1] = 0.0
	    self.viewer.cam.lookat[2] = 0.0
	    self.viewer.cam.elevation = -90           # camera rotation around the axis in the plane going through the frame origin (if 0 you just see a line)
	    self.viewer.cam.azimuth = 90        
	    self.viewer.cam.distance = 1.5

	 def set_goal_location(self,xLoc,yLoc):
	 	
	 	self.sim.data.qpos[0] = xLoc
	 	self.sim.data.qpos[1] = yLoc

	 def set_ball_location(self,xLoc,yLoc):
	 	
	 	self.sim.data.qpos[2] = xLoc
	 	self.sim.data.qpos[3] = yLoc	

	 def set_distance_threshold(self,distance_threshold):
	 	self.distance_threshold = distance_threshold

	 def set_frame_skip(self,frame_skip):
	 	self.frame_skip = frame_skip

	 def get_frame_skip(self):
	 	return (self.frame_skip)

	 def get_distance_threshold(self):
	 	return (self.distance_threshold) 

	 def get_ball_location(self):
	 	return (self.sim.data.qpos[2:4])	

	 def get_goal_location(self):
	 	return (self.sim.data.qpos[0:2])

	 def get_ball_velocity(self):
	 	return (self.sim.data.qvel[2:4])

	 def send_control_command(self,xDirectionControl,yDirectionControl):

	 	self.sim.data.ctrl[0] = xDirectionControl
	 	self.sim.data.ctrl[1] = yDirectionControl

	 def get_current_observation(self):
	 	return np.concatenate([self.get_goal_location(), self.get_ball_location(), self.get_ball_velocity()]).ravel()

	 def get_image_of_goal_observation(self, xLoc = None, yLoc = None):
	 	
	 	if not xLoc:
	 		xLoc = self.sim.data.qpos[0]
	 	
	 	if not yLoc:
	 		yLoc = self.sim.data.qpos[1]	
	 	
	 	self.sim.data.qpos[0] = xLoc
	 	self.sim.data.qpos[1] = yLoc
	 	self.sim.data.qpos[2] = xLoc
	 	self.sim.data.qpos[3] = yLoc

	 	self.render_view()

	 def do_simulation(self, ctrl, n_frames):

	 	self.send_control_command(ctrl[0],ctrl[1])
	 	for _ in range(n_frames):
	 		self.take_step()
	

	 def step(self, ctrl):
	 	
	 	self.do_simulation(ctrl, self.frame_skip)
	 	obs = self.get_current_observation()
	 	reward = self.get_reward()
	 	done = (reward == 1.0)
	 	return obs, reward, done

	 def take_step(self):
	 	self.sim.step()

	 def render_view(self):
	 	self.viewer.render()

	 def get_reward(self):

	 	dist = np.linalg.norm(self.sim.data.qpos[0:2] - self.sim.data.qpos[2:4], axis=-1)	
	 	return (dist < self.distance_threshold).astype(np.float32)
