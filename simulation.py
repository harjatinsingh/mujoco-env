from mujoco_py import load_model_from_path, MjSim, MjViewer
import os
import numpy as np
from pprint import pprint
from os import path
from goalEnvironment import goalEnvironment


env = goalEnvironment("/home/frc-vision/Desktop/git/mujoco-env/xmls/goal.xml")
env.viewer_setup()
env.set_goal_location(0.3,0.3)
env.set_frame_skip(2)

while True:
	#env.send_control_command(0.002,0.002)
	ctrl = np.asarray([0.002,0.002])
	obs, reward, done = env.step(ctrl)
	env.render_view()