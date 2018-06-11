# Created by Xingyu Lin, 18/03/2018                                                                                  
import gym
import envs
from envs.gym_robotics_visual import utils
import scipy.misc


def test_image_rendering():
    env = gym.make(test_env)
    obs = env.reset()
    rgbd_img = env.env.get_image_obs(depth=True)
    rgb_img = rgbd_img[:, :, :3]
    dep_img = rgbd_img[:, :, -1]
    scipy.misc.imsave('./test/obs_rgb.jpg', rgb_img)
    scipy.misc.imsave('./test/obs_d.jpg', dep_img)

    goal_rgb_img, goal_dep_img = utils.separate_img(obs['desired_goal'])
    scipy.misc.imsave('./test/goal_rgb.jpg', goal_rgb_img)
    scipy.misc.imsave('./test/goal_d.jpg', goal_dep_img)

    # while True:
    #     action = env.action_space.sample()
    #     obs, _, _, _ = env.step(action)
    #     env.render()

def test_visualization():
    env = gym.make(test_env)
    env.reset()
    while True:
        action = env.action_space.sample()
        obs, _, _, _ = env.step(action)
        env.render()


if __name__ == '__main__':
    test_env = 'VisualFetchSlide-v0'
    test_image_rendering()
    # test_visualization()
