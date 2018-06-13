from copy import deepcopy
import numpy as np
import json
import os
import gym

from baselines import logger
from baselines.her.ddpg import DDPG
from baselines.her.her import make_sample_her_transitions

DEFAULT_ENV_PARAMS = {
    'FetchReach-v0': {
        'n_cycles': 10,
    },
}

DEFAULT_PARAMS = {
    # env
    'max_u': 1.,  # max absolute value of actions on different coordinates
    # ddpg
    'layers': 3,  # number of layers in the critic/actor networks
    'hidden': 256,  # number of neurons in each hidden layers
    'network_class': 'baselines.her.actor_critic:ActorCritic',
    'Q_lr': 0.001,  # critic learning rate
    'pi_lr': 0.001,  # actor learning rate
    'buffer_size': int(1E4),  # for experience replay, was E6 changed!!! 
    'polyak': 0.95,  # polyak averaging coefficient
    'action_l2': 1.0,  # quadratic penalty on actions (before rescaling by max_u)
    'clip_obs': 200.,
    'scope': 'ddpg',  # can be tweaked for testing
    'relative_goals': False,
    # training
    'n_cycles': 50,  # per epoch was 50
    'rollout_batch_size': 2,  # per mpi thread
    'n_batches': 40,  # training batches per cycle was 40
    'batch_size': 256,  # per mpi thread, measured in transitions and reduced to even multiple of chunk_length.
    'n_test_rollouts': 10,  # number of test rollouts per epoch, each consists of rollout_batch_size rollouts
    'test_with_polyak': False,  # run test episodes with the target network
    # exploration
    'random_eps': 0.3,  # percentage of time a random action is taken
    'noise_eps': 0.2,  # std of gaussian noise added to not-completely-random actions as a percentage of max_u
    # HER
    'replay_strategy': 'future',  # supported modes: future, none
    'replay_sample_strategy': 'random',
    'reward_type': 'reward_func',
    'replay_k': 4,  # number of additional goals used for replay, only used if off_policy_data=future
    # normalization
    'norm_eps': 0.01,  # epsilon used for observation normalization
    'norm_clip': 5,  # normalized observations are cropped to this values
}

CACHED_ENVS = {}


def cached_make_env(make_env):
    """
    Only creates a new environment from the provided function if one has not yet already been
    created. This is useful here because we need to infer certain properties of the env, e.g.
    its observation and action spaces, without any intend of actually using it.
    """
    if make_env not in CACHED_ENVS:
        env = make_env()
        CACHED_ENVS[make_env] = env
    return CACHED_ENVS[make_env]


def prepare_params(kwargs):
    # DDPG params
    ddpg_params = dict()

    env_name = kwargs['env_name']

    def make_env():
        return gym.make(env_name)

    kwargs['make_env'] = make_env
    tmp_env = cached_make_env(kwargs['make_env'])
    assert hasattr(tmp_env, '_max_episode_steps')
    kwargs['T'] = tmp_env._max_episode_steps
    tmp_env.reset()
    kwargs['max_u'] = np.array(kwargs['max_u']) if type(kwargs['max_u']) == list else kwargs['max_u']
    kwargs['gamma'] = 1. - 1. / kwargs['T']
    if 'lr' in kwargs:
        kwargs['pi_lr'] = kwargs['lr']
        kwargs['Q_lr'] = kwargs['lr']
        del kwargs['lr']
    for name in ['buffer_size', 'hidden', 'layers',
                 'network_class',
                 'polyak',
                 'batch_size', 'Q_lr', 'pi_lr',
                 'norm_eps', 'norm_clip', 'max_u',
                 'action_l2', 'clip_obs', 'scope', 'relative_goals']:
        ddpg_params[name] = kwargs[name]
        kwargs['_' + name] = kwargs[name]
        del kwargs[name]
    kwargs['ddpg_params'] = ddpg_params

    return kwargs


def log_params(params, logger=logger):
    for key in sorted(params.keys()):
        logger.info('{}: {}'.format(key, params[key]))


def configure_her(params):
    env = cached_make_env(params['make_env'])
    env.reset()

    def reward_fun(ag_2, g, info):  # vectorized
        return env.compute_reward(achieved_goal=ag_2, desired_goal=g, info=info)

    # Prepare configuration for HER.
    her_params = {
        'reward_fun': reward_fun,
    }
    for name in ['replay_strategy', 'replay_k', 'replay_sample_strategy', 'reward_type']:
        her_params[name] = params[name]
        params['_' + name] = her_params[name]
        del params[name]
    sample_her_transitions = make_sample_her_transitions(**her_params)

    return sample_her_transitions


def simple_goal_subtract(a, b):
    assert a.shape == b.shape
    return a - b


def configure_ddpg(dims, shapes, params, reuse=False, use_mpi=True, clip_return=True):
    sample_her_transitions = configure_her(params)
    # Extract relevant parameters.
    gamma = params['gamma']
    rollout_batch_size = params['rollout_batch_size']
    ddpg_params = params['ddpg_params']

    input_dims = dims.copy()
    input_shapes = shapes.copy()

    # DDPG agent
    env = cached_make_env(params['make_env'])
    env.reset()
    ddpg_params.update({'input_dims': input_dims,  # agent takes an input observations
                        'image_input_shapes': input_shapes,
                        'T': params['T'],
                        'clip_pos_returns': True,  # clip positive returns
                        'clip_return': (1. / (1. - gamma)) if clip_return else np.inf,  # max abs of return
                        'rollout_batch_size': rollout_batch_size,
                        'subtract_goals': simple_goal_subtract,
                        'sample_transitions': sample_her_transitions,
                        'gamma': gamma,
                        })
    ddpg_params['info'] = {
        'env_name': params['env_name'],
    }
    policy = DDPG(reuse=reuse, **ddpg_params, use_mpi=use_mpi)
    return policy


def configure_shapes(params):
    env = cached_make_env(params['make_env'])
    env.reset()
    obs, _, _, info = env.step(env.action_space.sample())

    shapes = {
        'o': obs['observation'].shape,
        'u': env.action_space.shape,
        'g': obs['desired_goal'].shape,
    }
    for key, value in info.items():
        value = np.array(value)
        if value.ndim == 0:
            value = value.reshape(1)
        shapes['info_{}'.format(key)] = value.shape
    return shapes
