import os
import sys
import numpy as np
import json
from mpi4py import MPI

from baselines import logger
from baselines.common import set_global_seeds
from baselines.common.mpi_moments import mpi_moments
from baselines.her.rollout import RolloutWorker
from baselines.her.util import mpi_fork

import experiments.config as config

import envs


def shapes_to_dims(input_shapes):
    return {key: np.prod(val) for key, val in input_shapes.items()}


def mpi_average(value):
    if value == []:
        value = [0.]
    if not isinstance(value, list):
        value = [value]
    return mpi_moments(np.array(value))[0]


def train(policy, rollout_worker, evaluator,
          n_epochs, n_test_rollouts, n_cycles, n_batches, policy_save_interval,
          save_policies, **kwargs):
    rank = MPI.COMM_WORLD.Get_rank()

    latest_policy_path = os.path.join(logger.get_dir(), 'policy_latest.pkl')
    best_policy_path = os.path.join(logger.get_dir(), 'policy_best.pkl')
    periodic_policy_path = os.path.join(logger.get_dir(), 'policy_{}.pkl')

    logger.info("Training...")
    best_success_rate = -1
    for epoch in range(n_epochs):
        # train
        rollout_worker.clear_history()
        for _ in range(n_cycles):
            episode = rollout_worker.generate_rollouts()
            policy.store_episode(episode)
            for _ in range(n_batches):
                policy.train()
            policy.update_target_net()

        # test
        evaluator.clear_history()
        for _ in range(n_test_rollouts):
            evaluator.generate_rollouts()

        # record [logs
        logger.record_tabular('epoch', epoch)
        for key, val in evaluator.logs('test'):
            logger.record_tabular(key, mpi_average(val))
        for key, val in rollout_worker.logs('train'):
            logger.record_tabular(key, mpi_average(val))
        for key, val in policy.logs():
            logger.record_tabular(key, mpi_average(val))

        if rank == 0:
            logger.dump_tabular()

        # save the policy if it's better than the previous ones
        success_rate = mpi_average(evaluator.current_success_rate())
        if rank == 0 and success_rate >= best_success_rate and save_policies:
            best_success_rate = success_rate
            logger.info(
                'New best success rate: {}. Saving policy to {} ...'.format(best_success_rate, best_policy_path))
            evaluator.save_policy(best_policy_path)
            evaluator.save_policy(latest_policy_path)
        if rank == 0 and policy_save_interval > 0 and epoch % policy_save_interval == 0 and save_policies:
            policy_path = periodic_policy_path.format(epoch)
            logger.info('Saving periodic policy to {} ...'.format(policy_path))
            evaluator.save_policy(policy_path)

        # make sure that different threads have different seeds
        local_uniform = np.random.uniform(size=(1,))
        root_uniform = local_uniform.copy()
        MPI.COMM_WORLD.Bcast(root_uniform, root=0)
        if rank != 0:
            assert local_uniform[0] != root_uniform[0]


def run_task(vv, log_dir=None, exp_name=None):
    override_params = {}
    # Fork for multi-CPU MPI implementation.
    if vv['num_cpu'] > 1:
        whoami = mpi_fork(vv['num_cpu'])
        if whoami == 'parent':
            sys.exit(0)
        import baselines.common.tf_util as U
        U.single_threaded_session().__enter__()
    rank = MPI.COMM_WORLD.Get_rank()


    log_dir='/media/part/cmu_ri/deep/deep_RL/data/local/square2d-debug/square2d_debug_2018_06_17/' #hack for now, fix later

    # Configure logging
    if rank == 0:
        if log_dir or logger.get_dir() is None:
            from pathlib import Path
            logger.configure(dir=log_dir, exp_name=exp_name)
    else:
        if log_dir or logger.get_dir() is None:
            from pathlib import Path
            logger.configure(dir=log_dir, exp_name=exp_name)

    logdir = logger.get_dir()
    #logdir = ''# a quick hack, fix later
    assert logdir is not None
    os.makedirs(logdir, exist_ok=True)

    # Seed everything.
    rank_seed = vv['seed'] + 1000000 * rank
    set_global_seeds(rank_seed)

    # Prepare params.
    params = config.DEFAULT_PARAMS
    params['env_name'] = vv['env_name']
    params['replay_strategy'] = vv['replay_strategy']
    params['replay_sample_strategy'] = vv['replay_sample_strategy']
    params['reward_type'] = vv['reward_type']
    params['replay_k'] = vv['replay_k']
    if vv['network'] == 'fc':
        params['network_class'] = 'baselines.her.actor_critic:ActorCritic'
    elif vv['network'] == 'cnn_fc':
        params['network_class'] = 'baselines.her.cnn_actor_critic:CNNActorCritic'

    if vv['env_name'] in config.DEFAULT_ENV_PARAMS:
        params.update(config.DEFAULT_ENV_PARAMS[vv['env_name']])  # merge env-specific parameters in
    params.update(**override_params)  # makes it possible to override any parameter
    with open(os.path.join(logger.get_dir(), 'variant.json'), 'w') as f:
        json.dump(params, f)
    params = config.prepare_params(params)
    config.log_params(params, logger=logger)

    shapes = config.configure_shapes(params)
    dims = shapes_to_dims(shapes)
    policy = config.configure_ddpg(dims=dims, shapes=shapes, params=params, clip_return=vv['clip_return'])

    rollout_params = {
        'exploit': False,
        'use_target_net': False,
        'use_demo_states': True,
        'compute_Q': False,
        'T': params['T'],
    }

    eval_params = {
        'exploit': True,
        'use_target_net': params['test_with_polyak'],
        'use_demo_states': False,
        'compute_Q': True,
        'T': params['T'],
    }

    for name in ['T', 'rollout_batch_size', 'gamma', 'noise_eps', 'random_eps']:
        rollout_params[name] = params[name]
        eval_params[name] = params[name]

    rollout_worker = RolloutWorker(params['make_env'], policy, dims, logger, **rollout_params)
    rollout_worker.seed(rank_seed)

    evaluator = RolloutWorker(params['make_env'], policy, dims, logger, **eval_params)
    evaluator.seed(rank_seed)

    train(
        logdir=logdir, policy=policy, rollout_worker=rollout_worker,
        evaluator=evaluator, n_epochs=vv['n_epochs'], n_test_rollouts=params['n_test_rollouts'],
        n_cycles=params['n_cycles'], n_batches=params['n_batches'],
        policy_save_interval=vv['policy_save_interval'], save_policies=vv['save_policies'])

# @click.option('--logdir', type=str, default='/media/xingyu/ExtraDrive1/data_vHER/',
#               help='the path to where logs and policy pickles should go. If not specified, creates a folder in /tmp/')
# @click.option('--exp_name', type=str, default='push')
