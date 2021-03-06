import numpy as np


def make_sample_her_transitions(replay_strategy, replay_k, reward_fun, replay_sample_strategy, reward_type):
    """Creates a sample function that can be used for HER experience replay.

    Args:
        replay_strategy (in ['future', 'none']): the HER replay strategy; if set to 'none',
            regular DDPG experience replay is used
        replay_k (int): the ratio between HER replays and regular replays (e.g. k = 4 -> 4 times
            as many HER replays as regular replays are used)
        reward_fun (function): function to re-compute the reward with substituted goals
        replay_sample_strategy: 'random' or 'prioritized', if not random, use prioritized replay based on Bellman error
    """
    if replay_strategy == 'future':
        future_p = 1 - (1. / (1 + replay_k))
    elif replay_strategy == 'only_fake':
        future_p = 1
    else:  # 'replay_strategy' == 'none'
        future_p = 0

    def _sample_her_transitions(episode_batch, batch_size_in_transitions):
        """episode_batch is {key: array(buffer_size x T x dim_key)}
        """
        T = episode_batch['u'].shape[1]
        rollout_batch_size = episode_batch['u'].shape[0]
        batch_size = batch_size_in_transitions

        # Select which episodes and time steps to use.
        episode_idxs = np.random.randint(0, rollout_batch_size, batch_size)
        t_samples = np.random.randint(T, size=batch_size)
        transitions = {key: episode_batch[key][episode_idxs, t_samples].copy()
                       for key in episode_batch.keys()}

        # Select future time indexes proportional with probability future_p. These
        # will be used for HER replay by substituting in future goals.
        her_indexes = np.where(np.random.uniform(size=batch_size) < future_p)
        future_offset = np.random.uniform(size=batch_size) * (T - t_samples)

        future_offset = future_offset.astype(int)
        future_t = (t_samples + 1 + future_offset)[her_indexes]

        # Replace goal with achieved goal but only for the previously-selected
        # HER transitions (as defined by her_indexes). For the other transitions,
        # keep the original goal.
        future_ag = episode_batch['ag'][episode_idxs[her_indexes], future_t]
        transitions['g'][her_indexes] = future_ag

        # Reconstruct info dictionary for reward  computation.
        info = {}
        for key, value in transitions.items():
            if key.startswith('info_'):
                info[key.replace('info_', '')] = value

        # Re-compute reward since we may have substituted the goal.
        if reward_type == 'reward_func':
            reward_params = {k: transitions[k] for k in ['ag_2', 'g']}
            reward_params['info'] = info
            #print(reward_params['ag_2'].shape)
            #rint(reward_params['g'].shape)
            #print(reward_params['info']['is_success'].shape)
            transitions['r'] = reward_fun(**reward_params)

            #input("--------------")
        else:
            # reward_params = {k: transitions[k] for k in ['ag_2', 'g']}
            # reward_params['info'] = info
            # transitions['r'] = reward_fun(**reward_params)
            # print('original rewards', transitions['r'][:20])
            rewards = np.zeros((batch_size,))
            rewards[her_indexes] = -(np.array(future_offset[her_indexes]) != 0).astype(int)
            transitions['r'] = rewards

            # print('rewards:', rewards[:20])
            # print('future_offset', future_offset[:20])
            # print('her_indexes', her_indexes[:20])
            # exit()

        #print(batch_size)
        #print(transitions['r'].shape)
        #input("--------------")    
        transitions = {k: transitions[k].reshape(batch_size, *transitions[k].shape[1:])
                       for k in transitions.keys()}
        # print('g', transitions['g'])
        # print('o', transitions['o'])
        # print('r', np.mean(transitions['r']))
        # exit()
        assert (transitions['u'].shape[0] == batch_size_in_transitions)

        return transitions

    return _sample_her_transitions
