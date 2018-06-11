# Created by Xingyu Lin, 25/03/2018
import time
from chester.run_exp import run_experiment_lite, VariantGenerator
from experiments.train import run_task

if __name__ == '__main__':

    TestVisual = False
    Debug = True
    # Test FakeGoals
    if not TestVisual:
        # exp_prefix = 'FakeGoals_unique_zero'
        exp_prefix = 'square2d'
        vg = VariantGenerator()
        # vg.add('env_name', ['FetchPush-v0', 'FetchReach-v0'])
        vg.add('env_name', ['Square2d-v0'])
        # vg.add('env_name', ['FetchReach-v0', 'FetchSlide-v0', 'FetchPush-v0'])
        vg.add('network', ['fc'])
        vg.add('n_epochs', [50])

        # vg.add('replay_strategy', ['future', 'only_fake'])
        vg.add('replay_strategy', ['future'])
        vg.add('replay_sample_strategy', ['random'])  # TODO implementing 'prioritized', add to visual
        vg.add('reward_type',
               lambda replay_strategy: ['reward_func'] if replay_strategy == 'future' else ['reward_func',
                                                                                              'unique_zero'])
        # TODO add to visual
        vg.add('replay_k', lambda replay_strategy: [4] if replay_strategy == 'future' else [4])
    else:
        # Test Visual
        exp_prefix = 'VisualFetchPush'
        vg = VariantGenerator()
        vg.add('network', ['cnn_fc'])
        vg.add('env_name', ['VisualFetchPush-v0'])
        vg.add('n_epochs', [200])
        vg.add('replay_strategy', ['future'])
        vg.add('replay_k', lambda replay_strategy: [4] if replay_strategy == 'future' else [4])

    if Debug:
        exp_prefix += '_debug'
    # 'the HER replay strategy to be used. "future" uses HER, "none" disables HER.'
    vg.add('clip_return', [1])
    # 'whether or not returns should be clipped'
    vg.add('num_cpu', [1])
    vg.add('policy_save_interval', [5])
    vg.add('save_policies', [True])
    if Debug:
        vg.add('seed', [0])
    else:
        vg.add('seed', [0, 200])
    print('Number of configurations: ', len(vg.variants()))
    sub_process_popens = []
    for vv in vg.variants():
        while len(sub_process_popens) >= 2:
            sub_process_popens = [x for x in sub_process_popens if x.poll() is None]
            time.sleep(10)
        cur_popen = run_experiment_lite(
            stub_method_call=run_task,
            variant=vv,
            mode='local',
            exp_prefix=exp_prefix,
            wait_subprocess=Debug
        )
        if cur_popen is not None:
            sub_process_popens.append(cur_popen)
        if Debug:
            break
