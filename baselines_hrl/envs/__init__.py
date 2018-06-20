# Created by Xingyu Lin, 18/03/2018                                                                                  
from gym.envs.registration import register

# square2d registration
register(
    id='Square2d-v0',
    entry_point='envs.square2d:Square2dEnv',
    max_episode_steps=1000
)

register(
    id='Square2dVisual-v0',
    entry_point='envs.square2d:Square2dVisualEnv',
    max_episode_steps=1000
)


register(
     id='Square2dSimple-v0',
     entry_point='envs.square2d:Square2dSimpleEnv',
     max_episode_steps=1000
 )


register(
     id='Square2dVisualSimple-v0',
     entry_point='envs.square2d:Square2dVisualSimpleEnv',
     max_episode_steps=1000
 )
# Reference: Visual Gym registration
# ---------------------------

def _merge(a, b):
    a.update(b)
    return a


for reward_type in ['sparse', 'dense']:
    suffix = 'Dense' if reward_type == 'dense' else ''
    kwargs = {
        'reward_type': reward_type,
    }

    # Fetch
    register(
        id='VisualFetchSlide{}-v0'.format(suffix),
        entry_point='envs.gym_robotics_visual:FetchSlideEnv',
        kwargs=kwargs,
        max_episode_steps=50,
    )

    register(
        id='VisualFetchPickAndPlace{}-v0'.format(suffix),
        entry_point='envs.gym_robotics_visual:FetchPickAndPlaceEnv',
        kwargs=kwargs,
        max_episode_steps=50,
    )

    register(
        id='VisualFetchReach{}-v0'.format(suffix),
        entry_point='envs.gym_robotics_visual:FetchReachEnv',
        kwargs=kwargs,
        max_episode_steps=50,
    )

    register(
        id='VisualFetchPush{}-v0'.format(suffix),
        entry_point='envs.gym_robotics_visual:FetchPushEnv',
        kwargs=kwargs,
        max_episode_steps=50,
    )

    # Hand
    register(
        id='VisualHandReach{}-v0'.format(suffix),
        entry_point='envs.gym_robotics_visual:HandReachEnv',
        kwargs=kwargs,
        max_episode_steps=50,
    )

    register(
        id='VisualHandManipulateBlockRotateZ{}-v0'.format(suffix),
        entry_point='envs.gym_robotics_visual:HandBlockEnv',
        kwargs=_merge({'target_position': 'ignore', 'target_rotation': 'z'}, kwargs),
        max_episode_steps=100,
    )

    register(
        id='VisualHandManipulateBlockRotateParallel{}-v0'.format(suffix),
        entry_point='envs.gym_robotics_visual:HandBlockEnv',
        kwargs=_merge({'target_position': 'ignore', 'target_rotation': 'parallel'}, kwargs),
        max_episode_steps=100,
    )

    register(
        id='VisualHandManipulateBlockRotateXYZ{}-v0'.format(suffix),
        entry_point='envs.gym_robotics_visual:HandBlockEnv',
        kwargs=_merge({'target_position': 'ignore', 'target_rotation': 'xyz'}, kwargs),
        max_episode_steps=100,
    )

    register(
        id='VisualHandManipulateBlockFull{}-v0'.format(suffix),
        entry_point='envs.gym_robotics_visual:HandBlockEnv',
        kwargs=_merge({'target_position': 'random', 'target_rotation': 'xyz'}, kwargs),
        max_episode_steps=100,
    )

    # Alias for "Full"
    register(
        id='VisualHandManipulateBlock{}-v0'.format(suffix),
        entry_point='envs.gym_robotics_visual:HandBlockEnv',
        kwargs=_merge({'target_position': 'random', 'target_rotation': 'xyz'}, kwargs),
        max_episode_steps=100,
    )

    register(
        id='VisualHandManipulateEggRotate{}-v0'.format(suffix),
        entry_point='envs.gym_robotics_visual:HandEggEnv',
        kwargs=_merge({'target_position': 'ignore', 'target_rotation': 'xyz'}, kwargs),
        max_episode_steps=100,
    )

    register(
        id='VisualHandManipulateEggFull{}-v0'.format(suffix),
        entry_point='envs.gym_robotics_visual:HandEggEnv',
        kwargs=_merge({'target_position': 'random', 'target_rotation': 'xyz'}, kwargs),
        max_episode_steps=100,
    )

    # Alias for "Full"
    register(
        id='VisualHandManipulateEgg{}-v0'.format(suffix),
        entry_point='envs.gym_robotics_visual:HandEggEnv',
        kwargs=_merge({'target_position': 'random', 'target_rotation': 'xyz'}, kwargs),
        max_episode_steps=100,
    )

    register(
        id='VisualHandManipulatePenRotate{}-v0'.format(suffix),
        entry_point='envs.gym_robotics_visual:HandPenEnv',
        kwargs=_merge({'target_position': 'ignore', 'target_rotation': 'xyz'}, kwargs),
        max_episode_steps=100,
    )

    register(
        id='VisualHandManipulatePenFull{}-v0'.format(suffix),
        entry_point='envs.gym_robotics_visual:HandPenEnv',
        kwargs=_merge({'target_position': 'random', 'target_rotation': 'xyz'}, kwargs),
        max_episode_steps=100,
    )

    # Alias for "Full"
    register(
        id='VisualHandManipulatePen{}-v0'.format(suffix),
        entry_point='envs.gym_robotics_visual:HandPenEnv',
        kwargs=_merge({'target_position': 'random', 'target_rotation': 'xyz'}, kwargs),
        max_episode_steps=100,
    )
