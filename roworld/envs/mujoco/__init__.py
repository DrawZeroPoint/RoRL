import gym
from gym.envs.registration import register
import logging

LOGGER = logging.getLogger(__name__)
REGISTERED = False


def register_mujoco_envs():
    global REGISTERED
    if REGISTERED:
        return
    REGISTERED = True
    LOGGER.info("Registering RoWorld mujoco gym environments")
    register_roworld_envs()


def register_roworld_envs():
    register_canonical_envs()


def register_canonical_envs():
    register(
        id='MZ25ReachXYZEnv-v0',
        entry_point='roworld.envs.mujoco.nachi_mz25_xyz'
                    '.mz25_reach:MZ25ReachXYZEnv',
        kwargs={
            'hide_goal_markers': False,
            'norm_order': 2,
        },
    )
    register(
        id='UR5eReachXYZEnv-v0',
        entry_point='roworld.envs.mujoco.ur5e_xyz'
                    '.ur5e_reach:UR5eReachXYZEnv',
        kwargs={
            'hide_goal_markers': False,
            'norm_order': 2,
        },
    )
    register(
        id='UR5ePushXYZEnv-v0',
        entry_point='roworld.envs.mujoco.ur5e_xyz'
                    '.ur5e_push_and_reach:UR5ePushAndReachXYZEnv',
        kwargs={
            'hide_goal_markers': False,
            'norm_order': 2,
            'reward_type': 'puck_distance',
        },
    )
    # register(
    #     id='SawyerPush-v0',
    #     entry_point='multiworld.envs.mujoco.sawyer_xyz'
    #                 '.sawyer_push_nips:SawyerPushAndReachXYEasyEnv',
    #     kwargs=dict(
    #         force_puck_in_goal_space=False,
    #         mocap_low=(-0.1, 0.55, 0.0),
    #         mocap_high=(0.1, 0.65, 0.5),
    #         hand_goal_low=(-0.1, 0.55),
    #         hand_goal_high=(0.1, 0.65),
    #         puck_goal_low=(-0.15, 0.5),
    #         puck_goal_high=(0.15, 0.7),
    #
    #         hide_goal=True,
    #         reward_info=dict(
    #             type="state_distance",
    #         ),
    #     )
    # )
    #
    # register(
    #     id='SawyerPickup-v0',
    #     entry_point='multiworld.envs.mujoco.sawyer_xyz'
    #                 '.sawyer_pick_and_place:SawyerPickAndPlaceEnvYZ',
    #     kwargs=dict(
    #         hand_low=(-0.1, 0.55, 0.05),
    #         hand_high=(0.0, 0.65, 0.13),
    #         action_scale=0.02,
    #         hide_goal_markers=True,
    #         num_goals_presampled=1000,
    #         p_obj_in_hand=.75,
    #     )
    # )


def create_image_48_sawyer_reach_xy_env_v1():
    from multiworld.core.image_env import ImageEnv
    from multiworld.envs.mujoco.cameras import sawyer_xyz_reacher_camera_v0

    wrapped_env = gym.make('SawyerReachXYEnv-v1')
    return ImageEnv(
        wrapped_env,
        48,
        init_camera=sawyer_xyz_reacher_camera_v0,
        transpose=True,
        normalize=True,
    )


def create_image_84_sawyer_reach_xy_env_v1():
    from multiworld.core.image_env import ImageEnv
    from multiworld.envs.mujoco.cameras import sawyer_xyz_reacher_camera_v0

    wrapped_env = gym.make('SawyerReachXYEnv-v1')
    return ImageEnv(
        wrapped_env,
        84,
        init_camera=sawyer_xyz_reacher_camera_v0,
        transpose=True,
        normalize=True,
    )


def create_image_48_sawyer_push_and_reach_arena_env_v0():
    from multiworld.core.image_env import ImageEnv
    from multiworld.envs.mujoco.cameras import sawyer_pusher_camera_upright_v2

    wrapped_env = gym.make('SawyerPushAndReachArenaEnv-v0')
    return ImageEnv(
        wrapped_env,
        48,
        init_camera=sawyer_pusher_camera_upright_v2,
        transpose=True,
        normalize=True,
    )


def create_image_48_sawyer_push_and_reach_arena_env_reset_free_v0():
    from multiworld.core.image_env import ImageEnv
    from multiworld.envs.mujoco.cameras import sawyer_pusher_camera_upright_v2

    wrapped_env = gym.make('SawyerPushAndReachArenaResetFreeEnv-v0')
    return ImageEnv(
        wrapped_env,
        48,
        init_camera=sawyer_pusher_camera_upright_v2,
        transpose=True,
        normalize=True,
    )


def create_image_48_sawyer_door_hook_reset_free_v1():
    from multiworld.core.image_env import ImageEnv
    from multiworld.envs.mujoco.cameras import sawyer_door_env_camera_v0
    import os.path
    import numpy as np
    goal_path = os.path.join(
        os.path.dirname(os.path.realpath(__file__)),
        'goals/door_goals.npy'
    )
    goals = np.load(goal_path).item()
    return ImageEnv(
        wrapped_env=gym.make('SawyerDoorHookResetFreeEnv-v1'),
        imsize=48,
        init_camera=sawyer_door_env_camera_v0,
        transpose=True,
        normalize=True,
        presampled_goals=goals,
    )


def create_image_48_sawyer_pickup_easy_v0():
    from multiworld.core.image_env import ImageEnv
    from multiworld.envs.mujoco.cameras import sawyer_pick_and_place_camera
    import os.path
    import numpy as np
    goal_path = os.path.join(
        os.path.dirname(os.path.realpath(__file__)),
        'goals/pickup_goals.npy'
    )
    goals = np.load(goal_path).item()
    return ImageEnv(
        wrapped_env=gym.make('SawyerPickupEnvYZEasyFewGoals-v0'),
        imsize=48,
        init_camera=sawyer_pick_and_place_camera,
        transpose=True,
        normalize=True,
        presampled_goals=goals,
    )
