import gym
from gym.envs.registration import register
import logging

LOGGER = logging.getLogger(__name__)
REGISTERED = False


def register_webots_envs():
    global REGISTERED
    if REGISTERED:
        return
    REGISTERED = True
    LOGGER.info("Registering RoWorld Webots gym environments")
    register_canonical_envs()


def register_canonical_envs():
    register(
        id='UR5eVisualReachEnv-v0',
        entry_point='roworld.envs.webots.ur5e_xyz'
                    '.ur5e_reach:UR5eReachXYZEnv',
        kwargs={
            'hide_goal_markers': False,
            'norm_order': 2,
        },
    )


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
