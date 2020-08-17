from roworld.envs.mujoco import register_mujoco_envs
from roworld.envs.webots import register_webots_envs


def register_all_envs():
    register_mujoco_envs()
    register_webots_envs()
