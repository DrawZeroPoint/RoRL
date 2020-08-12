import abc
import numpy as np
import mujoco_py

from roworld.core.serializable import Serializable
from roworld.envs.mujoco.mujoco_env import MujocoEnv

import copy


class UR5eMocapBase(MujocoEnv, Serializable, metaclass=abc.ABCMeta):
    """
    Provides some commonly-shared functions for Sawyer Mujoco envs that use
    mocap for XYZ control.
    """
    mocap_low = np.array([-0.5, -0.5, 0.8])
    mocap_high = np.array([0.5, 0.5, 1.2])

    def __init__(self, model_name, frame_skip=50):
        # Increase frame skip makes the simulation slower
        MujocoEnv.__init__(self, model_name, frame_skip=frame_skip)
        self.reset_mocap_welds()

    def get_endeff_pos(self):
        return self.data.get_body_xpos('hand').copy()

    def get_gripper_pos(self):
        # FIXME qpos dim = DOF, set this according to robot type
        return np.array([self.data.qpos[6]])

    def get_env_state(self):
        joint_state = self.sim.get_state()
        mocap_state = self.data.mocap_pos, self.data.mocap_quat
        state = joint_state, mocap_state
        return copy.deepcopy(state)

    def set_env_state(self, state):
        joint_state, mocap_state = state
        self.sim.set_state(joint_state)
        mocap_pos, mocap_quat = mocap_state
        self.data.set_mocap_pos('mocap', mocap_pos)
        self.data.set_mocap_quat('mocap', mocap_quat)
        self.sim.forward()

    def __getstate__(self):
        state = super().__getstate__()
        return {**state, 'env_state': self.get_env_state()}

    def __setstate__(self, state):
        super().__setstate__(state)
        self.set_env_state(state['env_state'])

    def reset_mocap_welds(self):
        """Resets the mocap welds that we use for actuation."""
        sim = self.sim
        if sim.model.nmocap > 0 and sim.model.eq_data is not None:
            for i in range(sim.model.eq_data.shape[0]):
                if sim.model.eq_type[i] == mujoco_py.const.EQ_WELD:
                    # quaternion
                    sim.model.eq_data[i, :] = np.array(
                        [0., 0., 0., 1., 0., 0., 0.])
        sim.forward()


class UR5eXYZEnv(UR5eMocapBase, metaclass=abc.ABCMeta):
    # FIXME change hand low and high (workspace) according to env,
    #  coord is in world frame
    def __init__(
            self,
            *args,
            hand_low=(-0.35, -0.35, 0.9),
            hand_high=(0.35, 0.3, 1.2),
            mocap_low=None,
            mocap_high=None,
            action_scale=2./100,
            **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.action_scale = action_scale
        self.hand_low = np.array(hand_low)
        self.hand_high = np.array(hand_high)
        if mocap_low is None:
            mocap_low = hand_low
        if mocap_high is None:
            mocap_high = hand_high
        self.mocap_low = np.hstack(mocap_low)
        self.mocap_high = np.hstack(mocap_high)

    def set_xyz_action(self, action):
        action = np.clip(action, -1, 1)
        pos_delta = action * self.action_scale
        new_mocap_pos = self.data.mocap_pos + pos_delta[None]
        new_mocap_pos[0, :] = np.clip(
            new_mocap_pos[0, :],
            self.mocap_low,
            self.mocap_high,
        )
        self.data.set_mocap_pos('mocap', new_mocap_pos)
        # FIXME notice the quaternion
        self.data.set_mocap_quat('mocap', np.array([1, 0, 1, 0]))

    def set_xy_action(self, xy_action, fixed_z):
        delta_z = fixed_z - self.data.mocap_pos[0, 2]
        xyz_action = np.hstack((xy_action, delta_z))
        self.set_xyz_action(xyz_action)
