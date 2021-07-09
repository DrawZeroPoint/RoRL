import numpy as np

import rosbag
from geometry_msgs.msg import PoseStamped

from rorlkit.torch.data_management import transform


def sd_pose(pose):
    """Standardize the input pose to the 4x4 homogeneous transformation
    matrix in special Euclidean group SE(3).

    :param pose:
    :return: transformation matrix
    """
    if isinstance(pose, np.ndarray):
        if pose.ndim == 1 and pose.size == 7:
            t = pose[:3]
            q = pose[3:]
            tm = transform.translation_matrix(t)
            rm = transform.quaternion_matrix(q)
            # make sure to let tm left product rm
            return np.dot(tm, rm)
        elif pose.ndim == 1 and pose.size == 6:
            t = pose[:3]
            rpy = pose[3:]
            tm = transform.translation_matrix(t)
            rm = transform.euler_matrix(rpy[0], rpy[1], rpy[2])
            return np.dot(tm, rm)
        elif pose.shape == (4, 4):
            return pose
        else:
            raise NotImplementedError
    elif isinstance(pose, list):
        return sd_pose(np.array(pose))
    elif isinstance(pose, PoseStamped):
        p = pose.pose.position
        o = pose.pose.orientation
        return sd_pose(np.array([p.x, p.y, p.z, o.x, o.y, o.z, o.w]))
    else:
        raise NotImplementedError


def pose_msg_to_feature(pose_msg):
    assert isinstance(pose_msg, PoseStamped)
    pose_mat = sd_pose(pose_msg)
    # features: x, y, z, Rx1, Rx2 Rx3, Ry1, Ry2, Ry3
    feature = np.array([
        pose_mat[0, -1], pose_mat[1, -1], pose_mat[2, -1],
        pose_mat[0, 0], pose_mat[1, 0], pose_mat[2, 0],
        pose_mat[0, 1], pose_mat[1, 1], pose_mat[2, 1],
    ])
    return feature


def get_identity_feature():
    """No translation and identity rotation"""
    return np.array([0, 0, 0, 1, 0, 0, 0, 1, 0])


class PosesGetter(object):
    def __init__(
            self,
            file_name,
            topics=None
    ):
        if not isinstance(topics, list):
            raise TypeError('topics are not given as a list')
        self._msg_list = []
        self._read_bag(file_name, topics)
        self._n_topics = len(topics)
        self.n_unit = len(self._msg_list) // len(topics)

    def _read_bag(self, file_name, topics):
        bag = rosbag.Bag(file_name)
        for topic, msg, t in bag.read_messages(topics):
            self._msg_list.append(msg)
        bag.close()

    def get_unit(self, idx):
        """A unit is defined as a segment in the msg_list containing one of each topic msgs.

        :param idx: int The index of the unit to get, should be within the range [0, n_unit)
        """
        if idx < 0 or idx >= self.n_unit:
            raise ValueError('idx is not legal')

        output_list = []
        for i in range(self._n_topics):
            msg_i = self._msg_list[idx * self._n_topics + i]
            pose_msg = PoseStamped()
            pose_msg.pose = msg_i.pose
            output_list.append(pose_msg)
        return output_list

