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
            bag_files,
            topics=None
    ):
        if not isinstance(bag_files, list):
            raise TypeError('bag files are not given as a list')
        if not isinstance(topics, list):
            raise TypeError('topics are not given as a list')

        self.msg_lists = []
        for bag_path in bag_files:
            msg_list = self._read_bag(bag_path, topics)
            self.msg_lists.append(msg_list)

        self.n_topics = len(topics)

    @staticmethod
    def _read_bag(file_path, topics):
        msg_list = []
        bag = rosbag.Bag(file_path)
        for topic, msg, t in bag.read_messages(topics):
            msg_list.append(msg)
        bag.close()
        return msg_list

    def get_unit(self, record_idx, unit_idx):
        """A unit is defined as a segment containing one of each topic msgs.

        :param record_idx: int The index of the bag file
        :param unit_idx: int The index of the unit to get, should be within the range [0, n_unit)
        :return: list A list containing all topics' messages in a unit
        """
        if record_idx < 0 or record_idx >= len(self.msg_lists):
            raise ValueError('record idx {} is not legal'.format(record_idx))

        msg_list = self.msg_lists[record_idx]
        n_unit = len(msg_list) // self.n_topics
        if unit_idx < 0 or unit_idx >= n_unit:
            raise ValueError('unit idx is not legal')

        output_list = []
        for i in range(self.n_topics):
            msg_i = msg_list[unit_idx * self.n_topics + i]
            pose_msg = PoseStamped()
            pose_msg.pose = msg_i.pose
            output_list.append(pose_msg)
        return output_list

