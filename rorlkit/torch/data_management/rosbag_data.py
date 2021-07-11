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
            topics
    ):
        if not isinstance(bag_files, list):
            raise TypeError('bag files are not given as a list')
        if not isinstance(topics, list):
            raise TypeError('topics are not given as a list')
        self.topics = topics
        self.n_topics = len(topics)
        self.unit_key = 'n_unit'
        # This list contains dicts, one for each bag file. The keys of the dict are topics plus unit_key
        self.msg_dict_lists = []
        for bag_path in bag_files:
            msg_dict = self._read_bag(bag_path, topics)
            self.msg_dict_lists.append(msg_dict)

    def _read_bag(self, file_path, topics):
        """Read the bag file given by the file_path, and put msgs for each topic in respective key of the dict

        """
        n_unit = 2 ** 62
        msg_dict = {}
        bag = rosbag.Bag(file_path)
        for topic in topics:
            msg_list = []
            for _, msg, _ in bag.read_messages(topic):
                msg_list.append(msg)
            if len(msg_list) < n_unit:
                n_unit = len(msg_list)
            msg_dict[topic] = msg_list
        msg_dict[self.unit_key] = n_unit
        bag.close()
        return msg_dict

    def get_unit(self, record_idx, unit_idx):
        """A unit is defined as a set containing one of each topic msgs.
        These messages have similar timestamps that could be treated as captured concurrently.

        :param record_idx: int The index of the bag file.
        :param unit_idx: int The index of the unit to get, should be within the range [0, n_unit)
        :return: list A list containing all topics' messages in a unit
        """
        if record_idx < 0 or record_idx >= len(self.msg_dict_lists):
            raise ValueError('record idx {} is not legal'.format(record_idx))

        msg_dict = self.msg_dict_lists[record_idx]
        n_unit = msg_dict[self.unit_key]
        if unit_idx < 0 or unit_idx >= n_unit:
            raise ValueError('unit idx is not legal')

        output_list = []
        for i in range(self.n_topics):
            msg_i = msg_dict[self.topics[i]][unit_idx]
            pose_msg = PoseStamped()
            pose_msg.pose = msg_i.pose
            output_list.append(pose_msg)
        return output_list

    def get_xyz(self, record_idx, topic_idx, start=0, end=None):
        msg_dict = self.msg_dict_lists[record_idx]
        msg_list = msg_dict[self.topics[topic_idx]]
        x = []
        y = []
        z = []
        for i in range(len(msg_list)):
            if i < start:
                continue
            if end is not None and i >= end:
                break
            msg_i = msg_list[i]
            x.append(msg_i.pose.position.x)
            y.append(msg_i.pose.position.y)
            z.append(msg_i.pose.position.z)
        return x, y, z
