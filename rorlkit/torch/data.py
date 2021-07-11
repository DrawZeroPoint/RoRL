import numpy as np

import torch
from torch.utils.data import Dataset, Sampler
from torch_geometric.data import Data

from rorlkit.torch.data_management import rosbag_data


def normalize_image(image):
    assert image.dtype == np.uint8
    return np.float64(image) / 255.0


class ImageDataset(Dataset):

    def __init__(self, images, should_normalize=True):
        super().__init__()
        self.dataset = images
        self.dataset_len = len(self.dataset)
        assert should_normalize == (images.dtype == np.uint8)
        self.should_normalize = should_normalize

    def __len__(self):
        return self.dataset_len

    def __getitem__(self, idxs):
        samples = self.dataset[idxs, :]
        if self.should_normalize:
            samples = normalize_image(samples)
        return np.float32(samples)


class InfiniteRandomSampler(Sampler):

    def __init__(self, data_source):
        self.data_source = data_source
        self.iter = iter(torch.randperm(len(self.data_source)).tolist())

    def __iter__(self):
        return self

    def __next__(self):
        try:
            idx = next(self.iter)
        except StopIteration:
            self.iter = iter(torch.randperm(len(self.data_source)).tolist())
            idx = next(self.iter)
        return idx

    def __len__(self):
        return 2 ** 62


class InfiniteWeightedRandomSampler(Sampler):

    def __init__(self, data_source, weights):
        assert len(data_source) == len(weights)
        assert len(weights.shape) == 1
        self.data_source = data_source
        # Always use CPU
        self._weights = torch.from_numpy(weights)
        self.iter = self._create_iterator()

    def update_weights(self, weights):
        self._weights = weights
        self.iter = self._create_iterator()

    def _create_iterator(self):
        return iter(
            torch.multinomial(
                self._weights, len(self._weights), replacement=True
            ).tolist()
        )

    def __iter__(self):
        return self

    def __next__(self):
        try:
            idx = next(self.iter)
        except StopIteration:
            self.iter = self._create_iterator()
            idx = next(self.iter)
        return idx

    def __len__(self):
        return 2 ** 62


class MotionGraphDataset(Dataset):
    def __init__(self, db_len, bag_file_list, topic_list):
        super(MotionGraphDataset, self).__init__()
        self._n_bag = len(bag_file_list)
        self._db = []
        self.pg = rosbag_data.PosesGetter(bag_file_list, topic_list)
        self._build_db(db_len)
        self._db_len = db_len

    def __len__(self):
        return len(self._db)

    def __getitem__(self, item):
        graph_curr, graph_next = self._db[item]
        return graph_curr, graph_next

    def _build_db(self, db_len):
        for record_idx in range(self._n_bag):
            msg_dict = self.pg.msg_dict_lists[record_idx]
            n_unit = msg_dict[self.pg.unit_key]
            random_idx_list = np.random.randint(low=0, high=n_unit-1, size=db_len)
            for unit_idx in random_idx_list:
                graphs = self._get_motion_graphs(record_idx, unit_idx)
                self._db.append(graphs)
        print('Created database of length {}'.format(self.__len__()))

    def _get_motion_graphs(self, record_idx, unit_idx):
        """One frame of the graph is composed by: base link pose, left hand pose, right hand pose.
        Get one graph for training and one for evaluation.

        :param record_idx: int The index of the bag file.
        :param unit_idx: int Starting index of the unit in the rollout.
        """
        edge_index = torch.tensor([[0, 0], [1, 2]], dtype=torch.long)
        graphs = []
        # 2 is for getting the current and the next frames of the motion
        for i in range(2):
            _idx = unit_idx + i
            # Get root node features
            # TODO In a upper body fashion, we do not consider the movement of the base
            base_pose_feature = rosbag_data.get_identity_feature()
            # Get children features
            left_hand_pose, right_hand_pose = self.pg.get_unit(record_idx, _idx)
            left_hand_feature = rosbag_data.pose_msg_to_feature(left_hand_pose)
            right_hand_feature = rosbag_data.pose_msg_to_feature(right_hand_pose)
            # Create node features
            x = torch.tensor([base_pose_feature, left_hand_feature, right_hand_feature], dtype=torch.float)
            # Create graph
            data_i = Data(x=x, edge_index=edge_index)
            graphs.append(data_i)
        return graphs
