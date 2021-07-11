from rorlkit.torch.graph_networks import MotionGCN
from rorlkit.torch.gnn.gnn_trainer import GNNTrainer

from rorlkit.core import logger


logger.get_snapshot_dir()
train_dataset_info = {
    'bag_file_list': ['../dataset/bagfile_stirfry/stir_fry_slow.bag',
                      '../dataset/bagfile_stirfry/stir_fry_maxspeed.bag'],
    'topic_list': ['/cartesian/left_hand/reference',
                   '/cartesian/right_hand/reference']
}
test_dataset_info = {
    'bag_file_list': ['../dataset/bagfile_stirfry/stir_fry_fast.bag'],
    'topic_list': ['/cartesian/left_hand/reference',
                   '/cartesian/right_hand/reference']
}
model = MotionGCN(batch_size=128, is_training=True, n_vertices=3, n_frames=1, n_node_features=9)
trainer = GNNTrainer(model, train_dataset_info, test_dataset_info)

for epoch in range(1000):
    trainer.train_epoch(epoch)
    trainer.test_epoch(
        epoch,
        save_model=False,
    )

# logger.save_extra_data(model, 'motion_gcn.pkl', mode='pickle')
