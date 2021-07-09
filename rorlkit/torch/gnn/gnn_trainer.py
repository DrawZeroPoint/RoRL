from collections import OrderedDict
import numpy as np
import torch.nn.functional as F

from torch import optim
from torch_geometric.data import DataLoader
from rorlkit.core import logger
from rlkit.core.eval_util import create_stats_ordered_dict
from rlkit.torch import pytorch_util as ptu

from rorlkit.torch.data import (
    MotionGraphDataset,
    InfiniteRandomSampler,
)


class GNNTrainer(object):
    def __init__(
            self,
            model,
            train_dataset_info,
            test_dataset_info,
            batch_size=128,
            log_interval=10,
            lr=1e-3,
            n_workers=2,
            weight_decay=0,
    ):
        model.to(ptu.device)
        self.model = model

        self.batch_size = batch_size
        self.log_interval = log_interval

        self.lr = lr
        params = list(self.model.parameters())
        self.optimizer = optim.Adam(params, lr=self.lr, weight_decay=weight_decay,)

        self.train_dataset = MotionGraphDataset(20000, **train_dataset_info)
        self.train_dataloader = DataLoader(
            self.train_dataset,
            batch_size=batch_size,
            sampler=InfiniteRandomSampler(self.train_dataset),
            drop_last=False,
            num_workers=n_workers,
            pin_memory=True,
        )
        self.train_dataloader = iter(self.train_dataloader)

        self.test_dataset = MotionGraphDataset(10000, **test_dataset_info)
        self.test_dataloader = DataLoader(
            self.test_dataset,
            batch_size=batch_size,
            sampler=InfiniteRandomSampler(self.test_dataset),
            drop_last=False,
            num_workers=n_workers,
            pin_memory=True,
        )
        self.test_dataloader = iter(self.test_dataloader)

        self.eval_statistics = OrderedDict()
        self._extra_stats_to_log = None

    def get_graph_batch(self, is_training):
        if is_training:
            graphs_curr, graphs_next = next(self.train_dataloader)
        else:
            graphs_curr, graphs_next = next(self.test_dataloader)
        return graphs_curr, graphs_next

    def train_epoch(self, epoch):
        self.model.train()
        losses = []
        graphs_curr, graphs_next = self.get_graph_batch(is_training=True)
        graphs_next_pred = self.model(graphs_curr)
        graphs_next_gt = graphs_next.x.reshape(graphs_next_pred.size())

        loss = F.mse_loss(graphs_next_pred, graphs_next_gt)
        losses.append(loss.item())

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        if self.log_interval and epoch % self.log_interval == 0:
            print('Train Epoch: {} \tLoss: {:.6f}'.format(epoch, loss.item()))

        self.eval_statistics['train/loss'] = np.mean(losses)

    def get_diagnostics(self):
        return self.eval_statistics

    def test_epoch(
            self,
            epoch,
            save_model=True,
    ):
        self.model.eval()
        losses = []

        graphs_curr, graphs_next = self.get_graph_batch(is_training=False)
        graphs_next_pred = self.model(graphs_curr)
        graphs_next_gt = graphs_next.x.reshape(graphs_next_pred.size())

        loss = F.mse_loss(graphs_next_pred, graphs_next_gt)
        losses.append(loss.item())

        if self.log_interval and epoch % self.log_interval == 0:
            print('Test Epoch: {} \tLoss: {:.6f}'.format(epoch, loss.item()))

        self.eval_statistics['epoch'] = epoch
        self.eval_statistics['test/loss'] = np.mean(losses)
        if save_model:
            pass
            # logger.save_itr_params(epoch, self.model)
