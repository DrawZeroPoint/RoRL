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
from rlkit.util.ml_util import ConstantSchedule


class GNNTrainer(object):
    def __init__(
            self,
            model,
            batch_size=128,
            log_interval=10,
            lr=1e-3,
            train_data_workers=2,
            weight_decay=0,
    ):
        model.to(ptu.device)
        self.model = model

        self.batch_size = batch_size
        self.log_interval = log_interval

        self.lr = lr
        params = list(self.model.parameters())
        self.optimizer = optim.Adam(params, lr=self.lr, weight_decay=weight_decay,)

        self.train_data_workers = train_data_workers

        self.dataset = MotionGraphDataset(10000)
        self.dataloader = DataLoader(
            self.dataset,
            batch_size=batch_size,
            sampler=InfiniteRandomSampler(self.dataset),
            drop_last=False,
            num_workers=train_data_workers,
            pin_memory=True,
        )
        self.dataloader = iter(self.dataloader)

        self.eval_statistics = OrderedDict()
        self._extra_stats_to_log = None

    def get_graph_batch(self):
        graphs_curr, graphs_next = next(self.dataloader)
        return graphs_curr, graphs_next

    def train_epoch(self, epoch):
        self.model.train()
        losses = []
        graphs_curr, graphs_next = self.get_graph_batch()
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

        graphs_curr, graphs_next = self.get_graph_batch()
        graphs_next_pred = self.model(graphs_curr)
        graphs_next_gt = graphs_next.x.reshape(graphs_next_pred.size())

        loss = F.mse_loss(graphs_next_pred, graphs_next_gt)
        losses.append(loss.item())

        self.eval_statistics['epoch'] = epoch
        self.eval_statistics['test/loss'] = np.mean(losses)
        if save_model:
            pass
            # logger.save_itr_params(epoch, self.model)
