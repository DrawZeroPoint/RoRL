import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn import GCNConv


class MotionGCN(nn.Module):
    """This graph network consider a motion graph composed of n_frames frames,
    in each frame, the motion is described with n_vertices vertices representing
    the distal frames of a human. By default, n_vertices=6, where the base frame,
    the both hand frames, the pelvis frame, and both foot frames are included.
    The whole graph is constructed by connect each frame in the temporal domain.
    By default, we compose 5 frames, making the n_nodes in the graph equals to 5*6=30
    """
    def __init__(
            self,
            batch_size,
            is_training,
            n_vertices,
            n_frames,
            n_node_features,
            n_hidden_layers=1,
            init_w=1e-4,
    ):
        super(MotionGCN, self).__init__()
        self.batch_size = batch_size
        self.training = is_training
        self.n_vertices = n_vertices
        self.n_node_features = n_node_features
        self.n_nodes = n_frames * n_vertices
        print(n_vertices, n_frames, n_node_features)
        # Create the input graph convolution layer
        self.input_layer = GCNConv(n_node_features, 16)
        # Create the hidden graph convolution layer
        self.hidden_layers = nn.ModuleList()
        for i in range(n_hidden_layers):
            conv = GCNConv(16 * (i+1), 16 * (i+2))
            self.hidden_layers.append(conv)
        # Create output graph convolution layer
        self.output_layer = nn.Linear(16 * (n_hidden_layers+1), n_node_features)
        if self.training:
            self.output_layer.weight.data.uniform_(-init_w, init_w)
            self.output_layer.bias.data.uniform_(-init_w, init_w)

    def forward(self, data):
        """
        data.x: Node feature matrix with shape [num_nodes, num_node_features]
        """
        x, edge_index = data.x, data.edge_index
        # print('0:', x.size())
        # print('1:', edge_index.size())
        x = self.input_layer(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        for h_layer in self.hidden_layers:
            x = h_layer(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, training=self.training)
        x = x.view(self.batch_size, self.n_vertices, -1)
        # print('2:', x.size())
        x = self.output_layer(x)
        # print('3:', x.size())
        return x
