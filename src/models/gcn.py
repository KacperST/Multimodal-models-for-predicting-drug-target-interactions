import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_add_pool
import torch.nn as nn


class SimpleGCN(torch.nn.Module):
    def __init__(self, hidden_channels):
        super(SimpleGCN, self).__init__()

        self.atom_embedding = nn.Embedding(120, hidden_channels)

        self.bn1 = nn.BatchNorm1d(hidden_channels)
        self.bn2 = nn.BatchNorm1d(hidden_channels)
        self.bn3 = nn.BatchNorm1d(hidden_channels)

        self.conv1 = GCNConv(hidden_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.conv3 = GCNConv(hidden_channels, hidden_channels)

        self.lin = nn.Linear(hidden_channels, 1)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        x = x[:, 0].long()
        x = self.atom_embedding(x)

        x = self.conv1(x, edge_index)
        x = self.bn1(x)
        x = F.leaky_relu(x, negative_slope=0.01)

        x = self.conv2(x, edge_index)
        x = self.bn2(x)
        x = F.leaky_relu(x, negative_slope=0.01)

        x = self.conv3(x, edge_index)
        x = self.bn3(x)
        x = F.leaky_relu(x, negative_slope=0.01)

        x = global_add_pool(x, batch)

        return self.lin(x)
