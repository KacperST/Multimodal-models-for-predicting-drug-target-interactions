import torch
import torch.nn as nn
import torch.nn.functional as F

class DenseGCNConv(nn.Module):
    """Dense Spatial Graph Convolution (similar to Kipf & Welling GCN but for dense matrices)."""
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.linear = nn.Linear(in_channels, out_channels, bias=False)
        self.bias = nn.Parameter(torch.zeros(out_channels))
        
    def forward(self, x: torch.Tensor, adj_norm: torch.Tensor) -> torch.Tensor:
        # x: (B, N, in_channels)
        # adj_norm: (B, N, N)
        # A * X
        support = torch.bmm(adj_norm, x) # (B, N, in_channels)
        # (A * X) * W
        out = self.linear(support) + self.bias
        return out

class LincsGraphEncoder(nn.Module):
    """
    Treats the 978-dim LINCS L1000 profile as a graph where each gene is a node.
    Edges are dynamically computed per sample using a Gaussian kernel on expression differences.
    """
    def __init__(
        self,
        in_dim: int = 978,
        hidden_dim: int = 128,
        out_dim: int = 128,
        num_layers: int = 2,
        theta: float = 1.0,
        dropout: float = 0.5,
    ):
        super().__init__()
        self.in_dim = in_dim
        self.output_dim = out_dim
        self.theta = theta
        self.dropout = dropout
        
        # Initial projection from 1 scalar feature (expression value) to hidden_dim
        self.proj = nn.Linear(1, hidden_dim)
        
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        
        for _ in range(num_layers):
            self.convs.append(DenseGCNConv(hidden_dim, hidden_dim))
            self.bns.append(nn.BatchNorm1d(hidden_dim))
            
        self.pool_proj = nn.Linear(hidden_dim, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x is (B, 978)
        B, N = x.size()
        
        # 1. Compute dynamic adjacency matrix A = exp(- (x_i - x_j)^2 / (2 * theta^2))
        dist_sq = (x.unsqueeze(2) - x.unsqueeze(1)) ** 2  # (B, N, N)
        adj = torch.exp(-dist_sq / (2 * self.theta ** 2)) # (B, N, N)
        
        # 2. Normalize A: D^{-1/2} A D^{-1/2}
        deg = adj.sum(dim=-1).clamp(min=1e-5) # (B, N)
        deg_inv_sqrt = deg.pow(-0.5) # (B, N)
        # Efficient symmetric normalization without diag_embed
        adj_norm = deg_inv_sqrt.unsqueeze(2) * adj * deg_inv_sqrt.unsqueeze(1)
        
        # 3. Node features initialization
        h = x.unsqueeze(-1) # (B, N, 1)
        h = self.proj(h)    # (B, N, hidden_dim)
        
        # 4. Dense GCN Layers
        for conv, bn in zip(self.convs, self.bns):
            h = conv(h, adj_norm) # (B, N, hidden_dim)
            # BN expects (B, C, N) so we transpose
            h = h.transpose(1, 2)
            h = bn(h)
            h = h.transpose(1, 2)
            h = F.relu(h)
            h = F.dropout(h, p=self.dropout, training=self.training)
            
        # 5. Global Mean Pooling over the 978 nodes
        # h: (B, N, hidden_dim) -> (B, hidden_dim)
        h_pooled = h.mean(dim=1)
        
        # 6. Final projection
        out = self.pool_proj(h_pooled) # (B, out_dim)
        
        return out
