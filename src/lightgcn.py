import torch
import torch.nn as nn
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import degree
from config import *

class LightGCNConv(MessagePassing):
    """Single LightGCN layer — just normalised neighbourhood aggregation, no weights."""
    def __init__(self):
        super().__init__(aggr='add')
    
    def forward(self, x, edge_index):
        # Compute normalisation: 1 / sqrt(deg(i)) * 1 / sqrt(deg(j))
        row, col = edge_index
        deg = degree(col, x.size(0), dtype=x.dtype)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]
        return self.propagate(edge_index, x=x, norm=norm)
    
    def message(self, x_j, norm):
        return norm.unsqueeze(-1) * x_j


class LightGCN(nn.Module):
    def __init__(self, num_users, num_items, dim=DIM, n_layers=LAYERS, gamma=GAMMA, reg_weight=REG_WEIGHT):
        super().__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.n_layers = n_layers
        self.gamma = gamma
        self.dim = dim
        self.reg_weight = reg_weight
        
        # Embedding table for all nodes (users + items)
        self.embedding = nn.Embedding(num_users + num_items, dim)
        
        # Random init for users
        nn.init.xavier_uniform_(self.embedding.weight[:num_users])
        
        self.convs = nn.ModuleList([LightGCNConv() for _ in range(n_layers)])
    
    def init_item_embeddings(self, svd_embeddings, item_to_idx, item_ids):
        """
        item_ids: the item_ids returned by get_item_embeddings (metadata items)
        item_to_idx: 0-based item index mapping from build_graph
        """
        svd_tensor = torch.tensor(svd_embeddings, dtype=torch.float32)
        if svd_tensor.shape[1] != self.dim:
            projection = nn.Linear(svd_tensor.shape[1], self.dim, bias=False)
            nn.init.xavier_uniform_(projection.weight)
            with torch.no_grad():
                svd_tensor = projection(svd_tensor)
        
        # Only set embeddings for items that appear in both metadata and item_to_idx
        for item_id, emb in zip(item_ids, svd_tensor):
            if item_id in item_to_idx:
                idx = item_to_idx[item_id]
                self.embedding.weight.data[self.num_users + idx] = emb.detach()
    
    def forward(self, edge_index):
        x = self.embedding.weight  # [num_users + num_items, dim]
        
        # Collect embeddings at each layer
        layer_embeddings = [x]
        for conv in self.convs:
            x = conv(x, edge_index)
            layer_embeddings.append(x)
        
        # Exponentially decaying weights per layer using gamma
        weights = torch.tensor(
            [self.gamma ** k for k in range(self.n_layers + 1)],
            dtype=torch.float32, device=x.device
        )
        weights = weights / weights.sum()  # normalise to sum to 1
        
        weights = weights.unsqueeze(-1).unsqueeze(-1)  # [K+1, 1, 1] for broadcasting
        stacked = torch.stack(layer_embeddings, dim=0)  # [K+1, N, dim]
        out = (weights * stacked).sum(dim=0)  # [N, dim]
        user_emb = out[:self.num_users]
        item_emb = out[self.num_users:]
        return user_emb, item_emb
    
    def bpr_loss(self, user_emb, item_emb, users, pos_items, neg_items):
        u = user_emb[users]
        pos = item_emb[pos_items]
        neg = item_emb[neg_items]
        
        pos_scores = (u * pos).sum(dim=-1)
        neg_scores = (u * neg).sum(dim=-1)
        
        loss = -torch.log(torch.sigmoid(pos_scores - neg_scores) + 1e-10).mean()
        
        # L2 regularisation on embeddings
        reg_loss = (u.norm(2).pow(2) + pos.norm(2).pow(2) + neg.norm(2).pow(2)) / len(users)
        return loss + self.reg_weight * reg_loss