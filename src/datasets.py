from torch_geometric.data import Data
from torch_geometric.utils import k_hop_subgraph
import torch
import numpy as np
from config import *

def get_data(embeddings, edge_index, item_ids, split=SPLIT_PERC):
    x = torch.from_numpy(embeddings).float()
    n = len(embeddings)
    split_n = int(split * n)
    
    train_mask = torch.zeros(n, dtype=torch.bool)
    train_mask[:split_n] = True
    val_mask = ~train_mask
    item_ids=np.array(item_ids)

    data = Data(x=x, edge_index=edge_index, train_mask=train_mask, val_mask=val_mask, item_ids=item_ids)
    return data

# def get_loaders(data, batch_size=RQGAT_BATCH_SIZE, num_neighbours=K):
#     train_loader = NeighborLoader(
#         data,
#         num_neighbors=[num_neighbours],     # how many neighbours to sample per node
#         batch_size=batch_size,
#         input_nodes=data.train_mask,        # only sample seed nodes from train
#     )
#     val_loader = NeighborLoader(
#         data,
#         num_neighbors=[num_neighbours],
#         batch_size=batch_size,
#         input_nodes=data.val_mask,
#         shuffle=False,
#     )
#     return train_loader, val_loader

class SubgraphLoader:
    def __init__(self, data, train=True, batch_size=RQGAT_BATCH_SIZE, shuffle=True):
        self.data = data
        mask = data.train_mask if train else data.val_mask
        idx = mask.nonzero(as_tuple=True)[0]
        self.loader = torch.utils.data.DataLoader(idx, batch_size=batch_size, shuffle=shuffle)
    
    def __iter__(self):
        for seed_nodes in self.loader:
            node_idx, sub_edge_index, mapping, _ = k_hop_subgraph(
                seed_nodes, num_hops=1, edge_index=self.data.edge_index, relabel_nodes=True
            )
            yield self.data.item_ids[node_idx.numpy()], self.data.x[node_idx], sub_edge_index, mapping, seed_nodes
    
    def __len__(self):
        return len(self.loader)

def get_loaders(data, batch_size=RQGAT_BATCH_SIZE):
    train_loader = SubgraphLoader(data, train=True, batch_size=batch_size)
    val_loader = SubgraphLoader(data, train=False, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader