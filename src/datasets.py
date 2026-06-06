from torch_geometric.data import Data
from torch_geometric.loader import NeighborLoader
import torch
from config import *

def get_data(embeddings, edge_index, split=SPLIT_PERC):
    x = torch.from_numpy(embeddings).float()
    n = len(embeddings)
    split_n = int(split * n)
    
    train_mask = torch.zeros(n, dtype=torch.bool)
    train_mask[:split_n] = True
    val_mask = ~train_mask

    data = Data(x=x, edge_index=edge_index, train_mask=train_mask, val_mask=val_mask)
    return data

def get_loaders(data, batch_size=RQGAT_BATCH_SIZE, num_neighbours=K):
    train_loader = NeighborLoader(
        data,
        num_neighbors=[num_neighbours],     # how many neighbours to sample per node
        batch_size=batch_size,
        input_nodes=data.train_mask,        # only sample seed nodes from train
    )
    val_loader = NeighborLoader(
        data,
        num_neighbors=[num_neighbours],
        batch_size=batch_size,
        input_nodes=data.val_mask,
        shuffle=False,
    )
    return train_loader, val_loader