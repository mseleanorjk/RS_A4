from torch_geometric.data import Data
from torch.utils.data import Dataset
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

class BPRDataset(Dataset):
    def __init__(self, data, user_to_idx, item_to_idx):
        self.user_to_idx = user_to_idx
        self.item_to_idx = item_to_idx
        self.num_items = len(item_to_idx)
        
        # Build user positive items lookup for negative sampling
        self.user_pos_items = {}
        for user_id, group in data.groupby('user_id')['item_id']:
            user_idx = user_to_idx[user_id]
            self.user_pos_items[user_idx] = set(
                item_to_idx[iid] for iid in group.tolist()
            )
        
        # Build list of (user_idx, pos_item_idx) pairs
        self.interactions = []
        for user_idx, pos_items in self.user_pos_items.items():
            for pos_item in pos_items:
                self.interactions.append((user_idx, pos_item))
    
    def __len__(self):
        return len(self.interactions)
    
    def __getitem__(self, idx):
        user_idx, pos_item_idx = self.interactions[idx]
        
        # Sample a negative item not in user's history
        while True:
            neg_item_idx = np.random.randint(0, self.num_items)
            if neg_item_idx not in self.user_pos_items[user_idx]:
                break
        
        return (
            torch.tensor(user_idx, dtype=torch.long),
            torch.tensor(pos_item_idx, dtype=torch.long),
            torch.tensor(neg_item_idx, dtype=torch.long),
        )