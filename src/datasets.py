from torch.utils.data import Dataset
import torch
from utils import build_graph
from config import *

class RQGATDataset(Dataset):
    def __init__(self, edge_index, item_ids, embeddings, split=SPLIT_PERC, k=K):
        super().__init__()
        self.item_ids = item_ids
        self.x = torch.from_numpy(embeddings).float()
        self.edge_index = edge_index
        self.k = k
        
        # Build graph over ALL items
        #self.edge_index = build_graph(self.metadata, embeddings, k=k)
        
        # Create masks
        n = len(item_ids)
        self.split = split
        self.split_n = int(split * n)
        self.train_mask = torch.zeros(n, dtype=torch.bool)
        self.train_mask[:self.split_n] = True
        self.val_mask = ~self.train_mask

    def __len__(self):
        return len(self.item_ids)
    
    def __getitem__(self, idx):
        return self.item_ids[idx], self.x[idx], self.edge_index
    
    def get_full_data(self):
        return self.x, self.edge_index, self.train_mask, self.val_mask
