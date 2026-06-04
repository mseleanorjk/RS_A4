from torch.utils.data import Dataset
import numpy as np
import torch
from utils import build_graph
from config import *

class RQGATDataset(Dataset):
    def __init__(self, item_ids, embeddings, split=SPLIT_PERC, k=10):
        super().__init__()
        self.item_ids = item_ids
        self.x = torch.from_numpy(embeddings).float()
        
        # Build graph over ALL items
        self.edge_index = build_graph(embeddings, k=k)
        
        # Create masks
        n = len(item_ids)
        self.split = split
        self.split_n = int(split * n)
        self.train_mask = torch.zeros(n, dtype=torch.bool)
        self.train_mask[:self.split_n] = True
        self.val_mask = ~self.train_mask

    def __len__(self):
        return len(self.item_ids)
    
    def get_full_data(self):
        return self.x, self.edge_index, self.train_mask, self.val_mask
