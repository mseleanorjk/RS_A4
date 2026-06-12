from torch.utils.data import Dataset
import torch
import numpy as np
from config import *

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