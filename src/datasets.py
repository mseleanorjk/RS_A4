from torch.utils.data import Dataset
import numpy as np
import torch
from rqgat import knn_graph

def collate_fn(batch, k=10):
    item_ids, embeddings = zip(*batch)
    item_ids = torch.stack(item_ids) if torch.is_tensor(item_ids[0]) else list(item_ids)
    x = torch.from_numpy(np.asarray(embeddings, dtype=np.float32))   # [B, D]
    B = x.size(0)
    k = min(k, B - 1)
    edge_index = knn_graph(x, k=k, cosine=True)
    return item_ids, x, edge_index

class RQGATDataset(Dataset):
    def __init__(self, emb_tuple):
        super(RQGATDataset, self).__init__()
        self.item_ids = emb_tuple[0]
        self.embeddings = emb_tuple[1]

    def __len__(self):
        return len(self.item_ids)
    
    def __getitem__(self, index):
        return self.item_ids[index], self.embeddings[index]
