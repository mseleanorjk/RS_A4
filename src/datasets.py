from torch.utils.data import Dataset

class RQGATDataset(Dataset):
    def __init__(self, emb_dict):
        super(RQGATDataset, self).__init__()
        self.item_ids = emb_dict.keys().tolist()
        self.embeddings = emb_dict.values().tolist()
    
    def __len__(self):
        return len(self.item_ids)
    
    def __getitem__(self, index):
        return self.item_ids[index], self.embeddings[index]