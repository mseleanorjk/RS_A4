import torch
from torch.utils.data import DataLoader
from utils import *
from config import *
from plot_functions import plot_collisions
from rqgat import RQGAT
from datasets import *

item_ids, embeddings = get_item_embeddings()
rqgat = RQGAT(dim_in=embeddings.shape[1], dim_latent=32)
rqgat_dataset = RQGATDataset((item_ids, embeddings))
rqgat_loader = DataLoader(rqgat_dataset, collate_fn=collate_fn, batch_size=RQGAT_BATCH_SIZE, shuffle=True)
optimizer = torch.optim.AdamW(rqgat.parameters(), lr=RQGAT_LR)
item_semantic_ids = collect_semantic_ids(rqgat, optimizer, rqgat_loader)
suffixes, collisions = collect_suffixes(item_semantic_ids, verbose=True)
plot_collisions(suffixes, collisions, save=True)