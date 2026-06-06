import torch
from torch.utils.data import DataLoader
from utils import *
from config import *
from plot_functions import plot_collisions
from data_processor import DataProcessor
from rqgat import RQGAT
from datasets import *

item_ids, embeddings = get_item_embeddings()
train = DataProcessor("train.csv").df
edge_index = build_graph(train, item_ids)
data = get_data(embeddings, edge_index, item_ids)
rqgat = RQGAT(dim_in=embeddings.shape[1], dim_latent=32)
optimizer = torch.optim.AdamW(rqgat.parameters(), lr=RQGAT_LR)
item_semantic_ids = collect_semantic_ids(rqgat, optimizer, data)
suffixes, collisions = collect_suffixes(item_semantic_ids, verbose=True)
plot_collisions(suffixes, collisions, save=True)