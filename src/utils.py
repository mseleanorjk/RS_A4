from collections import defaultdict
from polars import groups
from sentence_transformers import SentenceTransformer
import pickle
import faiss
import os
import torch
import random
import numpy as np
from config import *
from data_processor import DataProcessor

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def reconstruction_loss(x_pred, x_true):
    return ((x_pred - x_true)**2).mean(axis=-1)

def get_item_embeddings():
    if os.path.exists(os.path.join("embeddings", "miniLM_embeddings.npz")):
        print(f'Loading cached embeddings from {os.path.join("embeddings", "miniLM_embeddings.npz")}')
        cached = np.load(os.path.join("embeddings", "miniLM_embeddings.npz"), allow_pickle=True)
        item_ids = cached['item_ids']
        embeddings = cached['embeddings'].astype(np.float32)
        print(f'Loaded {embeddings.shape[0]:,} embeddings of dim {embeddings.shape[1]}')
    else:
        metadata = DataProcessor("item_meta.csv").add_sequence()
        model = SentenceTransformer('all-MiniLM-L6-v2')
        item_ids = metadata['item_id'].tolist()
        sequences = metadata['sequence'].tolist()
        embeddings = model.encode(sequences, convert_to_numpy=True)
        embeddings = embeddings.astype(np.float32)
        np.savez_compressed(os.path.join("embeddings", "miniLM_embeddings.npz"), item_ids=item_ids, embeddings=embeddings)
    return item_ids, embeddings

def knn_graph(x, k=10, cosine=True):
    """Build a KNN graph to add graph edge indices to the data in preparation for the GAT"""
    if cosine:
        x_norm = torch.nn.functional.normalize(x, dim=-1)
        sim = x_norm @ x_norm.T
        # negative because the max the similarity the smaller the distance between the nodes
        dists = -sim
    else:
        dists = torch.cdist(x, x)
    # Exclude self-connections between nodes by setting diagonal to infinity
    dists.fill_diagonal_(float('inf'))
    # Get k nearest neighbours for each node
    _, nn_idx = dists.topk(k, dim=1, largest=False)
    # Build edge_index
    B = x.size(0)
    source = torch.arange(B, device=x.device).unsqueeze(1).expand(-1, k).reshape(-1)
    destination = nn_idx.reshape(-1)
    edge_index = torch.stack([source, destination], dim=0)
    return edge_index

def faiss_graph(embeddings, k=10):
    """Approximate KNN for saving memory. Does not materialise the full matrix

    Args:
        embeddings (list): The item embeddings from the pre-trained transformer
        k (int, optional): The number of nearest neighbors to consider. Defaults to 10.

    Returns:
        torch.Tensor: The edge index tensor representing the KNN graph.
    """
    embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)
    _, nn_idx = index.search(embeddings, k + 1)  # +1 because self is included
    nn_idx = nn_idx[:, 1:]  # remove self
    
    n = embeddings.shape[0]
    source = np.repeat(np.arange(n), k)
    destination = nn_idx.reshape(-1)
    edge_index = torch.tensor(np.stack([source, destination]), dtype=torch.long)
    return edge_index

def build_graph(embeddings, k=10, use_faiss=False, cosine=True):
    """Build a KNN graph to add graph edge indices to the data in preparation for the GAT"""
    x = torch.from_numpy(embeddings).float()
    if use_faiss:
        edge_index = faiss_graph(embeddings, k=k)
    else:
        edge_index = knn_graph(x, k=k, cosine=cosine)
    return edge_index

def build_disambiguation(item_semantic_ids):
    """
    Function to compute suffixes for different items with the same semantic ID.
    This suffix will be added at the end of the ID to differentiate them if there are collisions
    """
    # hash the items in ID buckets
    groups = defaultdict(list)
    for asin, codes in item_semantic_ids.items():
        # for each code put in the bucket the item ids that have it
        groups[tuple(codes)].append(asin)

        # Assign suffix per item
    item_suffix = {}
    for codes, asins in groups.items():
        for suffix, asin in enumerate(asins):
            # for each item in a bucket, save a unique suffix
            item_suffix[asin] = suffix  # append the ID within the bucket
    return item_suffix

def tokenise(code, suffix):
    """
    Function to add the disambiguation suffices to a semantic ID and shift
    adjacent codebook codes to match their codebook number. Takes one item
    and applies the respective suffix from the lookup table created earlier.
    """
    tokens = []
    offset = 0
    for i, codebook_idx in enumerate(code):
        tokens.append(int(codebook_idx) + offset) # append the shifted codebook index in the code
        # new offsetting process for dynamic codebook sizing
        offset += CENTROIDS * 2**i
    tokens.append(DIS_TOKEN + suffix)  # append disambiguation token
    return tokens

def save_checkpoint(model, optimizer, epoch, val_loss, path):
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'val_loss': val_loss,
    }, path)

def load_checkpoint(model, optimizer, path):
    checkpoint = torch.load(path, map_location=torch.device(device), weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    return checkpoint['epoch'], checkpoint['val_loss']

def collect_suffixes(item_semantic_ids, verbose=False):
    groups = defaultdict(list)
    for item_id, codes in item_semantic_ids.items():
        # for each code put in the bucket the item ids that have it
        groups[tuple(codes)].append(item_id)
    collisions = 0
    suffixes = []
    for v in groups.values():
        collisions += (len(v)-1)
        suffixes.append(len(v)-1)
    if verbose:
        print(f"Total collisions: {collisions}")
    return suffixes, collisions

def collect_semantic_ids(model, optimizer, dataloader, checkpoint_path="checkpoints/best_rqgat.pt", semid_path = "embeddings/item_semantic_ids.txt"):
    # if already calculated, load the semantic ids, otherwise collect them using the rqvae
    if os.path.exists(semid_path):
        print("Found cached semantic IDs. Loading them...")
        with open(semid_path, "rb") as semid:
            item_semantic_ids = pickle.load(semid)
    else:
        print("Did not find cached semantic IDs. Collecting them...")
        if os.path.exists(checkpoint_path):
            load_checkpoint(model, optimizer, checkpoint_path)
        else:
            raise FileNotFoundError("No checkpoint for RQ-GAT model. Please train the model first.")
        model.eval()
        item_semantic_ids = {}
        # collect semantic ids
        with torch.no_grad():
            for item_ids, x, edge_index in dataloader:
                x = torch.nn.functional.normalize(x.to(device), dim=-1)
                _, _, _, semantic_ids, _ = model(x, edge_index)  # (B, num_codebooks)
                for item_id, codes in zip(item_ids, semantic_ids):
                    item_semantic_ids[item_id] = tuple(codes.cpu().numpy())
        with open(semid_path, "wb") as semid:
            pickle.dump(item_semantic_ids, semid)
    return item_semantic_ids