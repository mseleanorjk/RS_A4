from collections import defaultdict
import pickle
import os
from torch_geometric.loader import NeighborLoader
import torch
import random
import numpy as np
from config import *
from sklearn.feature_extraction.text import TfidfVectorizer

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def reconstruction_loss(x_pred, x_true):
    return ((x_pred - x_true)**2).mean(axis=-1)

def get_item_embeddings(metadata=None):
    if os.path.exists(os.path.join("embeddings", "tf_idf_embeddings.npz")):
        print(f'Loading cached embeddings from {os.path.join("embeddings", "tf_idf_embeddings.npz")}')
        cached = np.load(os.path.join("embeddings", "tf_idf_embeddings.npz"), allow_pickle=True)
        item_ids = cached['item_ids']
        embeddings = cached['embeddings'].astype(np.float32)
        print(f'Loaded {embeddings.shape[0]:,} embeddings of dim {embeddings.shape[1]}')
    else:
        item_ids = metadata['item_id'].tolist() #type: ignore
        sequences = metadata['sequence'].tolist() #type: ignore
        tfidf = TfidfVectorizer(
            max_features=4096, # much richer vocabulary
            stop_words='english',
            sublinear_tf=True, # log-scale TF, helps with long descriptions
            min_df=2, # ignore terms appearing in only 1 item (likely noise)
            max_df=0.95, # ignore terms appearing in 95%+ of items (too generic)
        )
        embeddings = tfidf.fit_transform(sequences).toarray() #type:ignore
        embeddings = embeddings.astype(np.float32)
        print("Obtained TF-IDF embeddings with:")
        print(f"Shape: {embeddings.shape}")
        print(f"Sparsity: {(embeddings == 0).mean():.2%}")  
        np.savez_compressed(os.path.join("embeddings", "tf_idf_embeddings.npz"), item_ids=item_ids, embeddings=embeddings)
    return item_ids, embeddings

def build_graph(data, item_ids, k=K, w=W):
    """
    Build graph edges using user sequences. Each node (item) is connected to the 
    top k items that are most bought with it within a window of w items in the user sequences. 
    This captures co-purchase patterns.

    Args:
        data (pd.DataFrame): The user-item interaction data.
        item_ids (list): The list of item IDs.
        k (int, optional): The number of nearest neighbors to consider. Defaults to 10.
        w (int, optional): The window size for co-occurrence. Defaults to 10.
    Returns:
        torch.Tensor: The edge index tensor representing the copurchase graph.
    """
    item_to_idx = {item: idx for idx, item in enumerate(item_ids)}
    
    co_counts = defaultdict(int)
    for _, group in data.groupby('user_id')['item_id']:
        items = group.tolist()
        for i, item_i in enumerate(items):
            for j in range(i+1, min(i+w, len(items))):
                a = item_to_idx.get(item_i)
                b = item_to_idx.get(items[j])
                if a is not None and b is not None:
                    co_counts[(a, b)] += 1
                    co_counts[(b, a)] += 1
    
    neighbours = defaultdict(list)
    for (a, b), count in co_counts.items():
        neighbours[a].append((b, count))
    
    src, dst = [], []
    for item, nbrs in neighbours.items():
        top_k = sorted(nbrs, key=lambda x: -x[1])[:k]
        for nbr, _ in top_k:
            src.append(item)
            dst.append(nbr)
    
    edge_index = torch.tensor([src, dst], dtype=torch.long)
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

def collect_semantic_ids(model, optimizer, data, checkpoint_path="checkpoints/best_rqgat.pt", semid_path = "embeddings/item_semantic_ids.txt"):
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
        with torch.no_grad():
            item_semantic_ids = {}
            loader = NeighborLoader(
                data,
                num_neighbors=[K],
                batch_size=RQGAT_BATCH_SIZE,
                input_nodes=None,
                shuffle=False,
            )
            for batch in loader:
                batch = batch.to(device)
                _, _, _, semantic_ids, _ = model(batch.x, batch.edge_index)
                # Only take seed nodes (first batch.batch_size), not sampled neighbours
                seed_ids = batch.n_id[:batch.batch_size]         # global node indices
                seed_codes = semantic_ids[:batch.batch_size]     # corresponding codes
                for global_idx, codes in zip(seed_ids.cpu().tolist(), seed_codes):
                    item_semantic_ids[data.item_ids[global_idx]] = tuple(codes.cpu().numpy())
        with open(semid_path, "wb") as semid:
            pickle.dump(item_semantic_ids, semid)
    return item_semantic_ids