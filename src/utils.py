from collections import defaultdict
import pickle
import os
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import normalize
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
    if os.path.exists(os.path.join("embeddings", "tf_idf_svd_embeddings.npz")):
        print(f'Loading cached embeddings from {os.path.join("embeddings", "tf_idf_svd_embeddings.npz")}')
        cached = np.load(os.path.join("embeddings", "tf_idf_svd_embeddings.npz"), allow_pickle=True)
        item_ids = cached['item_ids']
        embeddings = cached['embeddings'].astype(np.float32)
        print(f'Loaded {embeddings.shape[0]:,} embeddings of dim {embeddings.shape[1]}')
    else:
        print("No cached embeddings found. Computing TF-IDF SVD embeddings...")
        item_ids = metadata['item_id'].tolist() #type: ignore
        sequences = metadata['sequence'].tolist() #type: ignore
        tfidf = TfidfVectorizer(
            max_features=4096, # much richer vocabulary
            stop_words='english',
            sublinear_tf=True,
            min_df=5,             # ← word must appear in at least 5 items (was 2)
            max_df=0.85,          # ← ignore very common words
            ngram_range=(1, 2),   # ← include bigrams like "stainless steel"
        )
        tfidf_matrix = tfidf.fit_transform(sequences)
        svd = TruncatedSVD(n_components=256, random_state=42)
        embeddings = svd.fit_transform(tfidf_matrix)   # dense [N, 256]
        print(f"Explained variance ratio: {svd.explained_variance_ratio_.sum():.2%}")
        embeddings = normalize(embeddings).astype(np.float32)
        print("Obtained SVD'd TF-IDF embeddings with:")
        print(f"Shape: {embeddings.shape}")
        print(f"Sparsity: {(embeddings == 0).mean():.2%}")  
        np.savez_compressed(os.path.join("embeddings", "tf_idf_svd_embeddings.npz"), item_ids=item_ids, embeddings=embeddings)
    return item_ids, embeddings

def build_graph(data):
    """
    Build graph edges using user sequences. Each user is connected with the 
    items they have interacted with in the training data. This creates a bipartite graph between users and items.

    Args:
        data (pd.DataFrame): The user-item interaction data.
    Returns:
        torch.Tensor: The edge index tensor representing the copurchase graph.
    """
    # Remap IDs to contiguous indices
    unique_users = data['user_id'].unique()
    unique_items = data['item_id'].unique()
    
    user_to_idx = {uid: idx for idx, uid in enumerate(unique_users)}
    item_to_idx = {iid: idx + len(unique_users) for idx, iid in enumerate(unique_items)}
    
    user_idx = data['user_id'].map(user_to_idx).values
    item_idx = data['item_id'].map(item_to_idx).values
    
    # Bipartite edges in both directions (user→item and item→user)
    src = np.concatenate([user_idx, item_idx])
    dst = np.concatenate([item_idx, user_idx])
    
    edge_index = torch.tensor(np.stack([src, dst]), dtype=torch.long)
    
    return edge_index, user_to_idx, item_to_idx

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

def collect_semantic_ids(model, optimizer, loader, checkpoint_path="checkpoints/best_rqgat.pt", semid_path = "embeddings/item_semantic_ids.txt"):
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
            for item_ids_sub, x_sub, sub_edge_index, mapping, _ in loader:
                x_sub = x_sub.to(device)
                sub_edge_index = sub_edge_index.to(device)
                _, _, _, semantic_ids, _ = model(x_sub, sub_edge_index)
                seed_codes = semantic_ids[mapping]
                for item_id, codes in zip(item_ids_sub[mapping.numpy()], seed_codes):
                    item_semantic_ids[item_id] = tuple(codes.cpu().numpy())
        with open(semid_path, "wb") as semid:
            pickle.dump(item_semantic_ids, semid)
    return item_semantic_ids