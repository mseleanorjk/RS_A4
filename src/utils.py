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

def get_item_embeddings(metadata=None, item_to_idx=None):
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
        if item_to_idx is not None:
            item_id_to_emb = dict(zip(item_ids, embeddings))
            embeddings = np.stack([
                item_id_to_emb[item_id]
                for item_id, _ in sorted(item_to_idx.items(), key=lambda x: x[1])
                if item_id in item_id_to_emb
            ])
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
    item_to_idx = {iid: idx for idx, iid in enumerate(unique_items)}
    
    user_idx = data['user_id'].map(user_to_idx).values
    item_idx = data['item_id'].map(lambda x: item_to_idx[x] + len(unique_users)).values
    
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

def evaluate(model, edge_index, val_df, user_to_idx, item_to_idx, train_df, k=10):
    model.eval()
    with torch.no_grad():
        user_emb, item_emb = model(edge_index.to(device))
    
    # Build user training history for filtering
    user_train_items = train_df.groupby('user_id')['item_id'].apply(set).to_dict()
    
    recalls, ndcgs = [], []
    
    for _, row in val_df.iterrows():
        user_id = row['user_id']
        true_item = row['item_id']
        
        # Skip users/items not in training
        if user_id not in user_to_idx or true_item not in item_to_idx:
            continue
        
        user_idx = user_to_idx[user_id]
        true_item_idx = item_to_idx[true_item]
        
        # Score all items
        u = user_emb[user_idx]                    # [dim]
        scores = item_emb @ u                     # [num_items]
        
        # Mask out training items
        train_items = user_train_items.get(user_id, set())
        for train_item in train_items:
            if train_item in item_to_idx:
                scores[item_to_idx[train_item]] = float('-inf')
        
        # Top-k items
        top_k = scores.topk(k).indices.tolist()
        
        # Recall@k — is the true item in top-k?
        hit = int(true_item_idx in top_k)
        recalls.append(hit)
        
        # NDCG@k — where in the top-k is the true item?
        if hit:
            rank = top_k.index(true_item_idx) + 1
            ndcgs.append(1 / np.log2(rank + 1))
        else:
            ndcgs.append(0.0)
    
    return np.mean(recalls), np.mean(ndcgs)