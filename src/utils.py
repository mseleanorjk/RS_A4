import os
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import normalize
import torch
import random
import pandas as pd
import numpy as np
from config import *
from sklearn.feature_extraction.text import TfidfVectorizer

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

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
            min_df=5, #word must appear in at least 5 items
            max_df=0.85, # ignore very common words
            ngram_range=(1, 2), # include bigrams like "stainless steel"
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

def generate_submission(model, edge_index, train_df, user_to_idx, item_to_idx, 
                        sample_submission_path="data/sample_submission.csv",
                        output_path="data/submission.csv"):
    model.eval()
    with torch.no_grad():
        user_emb, item_emb = model(edge_index.to(device))
    
    idx_to_item = {idx: item_id for item_id, idx in item_to_idx.items()}
    user_train_items = train_df.groupby('user_id')['item_id'].apply(set).to_dict()
    sample_sub = pd.read_csv(sample_submission_path)
    
    rows = []
    for user_id in sample_sub['user_id'].unique():
        if user_id not in user_to_idx:
            # Cold start — no interactions in training, predict most popular items
            scores = item_emb.norm(dim=-1)  # fallback
        else:
            user_idx = user_to_idx[user_id]
            u = user_emb[user_idx]
            scores = item_emb @ u
            for train_item in user_train_items.get(user_id, set()):
                if train_item in item_to_idx:
                    scores[item_to_idx[train_item]] = float('-inf')
        
        top10_idx = scores.topk(10).indices.tolist()
        top10_items = [idx_to_item[idx] for idx in top10_idx]
        
        rows.append({
            'ID': user_id,
            'user_id': user_id,
            'item_id': ','.join(map(str, top10_items))
        })
    
    submission = pd.DataFrame(rows)
    submission.to_csv(output_path, index=False)
    print(f"Submission saved to {output_path} with {len(submission)} users")
    return submission