import torch
import os
import numpy as np
from torch.utils.data import DataLoader

from config import *
from lightgcn import LightGCN
from callbacks import EarlyStopping
from data_processor import DataProcessor
from datasets import *
from utils import *
from plot_functions import plot_lightgcn_training, plot_metrics

set_seed(42)

def lightgcn_epoch(model, optimizer, train_loader, edge_index, train_df, val_df, user_to_idx, item_to_idx, scheduler=None):
    # Training
    model.train()
    user_emb, item_emb = model(edge_index)

    train_loss = 0
    for user_idx, pos_item_idx, neg_item_idx in train_loader:
        user_idx = user_idx.to(device)
        pos_item_idx = pos_item_idx.to(device)
        neg_item_idx = neg_item_idx.to(device)
        loss = model.bpr_loss(user_emb, item_emb, user_idx, pos_item_idx, neg_item_idx)
        
        optimizer.zero_grad()
        loss.backward(retain_graph=True)  # retain because user_emb/item_emb are reused
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        train_loss += loss.item()

    train_loss /= len(train_loader)

    recall10, ndcg10 = evaluate(model, edge_index, val_df, user_to_idx, item_to_idx, train_df, k=10)
    recall5, ndcg5 = evaluate(model, edge_index, val_df, user_to_idx, item_to_idx, train_df, k=5)

    if scheduler:
        scheduler.step(recall10)

    return train_loss, (recall10, ndcg10, recall5, ndcg5)

def train_lightgcn_model(model, optimizer, train_loader, edge_index, train_df, val_df, user_to_idx, item_to_idx, epochs=EPOCHS, early_stop=None, scheduler=None, verbose=True, save_checkpoints=True):
    final_epoch = epochs
    best_recall = float('-inf')
    train_losses, recall10s, ndcg10s, recall5s, ndcg5s = [], [], [], [], []
    for epoch in range(epochs):
        train_loss, metrics = lightgcn_epoch(model, optimizer, train_loader, edge_index, train_df, val_df, user_to_idx, item_to_idx, scheduler)
        recall10, ndcg10, recall5, ndcg5 = metrics
        
        train_losses.append(train_loss)
        recall10s.append(recall10)
        ndcg10s.append(ndcg10)
        recall5s.append(recall5)
        ndcg5s.append(ndcg5)
        
        if verbose:
            print(f"Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.4f}, Recall@10: {recall10:.4f}")

        if recall10 > best_recall and save_checkpoints:
            best_recall = recall10
            save_checkpoint(model, optimizer, epoch, recall10, os.path.join("checkpoints", "best_rqgat.pt"))
            if verbose:
                print(f"Checkpoint saved (Recall@10: {recall10:.4f})")
        
        if early_stop:
            should_stop = early_stop.step(recall10, epoch)
            if should_stop:
                if verbose:
                    print(f"Early stopping at epoch {epoch+1}")
                final_epoch = epoch+1
                break
    return train_losses, final_epoch, (recall10s, ndcg10s, recall5s, ndcg5s)


def main():
    # make sure the folders for checkpoints, embeddings and images exist
    os.makedirs("checkpoints", exist_ok=True)
    os.makedirs("embeddings", exist_ok=True)
    os.makedirs("images", exist_ok=True)

    # use new dataset structure to create all nodes t the same time instead of batched
    data_processor = DataProcessor("train.csv")
    train, val = data_processor.split_data()
    df = data_processor.df
    edge_index, user_to_idx, item_to_idx = build_graph(train)
    metadata = DataProcessor("item_meta.csv").add_sequence()
    item_ids, embeddings = get_item_embeddings(metadata)
    train_loader = DataLoader(
        BPRDataset(train, user_to_idx, item_to_idx),
        batch_size=BATCH_SIZE,
        shuffle=True
    )
    
    lightgcn = LightGCN(num_users=train["user_id"].nunique(), num_items=train["item_id"].nunique()).to(device)
    lightgcn.init_item_embeddings(embeddings, item_to_idx, item_ids)
    #rqgat.rvq.initialize_codebooks(x, edge_index, rqgat.encoder, device) # initialise the codebooks with kmeans before training
    optimizer = torch.optim.AdamW(lightgcn.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    early_stopping = EarlyStopping(patience=5, delta=0.0001, warmup_epochs=10)

    train_losses, final_epoch, metrics = train_lightgcn_model(
        lightgcn,
        optimizer,
        train_loader,
        edge_index,
        train,
        val,
        user_to_idx,
        item_to_idx,
        early_stop=early_stopping,
    )

    plot_lightgcn_training(
        train_losses,
        final_epoch=final_epoch,
    )
    
    plot_metrics(metrics)

if __name__ == "__main__":
    main()
