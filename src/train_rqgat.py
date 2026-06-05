import torch
import os
import numpy as np

from config import *
from rqgat import RQGAT
from snail import SNAIL
from callbacks import EarlyStopping_RQGAT
from data_processor import DataProcessor
from datasets import *
from utils import *
from plot_functions import plot_rqgat_training, plot_kl_divergence

set_seed(42)

def rqgat_epoch(model, optimizer, x, edge_index, train_mask, val_mask, scheduler=None, plot=False, entropy_weight=ENTROPY_WEIGHT):
    # Training
    x = x.to(device)
    edge_index = edge_index.to(device)
    train_mask = train_mask.to(device)
    val_mask = val_mask.to(device)
    
    model.train()
    model.reset_codebook_util()

    x_hat, _, _, _, rvq_loss = model(x, edge_index, plot=plot)
    # use train and val masks to compute the losses only on the respective splits
    recon_l = reconstruction_loss(x_hat[train_mask], x[train_mask]).mean()
    rqvae_l = rvq_loss
    entropy_l = model.codebook_entropy_loss()
    train_loss = recon_l + rqvae_l + entropy_weight * entropy_l

    optimizer.zero_grad()
    train_loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()

    # Validation
    model.eval()

    with torch.no_grad():
        x_hat_val, _, _, _, rvq_loss_val = model(x, edge_index)
        
        recon_l_val = reconstruction_loss(x_hat_val[val_mask], x[val_mask]).mean()
        rqvae_l_val = rvq_loss_val
        entropy_l_val = model.codebook_entropy_loss()
        val_loss = recon_l_val + rqvae_l_val + entropy_weight * entropy_l_val

    if scheduler:
        scheduler.step(val_loss)

    # Print the utilisation stats
    utilisation = model.compute_codebook_util()
    return (
        train_loss.item(), rqvae_l.item(), recon_l.item(), entropy_l.item(),
        val_loss.item(), rqvae_l_val.item(), recon_l_val.item(), entropy_l_val.item(),
        utilisation
    )

EPOCHS=50

def train_rqgat_model(model, optimizer, x, edge_index, train_mask, val_mask, epochs=EPOCHS, entropy_weight = ENTROPY_WEIGHT, early_stop=None, scheduler=None, verbose=True, save_checkpoints=True):
    final_epoch = epochs
    best_val_loss = float('inf')
    kl = defaultdict(list)
    train_losses, val_losses, train_recon_losses, val_recon_losses, train_rqvae_losses, val_rqvae_losses, train_entropy_losses, val_entropy_losses = [], [], [], [], [], [], [], []
    for epoch in range(epochs):
        train_loss, train_rqvae_l, train_recon_l, train_entropy_l, val_loss, val_rqvae_l, val_recon_l, val_entropy_l, utilisation = rqgat_epoch(model, optimizer, x, edge_index, train_mask, val_mask, entropy_weight=entropy_weight, scheduler=scheduler)
        
        train_recon_losses.append(train_recon_l)
        train_rqvae_losses.append(train_rqvae_l)
        train_entropy_losses.append(train_entropy_l)
        train_losses.append(train_loss)

        val_recon_losses.append(val_recon_l)
        val_rqvae_losses.append(val_rqvae_l)
        val_entropy_losses.append(val_entropy_l)
        val_losses.append(val_loss)
        
        for i in range(model.num_codebooks):
            klx, _ = utilisation[i]
            kl[i].append(klx)
        
        if verbose:
            print(f"Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
            print(*(f"Codebook {i}: KL divergence = {kl}, Used = {used}" for i, (kl, used) in enumerate(utilisation, start=1)), sep=", ")
        if val_loss < best_val_loss and save_checkpoints:
            best_val_loss = val_loss
            save_checkpoint(model, optimizer, epoch, val_loss, os.path.join("checkpoints", "best_rqgat.pt"))
            if verbose:
                print(f"Checkpoint saved (val loss: {val_loss:.4f})")
        
        if early_stop:
            should_stop = early_stop.step(val_loss, epoch)
            if should_stop:
                if verbose:
                    print(f"Early stopping at epoch {epoch+1}")
                final_epoch = epoch+1
                break
    return train_losses, val_losses, train_recon_losses, val_recon_losses, train_rqvae_losses, val_rqvae_losses, train_entropy_losses, val_entropy_losses, final_epoch, kl


def main():
    # make sure the folders for checkpoints, embeddings and images exist
    os.makedirs("checkpoints", exist_ok=True)
    os.makedirs("embeddings", exist_ok=True)
    os.makedirs("images", exist_ok=True)

    # use new dataset structure to create all nodes t the same time instead of batched
    metadata = DataProcessor("item_meta.csv").add_sequence()
    item_ids, embeddings = get_item_embeddings(metadata)
    train = DataProcessor("train.csv").df
    edge_index = build_graph(train, item_ids)
    rqgat_dataset = RQGATDataset(edge_index, item_ids, embeddings, split=SPLIT_PERC, k=K)
    x, edge_index, train_mask, val_mask = rqgat_dataset.get_full_data()
    x=x.to(device)
    edge_index = edge_index.to(device)
    train_mask = train_mask.to(device)
    val_mask = val_mask.to(device)
    
    rqgat = RQGAT(dim_in=embeddings.shape[1], dim_latent=32).to(device)
    #rqgat.rvq.initialize_codebooks(x, edge_index, rqgat.encoder, device) # initialise the codebooks with kmeans before training
    optimizer = torch.optim.AdamW(rqgat.parameters(), lr=RQGAT_LR, weight_decay=WEIGHT_DECAY_RQGAT)
    early_stopping = EarlyStopping_RQGAT(patience=5, delta=0.0001, warmup_epochs=10)

    train_losses, val_losses, train_recon_losses, val_recon_losses, train_rqvae_losses, val_rqvae_losses, train_entropy_losses, val_entropy_losses, final_epoch, kl = train_rqgat_model(
        rqgat,
        optimizer,
        x,
        edge_index,
        train_mask,
        val_mask,
        early_stop=early_stopping,
    )

    plot_rqgat_training(
        train_losses,
        val_losses,
        train_recon_losses,
        val_recon_losses,
        train_rqvae_losses,
        val_rqvae_losses,
        train_entropy_losses,
        val_entropy_losses,
        final_epoch=final_epoch,
    )
    plot_kl_divergence(kl, final_epoch=final_epoch)

    # Plot the centroids for the last epoch to visualise how the codebooks are distributed in the latent space
    _, _ = load_checkpoint(rqgat, optimizer=optimizer, path="checkpoints/best_rqgat.pt")
    with torch.no_grad():
        _, _, _, _, _ = rqgat(x, edge_index, plot=True)


if __name__ == "__main__":
    main()
