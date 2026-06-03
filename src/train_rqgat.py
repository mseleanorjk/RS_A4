import torch
import os
import numpy as np
from torch.utils.data import DataLoader

from config import *
from data_processor import DataProcessor
from rqgat import RQGAT
from snail import SNAIL
from callbacks import EarlyStopping_RQGAT
from datasets import *
from utils import *
from plot_functions import plot_rqgat_training, plot_kl_divergence

set_seed(42)

train = DataProcessor("train.csv").df
metadata_processor = DataProcessor("item_meta.csv")
test = DataProcessor("test.csv").df
metadata = metadata_processor.add_sequence()

item_ids, embeddings = get_item_embeddings(metadata)

rqgat = RQGAT(dim_in=embeddings.shape[1], dim_latent=32)
split = int(0.8 * len(item_ids))
train_rqgat_dataset = RQGATDataset((item_ids[:split], embeddings[:split]))
val_rqgat_dataset   = RQGATDataset((item_ids[split:], embeddings[split:]))
train_rqgat_loader = DataLoader(train_rqgat_dataset, collate_fn=collate_fn, batch_size=RQGAT_BATCH_SIZE, shuffle=True)
rqgat.rvq.initialize_codebooks(train_rqgat_loader, rqgat.encoder, device) # initialise the codebooks with kmeans before training
val_rqgat_loader = DataLoader(val_rqgat_dataset, collate_fn=collate_fn, batch_size=RQGAT_BATCH_SIZE, shuffle=False)

print(f"Train items: {len(train_rqgat_dataset)}, Val items: {len(val_rqgat_dataset)}")
print(f"Embeddings mean: {embeddings.mean():.4f}, std: {embeddings.std():.4f}")
print(f"Embeddings min: {embeddings.min():.4f}, max: {embeddings.max():.4f}")

def rqgat_epoch(model, optimizer, train_loader, val_loader, scheduler=None, plot=False, entropy_weight=ENTROPY_WEIGHT):
    # Training
    model.train()
    train_loss = 0
    train_recon_l = 0
    train_rqvae_l = 0
    train_entropy_l = 0
    
    val_loss = 0
    val_recon_l = 0
    val_rqvae_l = 0
    val_entropy_l = 0

    for _, x, edge_index in train_loader:
        x = x.to(device)
        edge_index = edge_index.to(device)
        x_hat, _, _, _, rvq_loss = model(x, edge_index, plot=plot)
        recon_l = reconstruction_loss(x_hat, x).mean()
        rqvae_l = rvq_loss
        entropy_l = model.codebook_entropy_loss()
        loss = recon_l + rqvae_l + entropy_weight * entropy_l

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        train_recon_l += recon_l.item()
        train_rqvae_l += rqvae_l.item()
        train_entropy_l += entropy_l.item()
        train_loss += loss.item()

    # Validation
    model.eval()
    model.reset_codebook_util()
    val_loss = 0

    with torch.no_grad():
        for _, x, edge_index in val_loader:
            x = x.to(device)
            x_hat, _, _, _, rqv_loss = model(x, edge_index)
            recon_l = reconstruction_loss(x_hat, x).mean()
            rqvae_l = rqv_loss
            entropy_l = model.codebook_entropy_loss()
            loss = recon_l + rqvae_l + entropy_weight * entropy_l

            val_recon_l += recon_l.item()
            val_rqvae_l += rqvae_l.item()
            val_entropy_l += entropy_l.item()
            val_loss += loss.item()

    avg_val_loss = val_loss / len(val_loader)

    if scheduler:
        scheduler.step(avg_val_loss)

    # Print the utilisation stats
    utilisation = model.compute_codebook_util()
    return train_loss, train_rqvae_l, train_recon_l, train_entropy_l, val_loss, val_rqvae_l, val_recon_l, val_entropy_l, utilisation

EPOCHS=50
optimizer = torch.optim.Adam(rqgat.parameters(), lr=RQGAT_LR)
early_stopping = EarlyStopping_RQGAT(patience=5, delta=0.01, warmup_epochs=10)

def train_rqgat(model, optimizer, train_loader, val_loader, epochs=EPOCHS, entropy_weight = ENTROPY_WEIGHT, early_stop=None, scheduler=None, verbose=True):
    final_epoch = epochs
    best_val_loss = float('inf')
    kl = defaultdict(list)
    train_losses, val_losses, train_recon_losses, val_recon_losses, train_rqvae_losses, val_rqvae_losses, train_entropy_losses, val_entropy_losses = [], [], [], [], [], [], [], []
    for epoch in range(epochs):
        train_loss, train_rqvae_l, train_recon_l, train_entropy_l, val_loss, val_rqvae_l, val_recon_l, val_entropy_l, utilisation = rqgat_epoch(model, optimizer, train_loader, val_loader, entropy_weight=entropy_weight, early_stop=early_stop, scheduler=scheduler)
        
        train_recon_losses.append(train_recon_l/len(train_loader))
        train_rqvae_losses.append(train_rqvae_l/len(train_loader))
        train_entropy_losses.append(train_entropy_l/len(train_loader))
        train_losses.append(train_loss/len(train_loader))

        val_recon_losses.append(val_recon_l/len(val_loader))
        val_rqvae_losses.append(val_rqvae_l/len(val_loader))
        val_entropy_losses.append(val_entropy_l/len(val_loader))
        val_losses.append(val_loss/len(val_loader))
        
        for i in range(NUM_CODEBOOKS):
            klx, _ = utilisation[i]
            kl[i].append(klx)
        
        if verbose:
            print(f"Epoch {epoch+1}/{epochs} - Train Loss: {train_loss/len(train_loader):.4f}, Val Loss: {val_loss/len(val_loader):.4f}")
            print(*(f"Codebook {i}: KL divergence = {kl}, Used = {used}" for i, (kl, used) in enumerate(utilisation, start=1)), sep=", ")
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint(rqgat, optimizer, epoch, val_loss, os.path.join("checkpoints", "best_rqgat.pt"))
            if verbose:
                print(f"Checkpoint saved (val loss: {val_loss/len(val_loader):.4f})")
        
        if early_stop:
            should_stop = early_stop.step(val_loss, epoch)
            if should_stop:
                if verbose:
                    print(f"Early stopping at epoch {epoch+1}")
                final_epoch = epoch+1
                break
    return train_losses, val_losses, train_recon_losses, val_recon_losses, train_rqvae_losses, val_rqvae_losses, train_entropy_losses, val_entropy_losses, final_epoch, kl

train_losses, val_losses, train_recon_losses, val_recon_losses, train_rqvae_losses, val_rqvae_losses, train_entropy_losses, val_entropy_losses, final_epoch, kl = train_rqgat(rqgat, optimizer, train_rqgat_loader, val_rqgat_loader, early_stop=early_stopping)

plot_rqgat_training(train_losses, val_losses, train_recon_losses, val_recon_losses, train_rqvae_losses, val_rqvae_losses, train_entropy_losses, val_entropy_losses, final_epoch=final_epoch)
plot_kl_divergence(kl, final_epoch=final_epoch)

# Plot the centroids for the last epoch to visualise how the codebooks are distributed in the latent space
plot_loader = DataLoader(val_rqgat_dataset, collate_fn=collate_fn, batch_size=len(val_rqgat_dataset), shuffle=False)
for _, x, edge_index in plot_loader:
    x = x.to(device)
    edge_index = edge_index.to(device)
    _, _, _, _, _ = rqgat(x, edge_index, plot=True)
