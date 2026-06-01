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

train = DataProcessor("train.csv").df
metadata_processor = DataProcessor("item_meta.csv")
test = DataProcessor("test.csv").df
metadata = metadata_processor.add_sequence()

emb_dict = get_item_embeddings(metadata)

rqgat = RQGAT(dim_in=len(emb_dict.values()[0]), dim_latent=32)
train_rqgat_dataset = RQGATDataset(emb_dict[:-1])
val_rqgat_dataset = RQGATDataset(emb_dict[-1])
train_rqgat_loader = DataLoader(train_rqgat_dataset)
val_rqgat_loader = DataLoader(val_rqgat_dataset)

def rqgat_epoch(model, optimizer, train_loader, val_loader, scheduler=None, **kwargs):
    # Training
    model.train()
    train_loss = 0

    for _, x in train_loader:
        x = x.to(device)
        x_hat, _, _, _, gat_loss = model(x)
        loss = reconstruction_loss(x_hat, x).mean() + gat_loss

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        train_loss += loss.item()
    
    avg_train_loss = train_loss/len(train_rqgat_loader)

    # Validation
    model.eval()
    model.reset_codebook_util()
    val_loss = 0

    with torch.no_grad():
        for _, x in val_loader:
            x = x.to(device)
            x_hat, _, _, _, rqv_loss = model(x)
            loss = reconstruction_loss(x_hat, x).mean() + rqv_loss
            val_loss += loss.item()

    avg_val_loss = val_loss / len(val_loader)

    if scheduler:
        scheduler.step(avg_val_loss)

    # Print the utilisation stats
    utilisation = model.compute_codebook_util()
    kl = [util[0] for util in utilisation]
    return avg_train_loss, avg_val_loss, kl

EPOCHS=50
optimizer = torch.optim.Adam(rqgat.parameters(), lr=RQVAE_LR)
early_stopping = EarlyStopping_RQGAT(patience=5, delta=0.01, warmup_epochs=10)

for epoch in range(EPOCHS):
    train_loss, val_loss, kl = rqgat_epoch(rqgat, optimizer, train_rqgat_loader, val_rqgat_loader)
    print(f"Epoch {epoch+1}/{EPOCHS} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
    print(*(f"Codebook {i}: KL divergence = {kl}" for i, kl in enumerate(kl, start=1)), sep=", ")
    should_stop = early_stopping.step(val_loss, epoch, rqgat)
    if should_stop:
        print(f"Early stopping at epoch {epoch+1}")
        final_epoch = epoch+1
        break