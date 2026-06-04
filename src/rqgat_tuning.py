import optuna
import torch
import time
import numpy as np

from config import *
from rqgat import RQGAT
from train_rqgat import train_rqgat_model
from datasets import *
from utils import *

metadata = DataProcessor("item_meta.csv").add_sequence()
item_ids, embeddings = get_item_embeddings(metadata)

graph_cache = {}

def get_graph(k, k_split):
    key = (k, k_split)
    if key not in graph_cache:
        graph_cache[key] = build_graph(metadata, embeddings, k=k, k_split=k_split)
    return graph_cache[key]

def objective(trial):
    set_seed(42)
    # set hyperparams
    num_codebooks = trial.suggest_int("num_codebooks", 2, 6)
    centroids = trial.suggest_categorical("centroids", [2,4,8])
    rqgat_lr = trial.suggest_float("rqgat_lr", 1e-5, 1e-3, log=True)
    weight_commit = trial.suggest_float("weight", 0.1, 0.9)
    weight_decay_rqgat = trial.suggest_float("weight_decay_rqgat", 1e-4, 1e-2, log=True)
    rqgat_hidden = trial.suggest_categorical("rqgat_hidden", [32, 64, 128])
    k = trial.suggest_int("k", 5, 20)
    k_split = trial.suggest_int("k_split", 1, 4)
    rqgat_heads = trial.suggest_categorical("rqgat_heads", [1, 2, 4])
    gat_layers = trial.suggest_int("gat_layers", 1, 4)
    rqgat_dropout = trial.suggest_float("rqgat_dropout", 0.1, 0.5)
    entropy_weight = trial.suggest_float("entropy_weight", 0.01, 0.1)
    split_perc = trial.suggest_float("split_perc", 0.7, 0.9)
    
    #rqgat_dataset = RQGATDataset(metadata, item_ids, embeddings, split=split_perc, k=k, k_split=k_split)
    #x, edge_index, train_mask, val_mask = rqgat_dataset.get_full_data()
    edge_index = get_graph(k, k_split)
    n = len(item_ids)
    split = int(split_perc * n)
    train_mask = torch.zeros(n, dtype=torch.bool)
    train_mask[:split] = True
    val_mask = ~train_mask
    x = torch.from_numpy(embeddings).float()
    edge_index = edge_index.to(device)
    train_mask = train_mask.to(device)
    val_mask = val_mask.to(device)
    
    rqgat = RQGAT(dim_in=embeddings.shape[1], dim_latent=32, 
                  num_codebooks=num_codebooks, 
                  centroids=centroids, 
                  hidden_size=rqgat_hidden, 
                  heads=rqgat_heads, 
                  layers=gat_layers, 
                  dropout = rqgat_dropout, 
                  weight_commit=weight_commit)
    rqgat.rvq.initialize_codebooks(x, edge_index, rqgat.encoder, device)
    optimizer = torch.optim.AdamW(rqgat.parameters(), lr=rqgat_lr, weight_decay=weight_decay_rqgat)
    
    train_losses, val_losses, _, _, _, _, _, _, _, kl = train_rqgat_model(rqgat, optimizer, x, edge_index, train_mask, val_mask, epochs=30, entropy_weight = entropy_weight, early_stop=None, scheduler=None, verbose=False, save_checkpoints=False)
    avg_last_train_loss = np.mean(train_losses[-5:])
    trial.set_user_attr("avg_last_train_loss", avg_last_train_loss)
    avg_last_val_loss = np.mean(val_losses[-5:])
    for i, kl_values in kl.items():
        trial.set_user_attr(f"kl_{i}", kl_values[-1])
    
    del rqgat, optimizer, train_losses, val_losses, kl
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return float(avg_last_val_loss)

study = optuna.create_study(
            storage='sqlite:///db.sqlite3',
            study_name=f"rqgat_experiment_{time.time()}",
            direction='minimize',
            pruner = optuna.pruners.MedianPruner(n_warmup_steps=10),
            load_if_exists=True
        )
study.optimize(objective, n_trials=50)

def save_to_csv(study, filename):
    df = study.trials_dataframe()
    df.to_csv(filename, index=False)
    return df

_ = save_to_csv(study, f"tuning/rqgat_experiment_{time.time()}.csv")
print(f"Study results saved in tuning/rqgat_experiment_{time.time()}.csv")