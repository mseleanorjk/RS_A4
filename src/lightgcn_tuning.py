import optuna
from torch.utils.data import DataLoader
import torch
import time

from train_lightgcn import train_lightgcn_model
from data_processor import DataProcessor
from lightgcn import LightGCN
from datasets import BPRDataset
from utils import *

def objective(trial):
    gamma = trial.suggest_float("gamma", 0.9, 1.0)
    learning_rate= trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True)
    weight_decay=trial.suggest_float("weight_decay", 1e-4, 1e-1, log=True)
    layers = trial.suggest_int("layers", 1, 4)
    latent_dim = trial.suggest_categorical("latent_dim", [32, 64, 128])
    batch_size = trial.suggest_categorical("batch_size", [64, 128, 256])
    reg_weight=trial.suggest_float("reg_weight", 0.001, 0.1, log=True)
    
    train_loader = DataLoader(
        BPRDataset(train, user_to_idx, item_to_idx),
        batch_size=batch_size,
        shuffle=True
    )
    
    lightgcn = LightGCN(num_users=train["user_id"].nunique(), num_items=train["item_id"].nunique(), dim=latent_dim, n_layers=layers, gamma=gamma, reg_weight=reg_weight).to(device)
    lightgcn.init_item_embeddings(embeddings, item_to_idx, item_ids)
    #rqgat.rvq.initialize_codebooks(x, edge_index, rqgat.encoder, device) # initialise the codebooks with kmeans before training
    optimizer = torch.optim.AdamW(lightgcn.parameters(), lr=learning_rate, weight_decay=weight_decay)
    
    train_losses, _, _ = train_lightgcn_model(
        lightgcn,
        optimizer,
        train_loader,
        edge_index,
        train,
        val,
        user_to_idx,
        item_to_idx,
        epochs=30,
        verbose=False,
        eval=False
    )
    recall_10, ndcg10 = evaluate(lightgcn, edge_index, val, user_to_idx, item_to_idx, train, k=10)
    recall_5, ndcg5 = evaluate(lightgcn, edge_index, val, user_to_idx, item_to_idx, train, k=5)
    
    trial.set_user_attr("train_loss", sum(train_losses[:-5])/len(train_losses[:-5]))
    trial.set_user_attr("ndcg10", ndcg10)
    trial.set_user_attr("ndcg5", ndcg5)
    trial.set_user_attr("recall10", recall_10)
    trial.set_user_attr("recall5", recall_5)
    
    del lightgcn, train_loader, train_losses, optimizer
    
    return recall_10

save_name = f"tuning/lightgcn_experiment_{time.time()}.csv"
study = optuna.create_study(
            storage='sqlite:///db.sqlite3',
            study_name=save_name,
            direction='maximize',
            pruner = optuna.pruners.MedianPruner(n_warmup_steps=10),
            load_if_exists=True
        )

train = DataProcessor("train.csv").df
val = DataProcessor("test.csv").df
edge_index, user_to_idx, item_to_idx = build_graph(train)
metadata = DataProcessor("item_meta.csv").add_sequence()
item_ids, embeddings = get_item_embeddings(metadata)

study.optimize(objective, n_trials=50, n_jobs=1)#type:ignore

def save_to_csv(study, filename):
    df = study.trials_dataframe()
    df.to_csv(filename, index=False)
    return df

_ = save_to_csv(study, save_name)
print(f"Study results saved in {save_name}")