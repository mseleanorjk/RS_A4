from plotly.subplots import make_subplots
import plotly.graph_objects as go
from collections import defaultdict
import plotly.express as px
import plotly.graph_objects as go
import numpy as np

from config import NUM_CODEBOOKS

def plot_rqgat_training(train_losses, val_losses, train_recon_losses, val_recon_losses, train_rqvae_losses, val_rqvae_losses, train_entropy_losses, val_entropy_losses, final_epoch=50, save=True):
    fig = make_subplots(
        rows=2, cols=3,
        specs=[
            [{"colspan": 3}, None, None],
            [{}, {}, {}]
        ]
    )
    fig.add_trace(go.Scatter(x=[x for x in range(final_epoch)],
                            y=train_losses,
                            mode="lines",
                            name="Train loss",
                            line=dict(color="blue")), row=1, col=1)
    fig.add_trace(go.Scatter(x=[x for x in range(final_epoch)],
                            y=val_losses,
                            mode="lines",
                            name="Validation loss",
                            line=dict(color="red")), row=1, col=1)
    fig.update_yaxes(row = 1, col =1, title_text="Total Loss", gridcolor="lightgrey")
    fig.update_xaxes(row=1, col=1, title_text="Epoch")

    fig.add_trace(go.Scatter(x=[x for x in range(final_epoch)],
                            y=train_recon_losses,
                            mode="lines", opacity=0.3,
                            showlegend=False,
                            name="Train reconstruction loss",
                            line=dict(color="blue")), row=2, col=1)
    fig.add_trace(go.Scatter(x=[x for x in range(final_epoch)],
                            y=val_recon_losses,
                            mode="lines", opacity=0.3,
                            showlegend=False,
                            name="Validation reconstruction loss",
                            line=dict(color="red")), row=2, col=1)
    fig.update_yaxes(row=2, col=1, title_text="Reconstruction loss", gridcolor="lightgrey")
    fig.update_xaxes(row=2, col=1, title_text="Epoch")

    fig.add_trace(go.Scatter(x=[x for x in range(final_epoch)],
                            y=train_rqvae_losses,
                            mode="lines", opacity=0.3,
                            showlegend=False,
                            name="Train RQ-GAT loss",
                            line=dict(color="blue")), row=2, col=2)
    fig.add_trace(go.Scatter(x=[x for x in range(final_epoch)],
                            y=val_rqvae_losses,
                            mode="lines", opacity=0.3,
                            showlegend=False,
                            name="Validation RQ-GAT loss",
                            line=dict(color="red")), row=2, col=2)
    fig.update_yaxes(row=2, col=2, title_text="RQ-GAT loss", gridcolor="lightgrey")
    fig.update_xaxes(row=2, col=2, title_text="Epoch")
    
    fig.add_trace(go.Scatter(x=[x for x in range(final_epoch)],
                            y=train_entropy_losses,
                            mode="lines", opacity=0.3,
                            showlegend=False,
                            name="Train entropy loss",
                            line=dict(color="blue")), row=2, col=3)
    fig.add_trace(go.Scatter(x=[x for x in range(final_epoch)],
                            y=val_entropy_losses,
                            mode="lines", opacity=0.3,
                            showlegend=False,
                            name="Validation entropy loss",
                            line=dict(color="red")), row=2, col=3)
    fig.update_yaxes(row=2, col=3, title_text="Entropy loss", gridcolor="lightgrey")
    fig.update_xaxes(row=2, col=3, title_text="Epoch")

    fig.update_layout(title=go.layout.Title(text="Loss components per training epoch",
                                            font=go.layout.title.Font(size=30)),
                    plot_bgcolor="white", height=700, width=1200, legend=dict(font=dict(size=17)))
    if save:
        fig.write_image("images/rqgat_losses.png")
    else:
        fig.show()

def plot_lightgcn_training(train_losses, final_epoch, save=True):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=[x for x in range(final_epoch)],
                            y=train_losses,
                            mode="lines",
                            name="Train loss",
                            line=dict(color="black")))
    fig.update_yaxes(title_text="Loss", gridcolor="lightgrey")
    fig.update_xaxes(title_text="Epoch")
    
    fig.update_layout(title=go.layout.Title(text="LightGCN training loss per epoch",
                                            font=go.layout.title.Font(size=30)),
                    plot_bgcolor="white", height=700, width=1200, legend=dict(font=dict(size=17)))
    if save:
        fig.write_image("images/lightgcn_loss.png")
    else:
        fig.show()

def plot_metrics(metrics, save=True):
    recall10s, ndcg10s, recall5s, ndcg5s = metrics
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=[x for x in range(0,101,5)],
                            y=ndcg5s,
                            mode="lines",
                            name="NCDG@5",
                            line=dict(color="lightgreen")))
    fig.add_trace(go.Scatter(x=[x for x in range(0,101,5)],
                            y=recall5s,
                            mode="lines",
                            name="Recall@5",
                            line=dict(color="gold")))
    fig.add_trace(go.Scatter(x=[x for x in range(0,101,5)],
                            y=ndcg10s,
                            mode="lines",
                            name="NCDG@10",
                            line=dict(color="green")))
    fig.add_trace(go.Scatter(x=[x for x in range(0,101,5)],
                            y=recall10s,
                            mode="lines",
                            name="Recall@10",
                            line=dict(color="darkorange")))
    fig.update_yaxes(title_text="Metric value", gridcolor="lightgrey")
    fig.update_xaxes(title_text="Epoch", tickvals=[0,5,10,15,20,25,30,35,40,45,50,55,60,65,70,75,80,85,90,95,100])
    fig.update_layout(title=go.layout.Title(text="Validation metrics for the LightGCN model",
                                            font=go.layout.title.Font(size=20)),
                    plot_bgcolor="white")
    if save:
        fig.write_image("images/lightgcn_metrics.png")
    else:
        fig.show()

def plot_kl_divergence(kl, final_epoch=50, save=True):
    fig = go.Figure()
    for i in range(NUM_CODEBOOKS):
        fig.add_trace(go.Scatter(x=[x for x in range(final_epoch)],
                                y=kl[i],
                                mode="lines",
                                name=f"Codebook {i+1}", opacity = 0.5))
    kl_sum =[sum(kl[i][epoch] for i in range(NUM_CODEBOOKS)) for epoch in range(final_epoch)]
    kl_avg = [kl_sum[epoch]/NUM_CODEBOOKS for epoch in range(final_epoch)]

    fig.add_trace(go.Scatter(x=[x for x in range(final_epoch)],
                            y=kl_avg, mode="lines",
                            name="Average KL divergence",
                            line=dict(color="black",dash="dash")))
    fig.update_layout(title=go.layout.Title(text="KL divergence of codebook usage distribution from uniform",
                                            font=go.layout.title.Font(size=22)),
                    plot_bgcolor="white", height=500, width=800, legend=dict(font=dict(size=13)))
    fig.update_yaxes(title_text="KL Divergence", gridcolor="lightgrey")
    fig.update_xaxes(title_text="Epoch")
    if save:
        fig.write_image("images/kl_divergence.png")
    else:
        fig.show()

def plot_centroids(z, centroids, level, save_path=None):
    """
    Plot residuals vs centroids for a codebook level.
    
    Args:
        z: Residuals (tensor or numpy array)
        centroids: Centroid positions (tensor or numpy array)
        level: Codebook level number
        save_path: Optional path to save image. If None, shows interactively.
    """
    import torch
    
    # Convert tensors to numpy
    if isinstance(z, torch.Tensor):
        z = z.detach().cpu().numpy()
    if isinstance(centroids, torch.Tensor):
        centroids = centroids.detach().cpu().numpy()
    
    fig = make_subplots(rows=4, cols=4)
    pairs = [(i, i+1) for i in range(0, 32, 2)]
    
    for idx, (d1, d2) in enumerate(pairs):
        row = idx // 4 + 1
        col = idx % 4 + 1
        show = idx == 0

        fig.add_trace(go.Scatter(x=z[:, d1], y=z[:, d2], mode="markers", name="Data",
                                showlegend=show, marker=dict(color="blue", opacity=0.3, size=4)), row=row, col=col)
        fig.add_trace(go.Scatter(x=centroids[:, d1], y=centroids[:, d2], mode="markers", name="Centroids",
                                showlegend=show, marker=dict(color="red", size=8)), row=row, col=col)
        fig.update_xaxes(title_text=f"Dim {d1}", row=row, col=col, gridcolor="lightgrey")
        fig.update_yaxes(title_text=f"Dim {d2}", row=row, col=col, gridcolor="lightgrey")
    
    title_text = f"Codebook {level+1} centroids plotted over {'data' if level == 0 else 'residuals'} in two dimensions"
    fig.update_layout(plot_bgcolor="white", height=1300, width=1500, 
                     title=go.layout.Title(text=title_text, font=go.layout.title.Font(size=30)), 
                     legend=dict(font=dict(size=20)))
    if save_path:
        fig.write_image(save_path)
    else:
        fig.show()

def plot_collisions(suffixes, collisions, save=True):
    fig = px.histogram(x=suffixes, color_discrete_sequence=['black'])
    fig.update_xaxes(title_text="Number of items in the bucket (collisions)", gridcolor="white")
    fig.update_yaxes(title_text="Buckets", gridcolor="lightgrey")
    fig.update_layout(plot_bgcolor="white", height=500, width=800, title=go.layout.Title(text="Distribution of collisions across buckets",
                                            font=go.layout.title.Font(size=20)))
    fig.add_annotation(x=15, y=4200,
            text=f"Total collisions: {collisions}",
            showarrow=False,
            xshift=0, yshift=0)
    if save:
        fig.write_image("images/collisions.png")
    else:
        fig.show()

def plot_descriptives(purchases, users, save=True):
    fig = px.histogram(x=purchases, color_discrete_sequence=['black'])
    fig.update_xaxes(title_text="Purchases", gridcolor="white", showline=True, linewidth=1, linecolor='black', tickvals=[t for t in range(0, max(purchases), 10)])
    fig.update_yaxes(title_text="Users", gridcolor="lightgrey")
    fig.update_layout(plot_bgcolor="white", height=500, width=800, title=go.layout.Title(text="Histogram of user purchases history length",
                                            font=go.layout.title.Font(size=20)))
    fig.add_vline(np.mean(purchases), line_dash="dash", line_color="red",
                annotation_text=f"Average purchases: {np.mean(purchases):.2f}", annotation_position="top right", annotation_font_color="red",
                annotation_xshift=10, annotation_yshift=-30)
    fig.add_annotation(x=110, y=5500,
            text=f"Total users: {users}",
            showarrow=False,
            xshift=0, yshift=0)
    if save:
        fig.write_image("images/descriptive.png")
    else:
        fig.show()