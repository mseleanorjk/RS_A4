from plotly.subplots import make_subplots
import plotly.graph_objects as go
import plotly.express as px
import plotly.graph_objects as go
import numpy as np

from config import EPOCHS

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
    fig.add_trace(go.Scatter(x=[x for x in range(0,EPOCHS+1,5)],
                            y=ndcg5s,
                            mode="lines",
                            name="NCDG@5",
                            line=dict(color="lightgreen")))
    fig.add_trace(go.Scatter(x=[x for x in range(0,EPOCHS+1,5)],
                            y=recall5s,
                            mode="lines",
                            name="Recall@5",
                            line=dict(color="gold")))
    fig.add_trace(go.Scatter(x=[x for x in range(0,EPOCHS+1,5)],
                            y=ndcg10s,
                            mode="lines",
                            name="NCDG@10",
                            line=dict(color="green")))
    fig.add_trace(go.Scatter(x=[x for x in range(0,EPOCHS+1,5)],
                            y=recall10s,
                            mode="lines",
                            name="Recall@10",
                            line=dict(color="darkorange")))
    fig.update_yaxes(title_text="Metric value", gridcolor="lightgrey")
    fig.update_xaxes(title_text="Epoch", tickvals=[i for i in range(0,EPOCHS+1,5)])
    fig.update_layout(title=go.layout.Title(text="Validation metrics for the LightGCN model",
                                            font=go.layout.title.Font(size=20)),
                    plot_bgcolor="white")
    if save:
        fig.write_image("images/lightgcn_metrics.png")
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