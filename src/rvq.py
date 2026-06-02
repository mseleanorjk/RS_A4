import torch
import numpy as np
from sklearn.cluster import KMeans

from config import *
from plot_functions import plot_centroids

class CodeBook(torch.nn.Module):
    def __init__(self, centroids, embedding_dim = 32):
        super(CodeBook, self).__init__()
        self.centroids = centroids
        self.embedding_dim = embedding_dim
        self.layer = torch.nn.Embedding(centroids, embedding_dim)

    def forward(self):
        return self.layer.weight

class ResidualVectorQuantizer(torch.nn.Module):
    def __init__(self, dim_in, centroids = CENTROIDS, num_codebooks = NUM_CODEBOOKS, weight = WEIGHT):
        super(ResidualVectorQuantizer, self).__init__()
        self.weight = weight
        self.codebooks = torch.nn.ModuleList()

        self.codebooks = torch.nn.ModuleList([
            CodeBook(centroids=centroids*2**c, embedding_dim=dim_in)
            for c in range(num_codebooks)
        ])

    # The random codebook weights were too far from the actual embeddings,
    # so all embeddings were mapped to one centroid. This function initialises
    # the codebooks so that the centroids are distributed more closely to the
    # embeddings to avoid collapse
    def initialize_codebooks(self, dataloader, encoder, device, n_batches=10):
        encoder.eval()
        vectors = []
        # Collect a few batches of embeddings passed through the encoder
        with torch.no_grad():
            for i, (_, x, edge_index) in enumerate(dataloader):
                if i >= n_batches:
                    break
                z = encoder(x.to(device), edge_index.to(device))
                vectors.append(z.cpu())
        vectors = np.concatenate(vectors, axis=0)  # (N, dim)
        residual = vectors.copy()

        for codebook in self.codebooks:
          # kmeans clustering for qweight initalisation
            kmeans = KMeans(n_clusters=codebook.centroids, n_init=10, random_state=42) #type: ignore
            kmeans.fit(residual)

            # Initialize codebook weights with cluster centers
            centers = torch.tensor(kmeans.cluster_centers_, dtype=torch.float32).to(device)
            codebook.layer.weight.data = centers # type: ignore

            # each subsequent codebook fits kmeans with the residuals from the last (mirroring rvq)
            # Predict the closest centroid via k-means
            closest = kmeans.predict(residual)
            # Update the residuals for each run such that they are the previous residuals - the closest centroid at this iteration
            residual = residual - kmeans.cluster_centers_[closest]

    def forward(self, x, plot=False):
        residual = x
        all_indices = []
        # initialise tensor of zeroes to save the per-codebook rq-vae loss
        rvq_loss = torch.tensor(0.0, device = x.device)

        for level, codebook in enumerate(self.codebooks):
            centroids = codebook()
            # plot for debugging
            if plot:
                plot_centroids(residual.detach().cpu().numpy(),
                        centroids.detach().cpu().numpy(),
                        level, save_path=f"images/centroids_level_{level}.png")
            distances = torch.cdist(residual, centroids)
            closest_idx = torch.argmin(distances, dim=1)
            closest_centroids = centroids[closest_idx]

            # update the loss in rvq_loss per codebook, by computing the stop-gradient per-term RQVAE loss on each codebook during the forward pass
            rvq_loss += ((residual.detach() - closest_centroids)**2).mean(dim = -1).mean()
            rvq_loss += self.weight * ((residual - closest_centroids.detach())**2).mean(dim = -1).mean()

            all_indices.append(closest_idx)
            #all_centroids.append(closest_centroids)
            residual = residual - closest_centroids

        semantic_ids = torch.stack(all_indices, dim=1)
        return semantic_ids, residual, rvq_loss
