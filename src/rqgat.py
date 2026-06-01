from torch_geometric.nn.models import GAT
import torch
from rvq import ResidualVectorQuantizer
from config import *

class RQGAT(torch.nn.Module):
    def __init__(self, dim_in, dim_latent, num_codebooks=NUM_CODEBOOKS, centroids=CENTROIDS, hidden_size=RQVAE_HIDDEN, heads=RQGAT_HEADS, layers=GAT_LAYERS, dropout = RQGAT_DROPOUT):
        super().__init__()
        self.encoder = GAT(
            in_channels=dim_in,
            hidden_channels=hidden_size,
            num_layers=layers,
            out_channels=dim_latent,
            heads=heads,
            dropout=dropout,
            act='relu',
            norm='layer_norm',
        )
        self.rvq = ResidualVectorQuantizer(dim_latent, centroids=centroids, num_codebooks=num_codebooks, weight = WEIGHT)
        
        self.decoder = GAT(
            in_channels=dim_latent,
            hidden_channels=hidden_size,
            num_layers=layers,
            out_channels=dim_in,
            heads=heads,
            dropout=dropout,
            act='relu',
            norm='layer_norm',
        )
        self.num_codebooks = num_codebooks
        # Initialise a count for the codes to implement codebook utilisation tracking
        self.count_list = [torch.zeros(centroids*2**c, device = "cuda") for c in range(num_codebooks)]

    def reset_codebook_util(self):
      # Call this at every epoch to reset the counts
      for count_tensor in self.count_list:
        count_tensor.zero_()

    def compute_codebook_util(self):
        utilisation = []
        for i, count_tensor in enumerate(self.count_list):
            # fetch the tensor that corresponds to this codebook
            # Compute the distribution of the true counts and initialise a uniform distribution to compare it to
            distr_counts = count_tensor/count_tensor.sum()
            uniform_distr = torch.ones_like(distr_counts)/distr_counts.shape[0]

            # Compare using KL-divergence
            divergence = torch.sum(distr_counts * torch.log((distr_counts + 1e-10) / uniform_distr)).item()
            # Compute also how many codes were actually utilised
            used_codes = (count_tensor != 0).sum().item()
            utilisation.append((divergence, used_codes))
        return utilisation

    def forward(self, x):
        x_norm = torch.nn.functional.normalize(x, dim=-1)
        z = self.encoder(x_norm)
        semantic_ids, residual, rvq_loss = self.rvq(z)
        z_q = z - residual
        z_q_st = z + (z_q - z).detach() # straight-through because argmin is not differentiable
        x_hat = self.decoder(z_q_st)

        # Add computation of codebook utilisation during the forward pass
        if not self.training:
          for codebook in range(semantic_ids.shape[1]):
            codebook_idx, codebook_counts = semantic_ids[:, codebook].unique(return_counts=True)
            self.count_list[codebook].scatter_add_(0, codebook_idx, codebook_counts.to(device, dtype=torch.float32))
        return x_hat, z, z_q, semantic_ids, rvq_loss