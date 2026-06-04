from torch_geometric.nn.models import GAT
import torch
from rvq import ResidualVectorQuantizer
from config import *

class RQGAT(torch.nn.Module):
    def __init__(self, dim_in, dim_latent, num_codebooks=NUM_CODEBOOKS, centroids=CENTROIDS, hidden_size=RQGAT_HIDDEN, heads=RQGAT_HEADS, layers=GAT_LAYERS, dropout = RQGAT_DROPOUT, weight_commit=WEIGHT):
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
        self.rvq = ResidualVectorQuantizer(dim_latent, centroids=centroids, num_codebooks=num_codebooks, weight = weight_commit)
        
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
        self.count_list = [torch.zeros(centroids*2**c, device = device) for c in range(num_codebooks)]

    def reset_codebook_util(self):
        # Call this at every epoch to reset the counts
        for count_tensor in self.count_list:
            count_tensor.zero_()

    def compute_codebook_util(self):
        utilisation = []
        for _, count_tensor in enumerate(self.count_list):
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
    
    def codebook_entropy_loss(self):
        """Differentiable entropy loss using assignment distributions."""
        losses = []
        for i, _ in enumerate(self.rvq.codebooks):
            counts = self.count_list[i].detach()
            probs = counts / (counts.sum() + 1e-10)
            entropy = -(probs * torch.log(probs + 1e-10)).sum()
            max_entropy = torch.log(torch.tensor(counts.size(0), dtype=torch.float, device=counts.device))
            # Normalise by max entropy so collapse penalty is comparable across different codebook sizes.
            losses.append((max_entropy - entropy) / (max_entropy + 1e-10))
        return torch.stack(losses).mean()

    def forward(self, x, edge_index, plot=False):
        x_norm = torch.nn.functional.normalize(x, dim=-1)
        z = self.encoder(x_norm, edge_index)
        semantic_ids, residual, rvq_loss = self.rvq(z, plot=plot)
        z_q = z - residual
        z_q_st = z + (z_q - z).detach() # straight-through because argmin is not differentiable
        x_hat = self.decoder(z_q_st, edge_index)

        # Add computation of codebook utilisation during the forward pass
        for codebook in range(semantic_ids.shape[1]):
            codebook_idx, codebook_counts = semantic_ids[:, codebook].unique(return_counts=True)
            self.count_list[codebook].scatter_add_(0, codebook_idx, codebook_counts.to(device, dtype=torch.float32).detach())
        return x_hat, z, z_q, semantic_ids, rvq_loss