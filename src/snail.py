import torch
import torch.nn as nn
import math
 
class CausalConv1d(nn.Module):
    """
    1D Causal Convolution layer. Contains function for the forward pass.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        kernel_size (int): Size of the convolutional kernel.
        dilation (int): Dilation factor for the convolution.
    """
    def __init__(self, in_channels, out_channels, kernel_size=2, dilation=1):
        super().__init__()
        padding = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size,
                              padding=padding, dilation=dilation)
        self.padding = padding

    def forward(self, x):
        out = self.conv(x)
        return out[:, :, :x.size(2)]  # trim right padding


class DenseBlock(nn.Module):
    """
    Dense Block of 2 1D causal convolutions with tanh and sigmoid activation functions.
    Contains function to do the forward pass.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        dilation (int): Dilation factor for the convolution.
        kernel_size (int): Size of the convolutional kernel.
    """
    def __init__(self, in_channels, out_channels, dilation=1, kernel_size=2):
        super().__init__()
        self.conv_f = CausalConv1d(in_channels, out_channels, kernel_size, dilation)
        self.conv_g = CausalConv1d(in_channels, out_channels, kernel_size, dilation)

    def forward(self, x):
        xf = torch.tanh(self.conv_f(x))
        xg = torch.sigmoid(self.conv_g(x))
        out = xf * xg
        return torch.cat([x, out], dim=1)  # dim=1 is channel dim in (B, C, T)


class TemporalConvolutionalBlock(nn.Module):
    """
    Temporal Convolutional Block consisting of multiple DenseBlocks with exponentially increasing dilation rates.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        sequence_length (int): Length of the input sequence.
        kernel_size (int): Size of the convolutional kernel.
    """
    def __init__(self, in_channels, out_channels, sequence_length, kernel_size=2):
        super().__init__()
        self.blocks = nn.ModuleList()
        self.sequence_length = sequence_length
        self.kernel_size = kernel_size
        
        # number of blocks derived from sequence length as per the paper
        num_blocks = math.ceil(math.log2(sequence_length))
        
        current_channels = in_channels
        for i in range(1, num_blocks + 1):  # i = 1, ..., ceil(log2(T))
            self.blocks.append(
                DenseBlock(current_channels, out_channels, dilation=2**i, kernel_size=self.kernel_size)
            )
            current_channels += out_channels  # grows due to dense concat

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        return x


class AttentionBlock(nn.Module):
    """
    Multi-head attention block for SNAIL architecture

    Args:
        in_channels (int): Number of input channels.
        num_heads (int): Number of attention heads.
        key_size (int): Size of the key vectors.
    """
    def __init__(self, in_channels, num_heads=4, key_size=64, sequence_length=None):
        super().__init__()
        assert key_size % num_heads == 0, "key_size must be divisible by num_heads"
        
        self.proj_in = nn.Linear(in_channels, key_size)  # ← project to fixed dimension
        self.attention = nn.MultiheadAttention(
            embed_dim=key_size,   
            num_heads=num_heads,
            batch_first=True
        )
        self.proj_out = nn.Linear(key_size, in_channels)  # ← project back

        # Pre-compute causal mask once if sequence length is known at build time.
        if sequence_length is not None:
            mask = torch.triu(
                torch.full((sequence_length, sequence_length), float('-inf')),
                diagonal=1
            )
            self.register_buffer('_causal_mask', mask, persistent=False)
        else:
            self._causal_mask = None

    def forward(self, x):
        # x: (B, C, T) → (B, T, C)
        x = x.permute(0, 2, 1)
        
        # project to key_size
        x_proj = self.proj_in(x)          # (B, T, key_size)
        
        if self._causal_mask is not None:
            causal_mask = self._causal_mask
        else:
            seq_len = x_proj.size(1)
            causal_mask = torch.triu(
                torch.full((seq_len, seq_len), float('-inf'), device=x.device),
                diagonal=1
            )
        out, _ = self.attention(x_proj, x_proj, x_proj, attn_mask=causal_mask)
        
        out = self.proj_out(out)           # (B, T, C) — back to original channels
        
        # back to (B, C, T)
        return out.permute(0, 2, 1)


class SNAIL(nn.Module):
    """
    Full SNAIL network architecture with TC layers and causal self-attention.

    Args:
        observation_space (gym.Space): The observation space of the environment.
        features_dim (int): The number of features to extract.
        num_blocks (int): The number of temporal convolutional blocks.
        num_layers (int): The number of layers in the SNAIL network.
        num_heads (int): The number of attention heads.
    """
    def __init__(self, observation_space, features_dim=64, sequence_length=10, 
                 num_filters=32, num_layers=2, num_heads=4, kernel_size=2):
        super().__init__()

        self.sequence_length = sequence_length  # ← store it
        self.obs_dim = observation_space.shape[0] // sequence_length
        self.kernel_size = kernel_size
        
        self.layers = nn.ModuleList()
        current_channels = self.obs_dim
        for _ in range(num_layers):
            tcb = TemporalConvolutionalBlock(current_channels, num_filters, self.sequence_length, self.kernel_size)
            current_channels += num_filters * math.ceil(math.log2(self.sequence_length))
            attn = AttentionBlock(current_channels, num_heads, sequence_length=self.sequence_length)
            self.layers.append(nn.ModuleList([tcb, attn]))

        self.output_proj = nn.Linear(current_channels, features_dim)

    def forward(self, x):
        B = x.size(0)
        x = x.view(B, self.sequence_length, self.obs_dim) 
        x = x.permute(0, 2, 1)
        for tcb, attn in self.layers:
            x = tcb(x)
            x = attn(x)
        x = x[:, :, -1]
        return self.output_proj(x)
