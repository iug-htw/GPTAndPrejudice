import torch
import torch.nn as nn
from typing import Optional, Tuple, Union, List

class TopKActivation(nn.Module):
    def __init__(self, k: int):
        super().__init__()
        self.k = k

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Zero out all but top-k activations per batch element
        topk_vals, _ = torch.topk(x, self.k, dim=1)
        threshold = topk_vals[:, -1].unsqueeze(1)
        return torch.where(x >= threshold, x, torch.zeros_like(x))

class SparseAutoencoder(nn.Module):
    """
    SAE with explicit encode/decode helpers for circuit work.
    """
    def __init__(self, input_dim: int = 512, hidden_dim: int = 2048, top_k: int = 50):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.decoder = nn.Linear(hidden_dim, input_dim)
        self.pre_encoder_bias = nn.Parameter(torch.zeros(input_dim))  # Pre-encoder bias
        self.encoder = nn.Linear(input_dim, hidden_dim)
        self.topk = TopKActivation(top_k)

    # ---- Helpers ----
    def encode_raw(self, x: torch.Tensor) -> torch.Tensor:
        """Encode WITHOUT sparsity (useful for causal probes)."""
        x = x - self.decoder.bias - self.pre_encoder_bias
        return self.encoder(x)

    def encode_sparse(self, x: torch.Tensor) -> torch.Tensor:
        """Encode WITH sparsity (analysis)."""
        z = self.encode_raw(x)
        return self.topk(z)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        z = self.encode_sparse(x)
        recon = self.decode(z)
        return recon, z

    @torch.no_grad()
    def intervene_and_decode(self, x: torch.Tensor, neuron_idx: Union[int, List[int]], boost: float = 5.0) -> torch.Tensor:
        """Boost specific latent(s) and decode back to model space."""
        z = self.encode_raw(x)
        if isinstance(neuron_idx, int):
            neuron_idx = [neuron_idx]
        for i in neuron_idx:
            z[..., i] += boost
        decoded = self.decode(z)
        return decoded

    @torch.no_grad()
    def decode_single_latent(self, latent_id: int, scale: float = 1.0, device: Optional[torch.device] = None) -> torch.Tensor:
        oh = torch.zeros(self.hidden_dim, device=device if device is not None else self.decoder.weight.device)
        oh[latent_id] = scale
        return self.decode(oh)
