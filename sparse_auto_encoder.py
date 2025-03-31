import torch
import torch.nn as nn
import torch.nn.functional as F

class TopKActivation(nn.Module):
    def __init__(self, k):
        super(TopKActivation, self).__init__()
        self.k = k

    def forward(self, x):
        # Zero out all but top-k activations per batch element
        topk_vals, _ = torch.topk(x, self.k, dim=1)
        threshold = topk_vals[:, -1].unsqueeze(1)
        return torch.where(x >= threshold, x, torch.zeros_like(x))

class SparseAutoencoder(nn.Module):
    def __init__(self, input_dim=512, hidden_dim=2048, top_k=50):
        super(SparseAutoencoder, self).__init__()

        self.decoder = nn.Linear(hidden_dim, input_dim)
        self.pre_encoder_bias = nn.Parameter(torch.zeros(input_dim))  # Pre-encoder bias
        self.encoder = nn.Linear(input_dim, hidden_dim)
        self.topk = TopKActivation(top_k)

    def forward(self, x):
        # Apply pre-encoder bias: x - decoder.bias
        x = x - self.decoder.bias - self.pre_encoder_bias

        # Encode and apply top-k sparsity
        z = self.encoder(x)
        z = self.topk(z)

        # Decode
        recon = self.decoder(z)
        return recon, z
    
    def intervene_and_decode(self, x, neuron_idx, boost=5.0):
        """
        Intervene in SAE latent space by boosting specific neuron(s).
        Args:
            x: input tensor from LLM (hidden state) [batch, seq_len, input_dim]
            neuron_idx: int or list of ints - neuron(s) to boost
            boost: value to add to the selected neuron(s)
        Returns:
            intervened hidden state in model space
        """
        x = x - self.decoder.bias - self.pre_encoder_bias
        z = self.encoder(x)

        # Boost specified neuron(s)
        if isinstance(neuron_idx, int):
            neuron_idx = [neuron_idx]
        for i in neuron_idx:
            z[..., i] += boost

        # No top-k sparsity here for causal probing
        decoded = self.decoder(z)
        return decoded