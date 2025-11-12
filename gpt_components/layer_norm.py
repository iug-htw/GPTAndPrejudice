import torch
import torch.nn as nn

class LayerNorm(nn.Module):
    def __init__(self, emb_dim):
        super().__init__()
        self.eps = 1e-6 # small value to avoid division by zero
        self.scale = nn.Parameter(torch.ones(emb_dim)) # trainable scale parameter
        self.shift = nn.Parameter(torch.zeros(emb_dim)) # trainable shift parameter

    def forward(self, x):
        '''
        In this implementation of Layer Normalization, the normalization is applied along 
        the last dimension of the input tensor 𝑋, which represents the embedding dimension (dim=-1). 
        Normalizing over the embedding dimension ensures that each word is treated independently,
        preventing one word from influencing another.

        For Transformer models, input data typically has the following shape: 
                        [batch_size, seq_len, emb_dim]
        '''

        mean = x.mean(dim=-1, keepdim=True)  
        var = x.var(dim=-1, keepdim=True, unbiased=False) # unbiased=False means that the variance is calculated with the Bessel correction
        norm_x = (x - mean) / torch.sqrt(var + self.eps)

        return self.scale * norm_x + self.shift 