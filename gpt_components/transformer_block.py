import torch.nn as nn

from .feed_forward import FeedForward
from .multi_head_attention import MultiHeadAttention

class TransformerBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.att = MultiHeadAttention(
            d_in=cfg["emb_dim"],
            d_out=cfg["emb_dim"],
            context_length=cfg["context_length"],
            num_heads=cfg["n_heads"], 
            dropout=cfg["drop_rate"],
            qkv_bias=cfg["qkv_bias"])
        self.ff = FeedForward(cfg)
        self.norm1 = nn.LayerNorm(cfg["emb_dim"])
        self.norm2 = nn.LayerNorm(cfg["emb_dim"])
        self.drop_shortcut = nn.Dropout(cfg["drop_rate"])
        

    def forward(self, x):
        '''
        The transformer block consists of two main components:
        - Multi-Head Self-Attention → Captures relationships between tokens.
        - Feedforward Neural Network (FFN) → Processes each token independently after attention.

        The output of the attention block is added to the input of the block (skip connection),
        which is then normalized using LayerNorm. The output is then passed through the FFN,    
        and the result is again added to the input of the block and normalized.

        The dropout is applied to the skip connections before adding them to the output of the
        attention and FFN blocks. This helps to prevent overfitting and improves generalization.

        Args:
            x (torch.Tensor): Input tensor of shape [batch_size, seq_len, emb_dim].

        Returns:
            torch.Tensor: Output tensor of shape [batch_size, seq_len, emb_dim].
        '''

        shortcut = x
        x = self.norm1(x)
        x, _attn_weights = self.att(x)
        x = self.drop_shortcut(x)
        x = x + shortcut

        shortcut = x
        x = self.norm2(x)
        x = self.ff(x)
        x = self.drop_shortcut(x)
        x = x + shortcut

        return x, _attn_weights
            
    