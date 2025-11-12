import torch.nn as nn

from .activation import GELU

class FeedForward(nn.Module):
    '''
    Feed-forward neural network with GELU activation function.
    - Multi-Head Self-Attention → Captures relationships between tokens.
    - Feedforward Neural Network (FFN) → Processes each token independently after attention.
    '''
    
    def __init__(self, cfg):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(cfg["emb_dim"], 4 * cfg["emb_dim"]),
            GELU(),
            nn.Linear(4 * cfg["emb_dim"], cfg["emb_dim"]),
        )
        
    def forward(self, x):
        return self.layers(x)