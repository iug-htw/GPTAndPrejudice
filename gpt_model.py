import torch
import torch.nn as nn

from dummy_transformer_block import TransformerBlock
from dummy_layer_norm import LayerNorm

class GPTModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])
        self.pos_emb = nn.Embedding(cfg["context_length"], cfg["emb_dim"])
        self.drop_emb = nn.Dropout(cfg["drop_rate"])

        self.trf_blocks = nn.Sequential(
            *[TransformerBlock(cfg) for _ in range(cfg["n_layers"])])

        self.final_norm = LayerNorm(cfg["emb_dim"])
        self.out_head = nn.Linear(cfg["emb_dim"], cfg["vocab_size"], bias=False)

    def forward(self, in_idx, output_hidden_states=False):
        '''
        Computes token and positional embeddings, applies dropout, processes
        the input through the transformer blocks, normalizes the output, and
        computes the logits for the output vocabulary with a linear output layer.

        If `output_hidden_states=True`, returns all hidden states.
        '''

        _, seq_len = in_idx.shape
        tok_embeds = self.tok_emb(in_idx)
        pos_embeds = self.pos_emb(torch.arange(seq_len, device=in_idx.device))
        x = tok_embeds + pos_embeds  # Shape [batch_size, num_tokens, emb_size]
        x = self.drop_emb(x)

        hidden_states = []  # Store intermediate hidden states
        for layer in self.trf_blocks:
            x = layer(x)
            if output_hidden_states:
                hidden_states.append(x.clone())  # Store each layer's output

        x = self.final_norm(x)
        logits = self.out_head(x)

        if output_hidden_states:
            return logits, hidden_states
        else:
            return logits