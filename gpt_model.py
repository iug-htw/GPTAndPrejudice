
import torch
import torch.nn as nn
from typing import Dict, Any, Optional, Tuple

from dummy_transformer_block import TransformerBlock
from dummy_layer_norm import LayerNorm

class InterventionPlan:
    """
    Hook object consulted during forward() to optionally replace activations.
    Override any of the methods below in your experiment code.
    """
    def maybe_replace_resid_pre(self, layer_idx: int, x: torch.Tensor) -> torch.Tensor:
        return x
    def maybe_replace_resid_post(self, layer_idx: int, x: torch.Tensor) -> torch.Tensor:
        return x
    # Optional: only works if your blocks expose per-head z or mlp outputs.
    def maybe_replace_head_z(self, layer_idx: int, z: torch.Tensor) -> torch.Tensor:
        return z
    def maybe_replace_mlp_out(self, layer_idx: int, h: torch.Tensor) -> torch.Tensor:
        return h
    
DEFAULT_CFG = {
  "vocab_size": 50257,
  "context_length": 256,
  "emb_dim": 896,
  "n_heads": 14,
  "n_layers": 8,
  "drop_rate": 0.2,
  "qkv_bias": True
}

class GPTModel(nn.Module):
    def __init__(self, cfg: Dict[str, Any]):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"]
        )
        self.pos_emb = nn.Embedding(cfg["context_length"], cfg["emb_dim"]
        )
        self.drop_emb = nn.Dropout(cfg["drop_rate"]
        )
        self.trf_blocks = nn.Sequential(
            *[TransformerBlock(cfg) for _ in range(cfg["n_layers"])]
        )
        self.final_norm = LayerNorm(cfg["emb_dim"]
        )
        self.out_head = nn.Linear(cfg["emb_dim"], cfg["vocab_size"], bias=False)

    @torch.no_grad()
    def cache_forward(self, in_idx: torch.Tensor):
        """Run a forward pass with caching enabled (no interventions)."""
        return self.forward(in_idx, enable_cache=True)

    def forward(
        self,
        in_idx: torch.Tensor,
        enable_cache: bool = False,
        intervention_plan: Optional[InterventionPlan] = None,
        output_hidden_states: bool = False,
        output_attentions_weights: bool = False,
        # Backward-compat args (ignored if plan is provided)
        intervene_layer: Optional[int] = None,
        edited_hidden: Optional[torch.Tensor] = None,
    ):
        """
        Mechanistic interpretability-friendly forward.
        Returns: logits, (optional) cache, (optional) hidden_states, (optional) attn_weights
        Cache keys: resid_pre[L], resid_post[L], attn_weights[L]
        """
        B, T = in_idx.shape
        device = in_idx.device

        tok_embeds = self.tok_emb(in_idx)                    # [B,T,d]
        pos_embeds = self.pos_emb(torch.arange(T, device=device))  # [T,d]
        x = self.drop_emb(tok_embeds + pos_embeds)

        cache: Dict[str, Dict[int, torch.Tensor]] = {}
        if enable_cache:
            cache = {"resid_pre": {}, "resid_post": {}, "attn_weights": {}}

        hidden_states = []
        attention_weights_per_layer = []

        # Fallback to legacy single-layer intervention if no plan provided
        legacy_layer = intervene_layer if (intervention_plan is None) else None
        legacy_edit = edited_hidden if (intervention_plan is None) else None

        for L, block in enumerate(self.trf_blocks):
            if enable_cache:
                cache["resid_pre"][L] = x.detach()

            # Entry-point intervention
            if intervention_plan is not None:
                x = intervention_plan.maybe_replace_resid_pre(L, x)
            elif legacy_layer is not None and legacy_edit is not None and L == legacy_layer:
                x = legacy_edit  # Inject casual intervention

            # Run block (assumed to return (x_out, attn_weights))
            block_out = block(x)
            if isinstance(block_out, tuple) and len(block_out) == 2:
                x, attn_w = block_out
            else:
                x = block_out
                attn_w = None

            if output_attentions_weights and attn_w is not None:
                attention_weights_per_layer.append(attn_w.detach())
                if enable_cache:
                    cache["attn_weights"][L] = attn_w.detach()

            # Exit-point intervention (rarely used, but handy)
            if intervention_plan is not None:
                x = intervention_plan.maybe_replace_resid_post(L, x)

            if output_hidden_states:
                hidden_states.append(x.clone())

            if enable_cache:
                cache["resid_post"][L] = x.detach()

        x = self.final_norm(x)
        logits = self.out_head(x)

        outputs = (logits,)
        if enable_cache:
            outputs += (cache,)
        if output_hidden_states:
            outputs += (hidden_states,)
        if output_attentions_weights:
            outputs += (attention_weights_per_layer,)
        return outputs if len(outputs) > 1 else outputs[0]
