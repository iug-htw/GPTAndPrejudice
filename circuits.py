
"""circuits.py
Minimal utilities for activation caching, layer-entry patching, and causal ranking.
Works with the circuit-friendly GPTModel.forward(enable_cache=True, intervention_plan=...).
"""
from typing import Dict, Tuple, Optional, List
import torch
import torch.nn.functional as F

# ---- Metric ----
def next_token_logit_diff(logits: torch.Tensor, target_id: int, distractor_id: int) -> float:
    """(logit[target] - logit[distractor]) at final position, averaged over batch."""
    last = logits[:, -1, :]
    return (last[:, target_id] - last[:, distractor_id]).mean().item()

# ---- Plans ----
class ResidPrePatchPlan:
    def __init__(self, clean_cache: Dict, L: int):
        self.clean_cache = clean_cache
        self.L = L
    def maybe_replace_resid_pre(self, layer_idx: int, x: torch.Tensor) -> torch.Tensor:
        if layer_idx == self.L:
            return self.clean_cache["resid_pre"][self.L].to(x.device)
        return x
    def maybe_replace_resid_post(self, layer_idx: int, x: torch.Tensor) -> torch.Tensor: return x
    def maybe_replace_head_z(self, layer_idx: int, z: torch.Tensor) -> torch.Tensor: return z
    def maybe_replace_mlp_out(self, layer_idx: int, h: torch.Tensor) -> torch.Tensor: return h

# ---- Core helpers ----
@torch.no_grad()
def cache_clean(model, toks_clean: torch.Tensor):
    model.eval()
    logits, cache = model(toks_clean, enable_cache=True)
    return logits, cache

@torch.no_grad()
def baseline_score(model, toks: torch.Tensor, target_id: int, distractor_id: int) -> float:
    model.eval()
    logits = model(toks)
    return next_token_logit_diff(logits, target_id, distractor_id)

@torch.no_grad()
def patch_layer_and_score(model, toks_corrupt: torch.Tensor, clean_cache: Dict, L: int, target_id: int, distractor_id: int) -> float:
    plan = ResidPrePatchPlan(clean_cache, L)
    logits = model(toks_corrupt, enable_cache=False, intervention_plan=plan)
    if isinstance(logits, tuple):
        logits = logits[0]
    return next_token_logit_diff(logits, target_id, distractor_id)

@torch.no_grad()
def rank_layers_by_gain(model, toks_clean: torch.Tensor, toks_corrupt: torch.Tensor, target_id: int, distractor_id: int, n_layers: int) -> List[Tuple[int, float]]:
    _, clean_cache = cache_clean(model, toks_clean)
    base = baseline_score(model, toks_corrupt, target_id, distractor_id)
    out = []
    for L in range(n_layers):
        s = patch_layer_and_score(model, toks_corrupt, clean_cache, L, target_id, distractor_id)
        out.append((L, s - base))
    out.sort(key=lambda x: x[1], reverse=True)
    return out
