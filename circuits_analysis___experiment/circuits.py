"""
===>  MUST BE MOVED TO ROOT DIRECTORY

circuits.py
Utilities for activation caching, patching, and causal metrics.

New in this version:
- KL-to-Clean objective (distributional alignment).
- Position-masked patching (e.g., patch only the final token).
- Epsilon mixing (blend clean+corrupt instead of hard replace).
- Head patch plan stub (works if model exposes z_per_head).

You can call the rankers directly, or from circuits_cli.py.
"""
from typing import Dict, Tuple, List, Optional, Iterable, Union
import torch
import torch.nn.functional as F


# ========= Metrics =========

def next_token_logit_diff(logits: torch.Tensor, target_id: int, distractor_id: int) -> float:
    """
    (logit[target] - logit[distractor]) at final position, averaged over batch.
    """
    last = logits[:, -1, :]
    return (last[:, target_id] - last[:, distractor_id]).mean().item()


def kl_to_clean(clean_logits: torch.Tensor, test_logits: torch.Tensor) -> torch.Tensor:
    """
    KL(p_clean || q_test) at the last token.
    Returns a scalar tensor (batchmean).
    """
    p_clean = F.softmax(clean_logits[:, -1, :], dim=-1)   # reference distribution
    log_q   = F.log_softmax(test_logits[:, -1, :], dim=-1)
    # KL(p||q) = sum p * (log p - log q). torch.nn.functional.kl_div takes log q as input.
    return F.kl_div(log_q, p_clean, reduction="batchmean", log_target=False)


# ========= Patch Plans =========

class ResidPrePatchPlan:
    """
    Hard replace the entire resid_pre[L] with the clean cache.
    CAUTION: This often snaps the run fully onto the clean trajectory.
    """
    def __init__(self, clean_cache: Dict, L: int):
        self.clean_cache, self.L = clean_cache, L

    def maybe_replace_resid_pre(self, layer_idx: int, x: torch.Tensor) -> torch.Tensor:
        if layer_idx == self.L:
            return self.clean_cache["resid_pre"][self.L].to(x.device)
        return x

    def maybe_replace_resid_post(self, layer_idx: int, x: torch.Tensor) -> torch.Tensor: return x
    def maybe_replace_head_z(self, layer_idx: int, z: torch.Tensor) -> torch.Tensor: return z
    def maybe_replace_mlp_out(self, layer_idx: int, h: torch.Tensor) -> torch.Tensor: return h


class MaskedMixResidPlan:
    """
    Position-masked + epsilon-mixing plan at resid_pre[L].

    - If positions == "last": operate only on the final position (x[:, -1, :]).
    - If positions is an Iterable[int]: operate on those indices (shared across batch).
    - If eps is None: hard replace (clean -> corrupt) at the masked positions.
    - If eps is float in (0,1]: x_masked = (1-eps)*x_masked + eps*clean_masked
    """
    def __init__(
        self,
        clean_cache: Dict,
        L: int,
        positions: Union[str, Iterable[int]] = "last",
        eps: Optional[float] = 0.1,
    ):
        self.clean_cache, self.L, self.positions, self.eps = clean_cache, L, positions, eps

    def _apply(self, x: torch.Tensor, xc: torch.Tensor) -> torch.Tensor:
        if self.positions == "last":
            if self.eps is None:
                x[:, -1, :] = xc[:, -1, :]
            else:
                x[:, -1, :] = (1 - self.eps) * x[:, -1, :] + self.eps * xc[:, -1, :]
            return x

        # explicit index list
        idxs = list(self.positions)
        if self.eps is None:
            x[:, idxs, :] = xc[:, idxs, :]
        else:
            x[:, idxs, :] = (1 - self.eps) * x[:, idxs, :] + self.eps * xc[:, idxs, :]
        return x

    def maybe_replace_resid_pre(self, layer_idx: int, x: torch.Tensor) -> torch.Tensor:
        if layer_idx != self.L: 
            return x
        xc = self.clean_cache["resid_pre"][self.L].to(x.device)
        x = x.clone()
        return self._apply(x, xc)

    def maybe_replace_resid_post(self, layer_idx: int, x: torch.Tensor) -> torch.Tensor: return x
    def maybe_replace_head_z(self, layer_idx: int, z: torch.Tensor) -> torch.Tensor: return z
    def maybe_replace_mlp_out(self, layer_idx: int, h: torch.Tensor) -> torch.Tensor: return h


class HeadPatchPlan:
    """
    Replace a single attention head's z (pre-W_O) with the clean cache at layer L.
    Works only if your model populates clean_cache["z_per_head"][L] during enable_cache=True.

    Note: This plan is provided as a stub — you'll need your TransformerBlock to
    expose per-head z and have GPTModel call maybe_replace_head_z in the right place.
    """
    def __init__(self, clean_cache: Dict, L: int, head_idx: int):
        self.clean_cache, self.L, self.h = clean_cache, L, head_idx

    def maybe_replace_head_z(self, layer_idx: int, z: torch.Tensor) -> torch.Tensor:
        if (layer_idx == self.L) and (self.L in self.clean_cache.get("z_per_head", {})):
            z = z.clone()
            z[..., self.h, :] = self.clean_cache["z_per_head"][self.L][..., self.h, :].to(z.device)
            return z
        return z

    def maybe_replace_resid_pre(self, layer_idx: int, x: torch.Tensor) -> torch.Tensor: return x
    def maybe_replace_resid_post(self, layer_idx: int, x: torch.Tensor) -> torch.Tensor: return x
    def maybe_replace_mlp_out(self, layer_idx: int, h: torch.Tensor) -> torch.Tensor: return h


# ========= Core helpers =========

@torch.no_grad()
def cache_clean(model, toks_clean: torch.Tensor):
    """
    Run a clean pass and return (clean_logits, clean_cache).
    """
    model.eval()
    clean_logits, clean_cache = model(toks_clean, enable_cache=True)
    return clean_logits, clean_cache


@torch.no_grad()
def baseline_score_logitdiff(model, toks: torch.Tensor, target_id: int, distractor_id: int) -> float:
    logits = model(toks)
    if not torch.is_tensor(logits): logits = logits[0]
    return next_token_logit_diff(logits, target_id, distractor_id)


# ========= Layer scans (logit diff + KL) =========

@torch.no_grad()
def rank_layers_by_gain(
    model,
    toks_clean: torch.Tensor,
    toks_corrupt: torch.Tensor,
    target_id: int,
    distractor_id: int,
    n_layers: int,
    positions: Union[str, Iterable[int]] = "last",
    eps: Optional[float] = 0.1,
) -> List[Tuple[int, float]]:
    """
    Old logit-diff layer scan, but now supports masked/mixed patching via MaskedMixResidPlan.
    """
    clean_logits, clean_cache = cache_clean(model, toks_clean)
    base = baseline_score_logitdiff(model, toks_corrupt, target_id, distractor_id)

    out = []
    for L in range(n_layers):
        plan = MaskedMixResidPlan(clean_cache, L, positions=positions, eps=eps)
        logits = model(toks_corrupt, intervention_plan=plan)
        if not torch.is_tensor(logits): logits = logits[0]
        score = next_token_logit_diff(logits, target_id, distractor_id)
        out.append((L, score - base))
    out.sort(key=lambda x: x[1], reverse=True)
    return out


@torch.no_grad()
def rank_layers_by_gain_kl(
    model,
    toks_clean: torch.Tensor,
    toks_corrupt: torch.Tensor,
    n_layers: int,
    positions: Union[str, Iterable[int]] = "last",
    eps: Optional[float] = 0.1,
) -> List[Tuple[int, float]]:
    """
    KL-to-Clean layer scan with masked/mixed patching.
    Gain = KL(clean, corrupt) - KL(clean, patched). Larger is better (KL reduced).
    """
    clean_logits, clean_cache = cache_clean(model, toks_clean)
    corrupt_logits = model(toks_corrupt)
    if not torch.is_tensor(corrupt_logits): corrupt_logits = corrupt_logits[0]
    base = kl_to_clean(clean_logits, corrupt_logits).item()

    out = []
    for L in range(n_layers):
        plan = MaskedMixResidPlan(clean_cache, L, positions=positions, eps=eps)
        patched_logits = model(toks_corrupt, intervention_plan=plan)
        if not torch.is_tensor(patched_logits): patched_logits = patched_logits[0]
        kl_val = kl_to_clean(clean_logits, patched_logits).item()
        gain = base - kl_val
        out.append((L, gain))
    out.sort(key=lambda x: x[1], reverse=True)
    return out
