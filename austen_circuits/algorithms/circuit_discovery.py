
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
import torch
from .metrics import io_logit_diff, mean_io_logit_diff
from .patching import ResidualPatcherPlan, make_mask_for_positions

@dataclass
class PatchResult:
    layer: int
    mode: str
    restored_mean: float

def run_model(model, ids, collect_attn: bool = True):
    """
    Calls model.forward with different signatures gracefully.
    Expects that when collect_attn=True, it returns attention weights per layer if available.
    Returns: logits, cache (dict or None), attn (list[tensor] or None)
    """
    out = model.forward(ids, enable_cache=True, output_attentions_weights=collect_attn)
    if isinstance(out, tuple):
        # Try to unpack common patterns
        if len(out) == 4:
            logits, cache, hidden, attn = out
        elif len(out) == 3:
            logits, cache, hidden = out; attn = None
        elif len(out) == 2:
            logits, cache = out; hidden = None; attn = None
        else:
            logits = out[0]; cache = None; attn = None
    else:
        logits, cache, hidden, attn = out, None, None, None
    return logits, cache, attn

def layerwise_resid_patching(model,
                             clean_cache: Dict,
                             ids_corrupt: torch.Tensor,
                             target_positions: List[int],
                             io_ids: List[int],
                             s_ids: List[int],
                             mode: str = "pre") -> List[PatchResult]:
    """
    For each layer L, replace resid_{mode}[L] at target positions in corrupt run with clean values and
    measure restoration in IO logit diff.
    Requires the model to *call* plan.maybe_replace_resid_pre/post(layer_idx, tensor).
    If not supported, this function will raise or no-op depending on model implementation.
    """
    device = next(model.parameters()).device
    B, T = ids_corrupt.shape
    # Baseline corrupt metric
    base_logits, _, _ = run_model(model, ids_corrupt, collect_attn=False)
    base = mean_io_logit_diff(base_logits, target_positions, io_ids, s_ids)

    results = []
    n_layers = getattr(model, "n_layers", None) or len(getattr(model, "trf_blocks", [])) or 12
    for L in range(n_layers):
        plan = ResidualPatcherPlan(clean_cache)
        mask = make_mask_for_positions(B, T, target_positions)
        plan.add_patch(L, mode, mask)
        try:
            out = model.forward(ids_corrupt, intervention_plan=plan)
        except TypeError:
            # Model does not accept intervention_plan → skip with zero restoration
            results.append(PatchResult(layer=L, mode=mode, restored_mean=0.0))
            continue
        logits_patched = out[0] if isinstance(out, tuple) else out
        score = mean_io_logit_diff(logits_patched, target_positions, io_ids, s_ids)
        results.append(PatchResult(layer=L, mode=mode, restored_mean=score - base))
    return results

def heuristic_name_mover_heads(attn_weights_per_layer: List[torch.Tensor],
                               target_positions: List[int],
                               io_ids: List[int],
                               ids_clean: torch.Tensor) -> List[Tuple[int,int,float]]:
    if not attn_weights_per_layer:
        return []
    B, T = ids_clean.shape
    # Locate IO token positions (last occurrence)
    io_positions = []
    ids_cpu = ids_clean.cpu()
    for b in range(B):
        tok = io_ids[b]
        pos = -1
        for t in range(T-1, -1, -1):
            if ids_cpu[b, t].item() == tok:
                pos = t; break
        io_positions.append(pos)

    scored = []
    for L, attn in enumerate(attn_weights_per_layer):
        # Expect [B, H, T_q, T_k]
        if attn.dim() != 4:
            continue
        H = attn.size(1)
        for h in range(H):
            a = attn[:, h, :, :]  # [B,T_q,T_k]
            total = 0.0; count = 0
            for b in range(B):
                tp = target_positions[b]; ip = io_positions[b]
                if tp < 0 or ip < 0 or tp >= a.size(1) or ip >= a.size(2):
                    continue
                total += float(a[b, tp, ip].item())
                count += 1
            if count > 0:
                scored.append((L, h, total / count))
    scored.sort(key=lambda x: x[2], reverse=True)
    return scored
