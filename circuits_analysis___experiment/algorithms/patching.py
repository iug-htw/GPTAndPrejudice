
from typing import Dict, List, Tuple, Optional
import torch

# Optional base class – if your GPTModel defines its own InterventionPlan, you can ignore this.
class BaseInterventionPlan:
    def maybe_replace_resid_pre(self, layer_idx: int, x: torch.Tensor) -> torch.Tensor:
        return x
    def maybe_replace_resid_post(self, layer_idx: int, x: torch.Tensor) -> torch.Tensor:
        return x
    # Placeholders for future per-head patching
    def maybe_replace_head_z(self, layer_idx: int, head_idx: int, z: torch.Tensor) -> torch.Tensor:
        return z

class ResidualPatcherPlan(BaseInterventionPlan):
    """
    Stores clean run residual streams and, when the model calls back into this plan
    during a corrupt run, swaps the specified token positions with clean values.
    """
    def __init__(self, clean_cache: Dict[str, Dict[int, torch.Tensor]]):
        super().__init__()
        self.clean_cache = clean_cache
        self.instructions = {}  # layer -> (mode, mask[B,T], clean[B,T,d])

    def add_patch(self, layer_idx: int, mode: str, mask: torch.Tensor):
        assert mode in ("pre", "post")
        key = "resid_pre" if mode == "pre" else "resid_post"
        clean_tensor = self.clean_cache[key][layer_idx]  # [B,T,d]
        self.instructions[layer_idx] = (mode, mask.clone(), clean_tensor.clone())

    def _apply(self, layer_idx: int, x: torch.Tensor, mode: str) -> torch.Tensor:
        instr = self.instructions.get(layer_idx, None)
        if instr is None:
            return x
        m, mask, clean = instr
        if m != mode:
            return x
        mask_e = mask.bool().unsqueeze(-1)  # [B,T,1]
        return torch.where(mask_e, clean.to(x.device), x)

    def maybe_replace_resid_pre(self, layer_idx: int, x: torch.Tensor) -> torch.Tensor:
        return self._apply(layer_idx, x, "pre")

    def maybe_replace_resid_post(self, layer_idx: int, x: torch.Tensor) -> torch.Tensor:
        return self._apply(layer_idx, x, "post")


def make_mask_for_positions(batch_size: int, seq_len: int, positions: List[int]) -> torch.Tensor:
    m = torch.zeros((batch_size, seq_len), dtype=torch.bool)
    for p in positions:
        if 0 <= p < seq_len:
            m[:, p] = True
    return m
