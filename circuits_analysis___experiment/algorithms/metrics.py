
import torch

def io_logit_diff(logits: torch.Tensor, target_pos, io_ids, s_ids) -> torch.Tensor:
    B, T, V = logits.shape
    device = logits.device
    idx = torch.arange(B, device=device)
    tp  = torch.tensor(target_pos, device=device)
    io  = torch.tensor(io_ids, device=device)
    sj  = torch.tensor(s_ids, device=device)
    lt = logits[idx, tp, :]  # [B,V]
    return lt[idx, io] - lt[idx, sj]

def mean_io_logit_diff(logits, target_pos, io_ids, s_ids) -> float:
    return io_logit_diff(logits, target_pos, io_ids, s_ids).mean().item()
