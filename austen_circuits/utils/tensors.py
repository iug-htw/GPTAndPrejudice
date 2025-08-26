# austen_circuits_v2/utils/tensors.py

import torch

def pad_and_stack(batch_token_ids, pad_id: int = 50256):
    T = max(len(x) for x in batch_token_ids)
    B = len(batch_token_ids)
    out = torch.full((B, T), pad_id, dtype=torch.long)
    for i, ids in enumerate(batch_token_ids):
        out[i, T-len(ids):] = torch.tensor(ids, dtype=torch.long)
    return out

def pad_and_stack_with_adjusted_positions(batch_token_ids, orig_positions, pad_id: int = 50256):
    """
    Left-pad to fixed length and shift each orig position by its left-pad offset.
    Returns (padded_ids [B,T], padded_positions [B]).
    """
    T = max(len(x) for x in batch_token_ids)
    B = len(batch_token_ids)
    out = torch.full((B, T), pad_id, dtype=torch.long)
    new_pos = []
    for i, ids in enumerate(batch_token_ids):
        offset = T - len(ids)          # number of pads on the left
        out[i, offset:] = torch.tensor(ids, dtype=torch.long)
        new_pos.append(offset + orig_positions[i])
    return out, new_pos
