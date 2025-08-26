
import torch

def pad_and_stack(batch_token_ids, pad_id: int = 50256):
    T = max(len(x) for x in batch_token_ids)
    B = len(batch_token_ids)
    out = torch.full((B, T), pad_id, dtype=torch.long)
    for i, ids in enumerate(batch_token_ids):
        out[i, T-len(ids):] = torch.tensor(ids, dtype=torch.long)
    return out
