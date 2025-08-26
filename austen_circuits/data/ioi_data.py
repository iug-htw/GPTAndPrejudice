
from dataclasses import dataclass
from typing import List
import random

try:
    import tiktoken
except ImportError:
    tiktoken = None

AUSTEN_SINGLETON_NAMES = [
    "Mary","John","Emma","Charles","Jane","Elizabeth","George",
    "Anne","Henry","Catherine","James","Edward","Louisa","Harriet",
    "William","Tom","Susan","Robert","Frank"
]

TEMPLATE_CLEAN   = "When {A} and {B} went to the assembly, {B} gave a letter to {A}."
TEMPLATE_CORRUPT = "When {A} and {B} went to the assembly, {A} gave a letter to {B}."

@dataclass
class IOIBatch:
    token_ids_clean: List[List[int]]
    token_ids_corrupt: List[List[int]]
    io_token_ids: List[int]
    s_token_ids: List[int]
    target_positions_clean: List[int]
    target_positions_corrupt: List[int]

def _get_gpt2_tokenizer():
    if tiktoken is None:
        raise RuntimeError("tiktoken is required. Please `pip install tiktoken`.")
    return tiktoken.get_encoding("gpt2")

def _is_single_token(name: str, enc) -> bool:
    return len(enc.encode(name)) == 1

def _last_index(seq, tok_id: int) -> int:
    for i in range(len(seq)-1, -1, -1):
        if seq[i] == tok_id:
            return i
    return -1

def build_ioi_dataset(n: int = 256, seed: int = 0) -> IOIBatch:
    random.seed(seed)
    enc = _get_gpt2_tokenizer()
    names = [nm for nm in AUSTEN_SINGLETON_NAMES if _is_single_token(nm, enc)]
    if len(names) < 4:
        raise RuntimeError("Not enough single-token names for IOI templates.")

    token_ids_clean, token_ids_corrupt = [], []
    io_token_ids, s_token_ids = [], []
    target_positions_clean, target_positions_corrupt = [], []

    for _ in range(n):
        A, B = random.sample(names, 2)
        clean   = TEMPLATE_CLEAN.format(A=A, B=B)
        corrupt = TEMPLATE_CORRUPT.format(A=A, B=B)
        ids_c = enc.encode(clean)
        ids_k = enc.encode(corrupt)
        A_id = enc.encode(A)[0]; B_id = enc.encode(B)[0]

        tgt_c = _last_index(ids_c, A_id)  # IO is A in clean
        tgt_k = _last_index(ids_k, B_id)  # IO is B in corrupt
        if tgt_c < 0 or tgt_k < 0:
            continue

        token_ids_clean.append(ids_c)
        token_ids_corrupt.append(ids_k)
        io_token_ids.append(A_id)
        s_token_ids.append(B_id)
        target_positions_clean.append(tgt_c)
        target_positions_corrupt.append(tgt_k)

    return IOIBatch(
        token_ids_clean, token_ids_corrupt, io_token_ids, s_token_ids,
        target_positions_clean, target_positions_corrupt
    )
