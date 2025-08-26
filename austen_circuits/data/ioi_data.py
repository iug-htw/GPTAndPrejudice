# austen_circuits/data/ioi_data.py

from dataclasses import dataclass
from typing import List, Tuple
import random

try:
    import tiktoken
except ImportError:
    tiktoken = None

CANDIDATE_NAMES = [
    "Elizabeth","Jane","Emma","Anne","Catherine","Lydia","Mary","Elinor","Marianne",
    "Fanny","Harriet","Charlotte","Lucy","Caroline","Isabella","Georgiana","Sophia","Louisa",
    "Darcy","Bingley","George","Henry","Charles","William","Robert","John","James","Frank",
    "Edward","Frederick","Arthur","Thomas","Philip"
]

PLACES = [
    "the assembly rooms","Netherfield","Pemberley","Hartfield","Kellynch","Rosings",
    "Bath","Meryton","Highbury","Lambton","Lyme","the pump-room","the parsonage",
    "the rectory","the ballroom","the garden","the library"
]

OBJECTS = [
    "a letter","a note","a ribbon","a bonnet","a brooch","a parcel","a bouquet","a book",
    "a shawl","a fan","a locket","an invitation","the dancing card","a card","a glove","a purse"
]

VERBS = ["gave","handed","returned","entrusted","delivered","presented"]

OPENING_TEMPLATES = [
    "When {A} and {B} arrived at {PLACE}, ",
    "As {A} and {B} stood in {PLACE}, ",
    "While {A} and {B} were at {PLACE}, "
]

# CLEAN:   ... {B} {VERB} {OBJ} to {A}.
# CORRUPT: ... {A} {VERB} {OBJ} to {B}.
TEMPLATE_CLEAN   = "{OPEN}{B} {VERB} {OBJ} to {A}."
TEMPLATE_CORRUPT = "{OPEN}{A} {VERB} {OBJ} to {B}."

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

def _encode(s, enc): return enc.encode(s)

def _is_single_token_with_leading_space(name, enc):
    return len(enc.encode(" " + name)) == 1

def _last_index(seq, tok_id):
    for i in range(len(seq)-1, -1, -1):
        if seq[i] == tok_id:
            return i
    return -1

def _sample(A, B):
    open_ = random.choice(OPENING_TEMPLATES).format(A=A, B=B, PLACE=random.choice(PLACES))
    obj   = random.choice(OBJECTS)
    verb  = random.choice(VERBS)
    clean   = TEMPLATE_CLEAN.format(OPEN=open_,  B=B, VERB=verb, OBJ=obj, A=A)
    corrupt = TEMPLATE_CORRUPT.format(OPEN=open_, A=A, VERB=verb, OBJ=obj, B=B)
    return clean, corrupt

def build_ioi_dataset(n: int = 256, seed: int = 0) -> IOIBatch:
    random.seed(seed)
    enc = _get_gpt2_tokenizer()

    names = [nm for nm in CANDIDATE_NAMES if _is_single_token_with_leading_space(nm, enc)]
    if len(names) < 4:
        raise RuntimeError("Not enough single-token names (with leading space).")

    token_ids_clean, token_ids_corrupt = [], []
    io_token_ids, s_token_ids = [], []
    target_positions_clean, target_positions_corrupt = [], []

    attempts = 0
    while len(token_ids_clean) < n and attempts < 10*n:
        attempts += 1
        A, B = random.sample(names, 2)
        s_clean, s_corrupt = _sample(A, B)

        # Hard guard: force ending “… to {NAME}.”
        if not s_clean.endswith(f" to {A}.") or not s_corrupt.endswith(f" to {B}."):
            continue

        ids_c = _encode(s_clean, enc)
        ids_k = _encode(s_corrupt, enc)
        A_id = enc.encode(" " + A)[0]
        B_id = enc.encode(" " + B)[0]

        t_c = _last_index(ids_c, A_id)
        t_k = _last_index(ids_k, B_id)
        if t_c == -1 or t_k == -1:
            continue

        # Sanity: token right before IO should decode to "to"
        if t_c <= 0 or t_k <= 0: continue
        if enc.decode([ids_c[t_c-1]]).strip() != "to": continue
        if enc.decode([ids_k[t_k-1]]).strip() != "to": continue

        token_ids_clean.append(ids_c); token_ids_corrupt.append(ids_k)
        io_token_ids.append(A_id);     s_token_ids.append(B_id)
        target_positions_clean.append(t_c); target_positions_corrupt.append(t_k)

    if not token_ids_clean:
        raise RuntimeError("IOI dataset is empty. Sentences must end with ' to {NAME}.'")

    if len(token_ids_clean) < n:
        print(f"[WARN] Only {len(token_ids_clean)}/{n} examples after {attempts} attempts.")

    return IOIBatch(
        token_ids_clean[:n], token_ids_corrupt[:n],
        io_token_ids[:n],    s_token_ids[:n],
        target_positions_clean[:n], target_positions_corrupt[:n],
    )
