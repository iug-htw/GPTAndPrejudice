
"""circuit_tasks.py
Generators for clean/corrupt minimal-pair prompts to probe circuits.
They return tuples of (clean_text, corrupt_text, target_str, distractor_str).
Use your own tokenizer to convert strings to IDs (see to_token_ids).
"""
from typing import List, Tuple, Callable
import random

# Deterministic sampling by default; override from the caller if needed.
_rng = random.Random(0)

FEMALE_NAMES = ["Elizabeth", "Jane", "Emma", "Catherine", "Marianne", "Elinor", "Anne", "Harriet"]
MALE_NAMES   = ["Darcy", "Bingley", "Knightley", "Wentworth", "Brandon", "Edmund", "William", "Henry"]
PLACES       = ["Netherfield", "Highbury", "Bath", "Pemberley", "Hertfordshire"]
MARRIAGE_WORDS = ["marriage", "wedlock", "engagement", "proposal"]
WEALTH_WORDS   = ["fortune", "inheritance", "income", "estate"]
DUTY_WORDS     = ["duty", "obligation", "conduct", "propriety"]
EMOTION_WORDS  = ["love", "affection", "admiration", "esteem"]
SOC_WORDS      = ["society", "reputation", "standing", "station"]

def seed(s: int) -> None:
    """Optionally re-seed the internal RNG used by the generators."""
    global _rng
    _rng = random.Random(s)

def build_gender_swap_pairs(n: int = 64) -> List[Tuple[str, str, str, str]]:
    """Return (clean, corrupt, target_str, distractor_str).
    Clean uses a female subject (biasing next token toward 'she'); corrupt swaps to male ('he').
    """
    out = []
    for _ in range(n):
        f = _rng.choice(FEMALE_NAMES)
        m = _rng.choice(MALE_NAMES)
        place = _rng.choice(PLACES)
        clean   = f"{f} arrived at {place} after a long journey. Without hesitation, "
        corrupt = f"{m} arrived at {place} after a long journey. Without hesitation, "
        out.append((clean, corrupt, "she", "he"))
    return out

def build_ioi_pairs(n: int = 64) -> List[Tuple[str, str, str, str]]:
    """IOI-style: two names; minimal pair flips which name the pronoun refers to.
    Target/distractor are the correct/incorrect names as next-token candidates.
    """
    out = []
    for _ in range(n):
        A = _rng.choice(FEMALE_NAMES + MALE_NAMES)
        B = _rng.choice(FEMALE_NAMES + MALE_NAMES)
        while B == A:
            B = _rng.choice(FEMALE_NAMES + MALE_NAMES)
        clean   = f"{A} wrote to {B} because she wished to speak with "
        corrupt = f"{B} wrote to {A} because she wished to speak with "
        out.append((clean, corrupt, A, B))
    return out

def build_marriage_vs_wealth_pairs(n: int = 64) -> List[Tuple[str, str, str, str]]:
    """Pairs that disambiguate next-token between marriage vs wealth vocabulary."""
    out = []
    for _ in range(n):
        w_mar = _rng.choice(MARRIAGE_WORDS)
        w_wlt = _rng.choice(WEALTH_WORDS)
        name = _rng.choice(FEMALE_NAMES + MALE_NAMES)
        clean   = f"For {name}, happiness depended chiefly on \"{w_mar}\" rather than \""
        corrupt = f"For {name}, happiness depended chiefly on \"{w_wlt}\" rather than \""
        # After 'rather than', we expect the contrasting concept
        out.append((clean, corrupt, w_wlt, w_mar))
    return out

def build_emotion_vs_duty_pairs(n: int = 64) -> List[Tuple[str, str, str, str]]:
    out = []
    for _ in range(n):
        e = _rng.choice(EMOTION_WORDS)
        d = _rng.choice(DUTY_WORDS)
        name = _rng.choice(FEMALE_NAMES + MALE_NAMES)
        clean   = f"{name} prized {e} above "
        corrupt = f"{name} prized {d} above "
        # Likely continuation from the *other* set
        out.append((clean, corrupt, d, e))
    return out

def to_token_ids(
    pairs: List[Tuple[str, str, str, str]],
    tokenize: Callable[[List[str]], List[List[int]]],
    vocab_lookup: Callable[[str], int],
) -> List[Tuple[List[int], List[int], int, int]]:
    """Map string pairs to token IDs using provided tokenizer & vocab lookup.
    Returns: (toks_clean, toks_corrupt, target_id, distractor_id)
    """
    texts = []
    for clean, corrupt, _, _ in pairs:
        texts.extend([clean, corrupt])
    enc = tokenize(texts)
    assert len(enc) == 2 * len(pairs), "Tokenizer returned unexpected count"
    out = []
    for i, (clean, corrupt, tgt, dis) in enumerate(pairs):
        t_clean   = enc[2*i]
        t_corrupt = enc[2*i + 1]
        out.append((t_clean, t_corrupt, vocab_lookup(tgt), vocab_lookup(dis)))
    return out
