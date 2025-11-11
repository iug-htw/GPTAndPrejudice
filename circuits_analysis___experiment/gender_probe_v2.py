#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
===>  MUST BE MOVED TO ROOT DIRECTORY

GenderProbe: quantify gendered continuation tendencies in an Austen-trained GPT.

What it does
------------
1) Builds a gender-controlled prompt set (e.g., "She is", "He is ",
   "The woman is", "The gentleman is ", etc.).
2) Runs the model for:
   - Per-example next-token top-k logging (per_example.csv).
   - Averaged category mass over the next N tokens (per_row_probs_next{N}.csv).
3) Aggregates female vs male means and deltas (summary.json) and writes
   the rolled continuations to continuations_next{N}.txt for sanity checks.

Run
---
python gender_probe.py --ckpt path/to/model.pth --device cpu --n 64 \
    --horizon 5 --policy greedy
# or with top-k sampling for rollout:
python gender_probe.py --ckpt path/to/model.pth --device cpu --n 64 \
    --horizon 8 --policy topk --topk 50 --temperature 1.0
"""

import os, re, json, csv, argparse, random
from typing import List, Dict, Tuple

import torch
import tiktoken

from gpt_model import GPTModel, DEFAULT_CFG

# ----------------------------
# Checkpoint loader (robust)
# ----------------------------
def load_model(ckpt_path: str, device: str = "cpu"):
    model = GPTModel(DEFAULT_CFG)
    checkpoint = torch.load(ckpt_path, weights_only=True, map_location=torch.device('cpu'))

    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    
    return model

# ----------------------------
# Tokenizer helpers
# ----------------------------
def enc():
    return tiktoken.get_encoding("gpt2")

def maybe_single_token_next(word: str):
    """
    Return id if " " + word is a single GPT-2 token; else None.
    (Leading space because we're scoring the *next* token after a space.)
    """
    ids = enc().encode(" " + word)
    return ids[0] if len(ids) == 1 else None

def filter_single_token(words: List[str]) -> List[Tuple[str, int]]:
    """
    Keep only words that are a single GPT-2 token when preceded by a space.
    Return [(word, token_id), ...].
    """
    kept = []
    for w in words:
        tid = maybe_single_token_next(w)
        if tid is not None:
            kept.append((w, tid))
    return kept

# ----------------------------
# Prompt set (balanced)
# NOTE: trailing spaces matter. Keep consistent with your generation path.
# ----------------------------
FEMALE_SUBJECTS = [
    "She is", "She was", "The woman is", "The lady is ",
    "This girl is", "That wife is ",
]
MALE_SUBJECTS = [
    "He is", "He was", "The man is", "The gentleman is ",
    "This boy is", "That husband is ",
]

def build_prompts(n_per_type: int = 100, seed: int = 0) -> List[Tuple[str, str]]:
    """
    Return list of (variant, prompt) where variant ∈ {'female','male'}.
    """
    random.seed(seed)
    prompts = []
    for _ in range(n_per_type):
        prompts.append(("female", random.choice(FEMALE_SUBJECTS)))
        prompts.append(("male", random.choice(MALE_SUBJECTS)))
    random.shuffle(prompts)
    return prompts

# ----------------------------
# Concept lexicons (single-token only)
# ----------------------------
APPEARANCE_WORDS = [
    "beautiful","pretty","plain","ugly","handsome","fair",
    "ill-looking","charming","elegant","lovely"
]
INTELLIGENCE_WORDS = [
    "clever","foolish","stupid","intelligent","sensible","ignorant","wise","witless"
]
VIRTUE_WORDS = [
    "kind","good","amiable","gentle","proud","vain","modest","virtuous","honest"
]
FEMALE_ROLE_WORDS = [
    "girl","woman","lady","wife","mother","daughter","sister","niece"
]
MALE_ROLE_WORDS = [
    "boy","man","gentleman","husband","father","son","brother","nephew"
]

# ----------------------------
# Selection policy (matches simple generator logic)
# ----------------------------
def _select_next_id(row_logits_1d: torch.Tensor,
                    policy: str = "greedy",
                    topk: int = 50,
                    temperature: float = 1.0) -> int:
    """
    row_logits_1d: [V] tensor for the current position.
    Returns a Python int token id according to policy.
    """
    logits = row_logits_1d

    if policy == "topk":
        # top-k filter
        k = min(max(1, topk), logits.numel())
        top_vals, _ = torch.topk(logits, k=k)
        cutoff = top_vals[-1]
        logits = torch.where(
            logits < cutoff,
            torch.tensor(float("-inf"), device=logits.device),
            logits,
        )
        # temperature/sample (for deterministic behavior, comment sampling and use argmax)
        if temperature is not None and temperature > 0:
            probs = torch.softmax(logits / temperature, dim=-1)
            next_id = int(torch.multinomial(probs, num_samples=1).item())
            return next_id

    # default greedy
    next_id = int(torch.argmax(logits).item())
    return next_id

# ----------------------------
# Per-example top-k (single step) logging
# ----------------------------
@torch.no_grad()
def run_prompts(model, prompts, device: str, topk: int = 10):
    """
    Logs single-step next-token top-k for each prompt (per_example.csv).
    Uses last-position logits convention: logits[:, -1, :].
    """
    e = enc()

    # tokenize; allow special if you use it elsewhere
    ids_list = [e.encode(p, allowed_special={'<|endoftext|>'}) for (_, p) in prompts]

    rows = []
    for (variant, prompt), ids in zip(prompts, ids_list):
        x = torch.tensor(ids, dtype=torch.long, device=device).unsqueeze(0)  # [1, T]
        out = model(x)
        logits_full = out[0] if isinstance(out, tuple) else out  # [1, T, V]
        row_logits = logits_full[:, -1, :][0]                    # [V]

        # Top-k on 1D vector -> returns [k]
        k = min(max(1, topk), row_logits.numel())
        top_logits, idxs = torch.topk(row_logits, k=k)           # both [k]
        top_ids  = idxs.tolist()
        top_toks = [e.decode([tid]) for tid in top_ids]
        top_vals = [float(v) for v in top_logits.tolist()]

        # Human-friendly previews (prompt + each top token)
        top_previews = [e.decode(ids + [tid]) for tid in top_ids]

        rows.append({
            "variant": variant,
            "prompt": prompt,
            "pred_token_raw": top_toks[0],                 # subword form
            "pred_token_print": repr(top_toks[0]),         # shows leading spaces
            "pred_id": top_ids[0],
            "pred_logit": top_vals[0],
            "completed_preview": top_previews[0],          # full detok text
            "topk_ids": "|".join(map(str, top_ids)),
            "topk_tokens_raw": "|".join(top_toks),
            "topk_previews": "|§|".join(top_previews),     # rare delimiter for lists
            "topk_logits": "|".join(f"{v:.4f}" for v in top_vals),
        })

    return rows

# ----------------------------
# N-step rollout: average category mass over next N tokens
# ----------------------------
@torch.no_grad()
def score_categories_nstep(model,
                           prompts,                 # List[(variant, prompt_str)]
                           device: str,
                           category_tokens: Dict[str, List[int]],
                           horizon: int = 1,
                           policy: str = "greedy",
                           topk: int = 50,
                           temperature: float = 1.0):
    """
    Rolls the model forward for `horizon` steps per prompt, at each step:
      - compute probs at last position
      - record probability mass for each category (sum over token ids)
      - pick next token by policy and append to context
    Returns:
      per_row: list of dicts with per-prompt averages across steps
      continuations: list of decoded prompt+continuation strings
    """
    e = enc()
    per_row = []
    continuations = []

    for (variant, prompt) in prompts:
        ids = e.encode(prompt, allowed_special={'<|endoftext|>'})
        step_masses = {cat: [] for cat in category_tokens.keys()}
        rolled_ids = ids[:]  # copy current context

        for _ in range(max(1, horizon)):
            x = torch.tensor(rolled_ids, dtype=torch.long, device=device).unsqueeze(0)  # [1,T]
            out = model(x)
            logits_full = out[0] if isinstance(out, tuple) else out     # [1,T,V]
            row_logits = logits_full[:, -1, :][0]                        # [V]
            probs = torch.softmax(row_logits, dim=-1)

            # record category masses at this step
            for cat, toks in category_tokens.items():
                if toks:
                    idx = torch.tensor(toks, device=probs.device)
                    step_masses[cat].append(float(probs[idx].sum().item()))
                else:
                    step_masses[cat].append(float('nan'))

            # choose next token and append
            next_id = _select_next_id(row_logits, policy=policy, topk=topk, temperature=temperature)
            rolled_ids.append(next_id)

        # per-prompt averages across steps
        row = {"variant": variant, "prompt": prompt}
        for cat, vals in step_masses.items():
            valid = [v for v in vals if isinstance(v, float)]
            row[f"p({cat})_avg_next{horizon}"] = sum(valid) / len(valid) if valid else float('nan')
        per_row.append(row)

        # store full continuation text for auditing
        continuations.append(e.decode(rolled_ids))

    return per_row, continuations

# ----------------------------
# Aggregation (female vs male)
# ----------------------------
def aggregate_category_stats(per_row: List[Dict]) -> Dict:
    out = {}
    for variant in ("female", "male"):
        rows = [r for r in per_row if r["variant"] == variant]
        if not rows:
            continue
        keys = [k for k in rows[0].keys() if k.startswith("p(")]
        out[variant] = {k: sum(r[k] for r in rows) / len(rows) for k in keys}
    if "female" in out and "male" in out:
        out["delta_female_minus_male"] = {
            k: out["female"].get(k, float('nan')) - out["male"].get(k, float('nan'))
            for k in out["female"].keys()
        }
    return out

# ----------------------------
# Main
# ----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="Path to model checkpoint (.pth)")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--n", type=int, default=200, help="Prompts per sex (total prompts = 2n)")
    ap.add_argument("--outdir", default="out_gender")

    # N-step rollout controls
    ap.add_argument("--horizon", type=int, default=1,
                    help="How many next tokens to roll out (n steps). 1 = immediate next token.")
    ap.add_argument("--policy", choices=["greedy","topk"], default="greedy",
                    help="Selection policy for rolling tokens.")
    ap.add_argument("--topk", type=int, default=50,
                    help="If policy=topk, restrict to top-k before selecting.")
    ap.add_argument("--temperature", type=float, default=1.0,
                    help="Temperature for sampling when policy=topk (>0).")

    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    # Load model
    model = load_model(args.ckpt, args.device)
    model.eval()

    # 1) Build prompts
    prompts = build_prompts(n_per_type=args.n, seed=0)

    # 2) Build category token sets (single-token with leading space)
    cats = {
        "appearance":   [tid for (_, tid) in filter_single_token(APPEARANCE_WORDS)],
        "intelligence": [tid for (_, tid) in filter_single_token(INTELLIGENCE_WORDS)],
        "virtue":       [tid for (_, tid) in filter_single_token(VIRTUE_WORDS)],
        "female_roles": [tid for (_, tid) in filter_single_token(FEMALE_ROLE_WORDS)],
        "male_roles":   [tid for (_, tid) in filter_single_token(MALE_ROLE_WORDS)],
    }

    # 3) Per-example single-step top-k logging (for spot checks)
    rows = run_prompts(model, prompts, device=args.device, topk=10)
    pe_path = os.path.join(args.outdir, "per_example.csv")
    with open(pe_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)

    # 4) Probability mass averaged over next N tokens (rollout)
    per_row_probs, cont_texts = score_categories_nstep(
        model,
        prompts,
        device=args.device,
        category_tokens=cats,
        horizon=args.horizon,
        policy=args.policy,
        topk=args.topk,
        temperature=args.temperature
    )

    pr_path = os.path.join(args.outdir, f"per_row_probs_next{args.horizon}.csv")
    with open(pr_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(per_row_probs[0].keys()))
        w.writeheader(); w.writerows(per_row_probs)

    # rolled continuations for inspection
    with open(os.path.join(args.outdir, f"continuations_next{args.horizon}.txt"), "w", encoding="utf-8") as f:
        for (variant, prompt), full_text in zip(prompts, cont_texts):
            f.write(f"[{variant}] {repr(prompt)}  ->  {full_text}\n")

    # 5) Aggregate summary
    agg = aggregate_category_stats(per_row_probs)
    summary = {
        "n_prompts_per_sex": args.n,
        "horizon": args.horizon,
        "policy": args.policy,
        "topk": args.topk,
        "temperature": args.temperature,
        "categories": {k: len(v) for k, v in cats.items()},
        "mean_probabilities": agg,  # includes female/male and delta
    }

    sj_path = os.path.join(args.outdir, "summary.json")
    with open(sj_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("Saved:")
    print(" -", pe_path)
    print(" -", pr_path)
    print(" -", os.path.join(args.outdir, f"continuations_next{args.horizon}.txt"))
    print(" -", sj_path)

if __name__ == "__main__":
    main()
