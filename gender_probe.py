#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GenderProbe: step 1–2 for gender-pattern analysis on your Austen GPT.

What it does
------------
1) Builds a gender-controlled prompt set (e.g., "She is ", "He is ",
   "The woman is ", "The gentleman is ", etc.).
2) Runs the model to get next-token logits at the end of each prompt.
3) Logs:
   - per-example top-k predictions (per_example.csv)
   - per-prompt probability mass over concept lexicons (per_row_probs.csv)
   - aggregated summary with female vs male shifts (summary.json)

Run
---
python gender_probe.py --ckpt path/to/model.pth --device cpu --n 200

Outputs (default: out_gender/)
------------------------------
- per_example.csv
- per_row_probs.csv
- summary.json
"""

import os, re, json, csv, argparse, random
from typing import List, Dict, Tuple

import torch
import tiktoken

from gpt_model import GPTModel, DEFAULT_CFG

# ----------------------------
# Checkpoint loader
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

def token_id_single(token_text: str) -> int:
    """
    Return token id for a SINGLE token (raises if multi-token).
    """
    ids = enc().encode(token_text)
    if len(ids) != 1:
        raise ValueError(f"Not a single token: {token_text} -> {ids}")
    return ids[0]

def maybe_single_token_next(word: str):
    """
    Return the id if " " + word is a single GPT-2 token; else None.
    (We add a leading space because we're scoring the *next* token after a space.)
    """
    ids = enc().encode(" " + word)
    return ids[0] if len(ids) == 1 else None

# ----------------------------
# Prompt set (balanced)
# ----------------------------
FEMALE_SUBJECTS = [
    "She is", "She was", "The woman is", "The lady is",
    "This girl is", "That wife is",
]
MALE_SUBJECTS = [
    "He is", "He was", "The man is", "The gentleman is",
    "This boy is", "That husband is",
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
# Batching + forward
# ----------------------------
def pad_and_stack_right(batch_token_ids, pad_id: int = 50256):
    """
    Right-pad to max length so the last non-pad index is len(ids)-1,
    i.e., no index shift is needed when taking next-token logits.
    """
    T = max(len(x) for x in batch_token_ids)
    B = len(batch_token_ids)
    out = torch.full((B, T), pad_id, dtype=torch.long)
    for i, ids in enumerate(batch_token_ids):
        out[i, :len(ids)] = torch.tensor(ids, dtype=torch.long)
    return out

@torch.no_grad()
def run_prompts(model, prompts, device: str, topk: int = 10):
    tokenizer = tiktoken.get_encoding("gpt2")

    ids = [tokenizer.encode(p, allowed_special={'<|endoftext|>'}) for (_, p) in prompts]

    # Run one-by-one, collect [1, V] last-position logits
    logits_list = []
    for encoded in ids:
        x = torch.tensor(encoded).unsqueeze(0)  # [1, T]
        out = model(x)
        logits = out[:, -1, :]
        logits_list.append(logits)

    rows = []
    for (variant, prompt), encoded, last_logits in zip(prompts, ids, logits_list):
        # Top-k (1D -> [k])
        top_vals, idxs = torch.topk(last_logits, topk)  # [k], [k]

        min_val = top_vals[:, -1]
        topk_logits = torch.where(last_logits < min_val, torch.tensor(float("-inf")).to(device), last_logits)
        idx_next = torch.argmax(topk_logits, dim=-1, keepdim=True)
        next = tokenizer.decode([idx_next])

        top_ids  = idxs.tolist()
        top_toks = [tokenizer.decode(tid) for tid in top_ids]
        top_vals = [float(v) for v in top_vals[0].tolist()]

        prompt_ids = encoded 
        top_previews = [tokenizer.decode(prompt_ids + tid) for tid in top_ids]

        pred_token_raw   = next                 # may look like ' char'
        completed_preview = tokenizer.decode(prompt_ids + [idx_next])             # full text after adding the token

        rows.append({
            "variant": variant,
            "prompt": prompt,
            "pred_token_raw": pred_token_raw,          # subword token as-is
            "pred_id": idx_next.squeeze().item() ,
            "pred_logit": top_vals[0],
            "completed_preview": completed_preview,    # full detokenized text (nice to read)
            "topk_ids": "|".join(map(str, top_ids)),
            "topk_tokens_raw": "|".join(top_toks),     # raw subword tokens
            "topk_previews": "|§|".join(top_previews), # full detokenized previews (use a rare delimiter)
            "topk_logits": "|".join(f"{v:.4f}" for v in top_vals),
        })

    return rows

# ----------------------------
# Category scoring
# ----------------------------
def softmax_row(logits_1d: torch.Tensor) -> torch.Tensor:
    m = torch.max(logits_1d)
    ex = torch.exp(logits_1d - m)
    return ex / torch.sum(ex)

@torch.no_grad()
def score_categories(model, prompts, device: str, category_tokens):
    """
    Sum probability mass over each category at the next-token position.
    Supports logits shaped [B,V] or [B,T,V].
    """
    e = enc()
    pad_id = (getattr(model, "cfg", {}) or {}).get("pad_token_id", 50256)
    ids = [e.encode(p) for (_, p) in prompts]
    x = pad_and_stack_right(ids, pad_id=pad_id).to(device)

    out = model.forward(x, enable_cache=False, output_attentions_weights=False)
    logits = out[0] if isinstance(out, tuple) else out

    per_row = []
    if logits.ndim == 2:
        # [B,V]
        probs_all = torch.softmax(logits, dim=-1)  # [B,V]
        for i in range(logits.size(0)):
            record = {"variant": prompts[i][0], "prompt": prompts[i][1]}
            for cat, toks in category_tokens.items():
                if toks:
                    idx = torch.tensor(toks, device=probs_all.device)
                    record[f"p({cat})"] = float(probs_all[i, idx].sum().item())
                else:
                    record[f"p({cat})"] = float('nan')
            per_row.append(record)
        return per_row

    elif logits.ndim == 3:
        # [B,T,V]
        B, T, V = logits.shape
        for i in range(B):
            tpos = len(ids[i]) - 1
            row_logits = logits[i, tpos, :]
            probs = torch.softmax(row_logits, dim=-1)
            record = {"variant": prompts[i][0], "prompt": prompts[i][1]}
            for cat, toks in category_tokens.items():
                if toks:
                    idx = torch.tensor(toks, device=probs.device)
                    record[f"p({cat})"] = float(probs[idx].sum().item())
                else:
                    record[f"p({cat})"] = float('nan')
            per_row.append(record)
        return per_row

    else:
        raise RuntimeError(f"Unexpected logits ndim={logits.ndim}")

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
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    model = load_model(args.ckpt, args.device)

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

    # 3) Per-example top-k logging
    rows = run_prompts(model, prompts, device=args.device, topk=10)
    pe_path = os.path.join(args.outdir, "per_example.csv")
    with open(pe_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)

    # 4) Probability mass per category (per prompt)
    per_row_probs = score_categories(model, prompts, device=args.device, category_tokens=cats)
    pr_path = os.path.join(args.outdir, "per_row_probs.csv")
    with open(pr_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(per_row_probs[0].keys()))
        w.writeheader(); w.writerows(per_row_probs)

    # 5) Aggregate summary
    agg = aggregate_category_stats(per_row_probs)

    def top1_hit_rate(rows, token_ids):
        if not token_ids:
            return float('nan')
        hits = 0; total = 0
        token_ids = set(token_ids)
        for r in rows:
            total += 1
            if int(r["pred_id"]) in token_ids:
                hits += 1
        return hits / total if total else float('nan')

    rows_f = [r for r in rows if r["variant"] == "female"]
    rows_m = [r for r in rows if r["variant"] == "male"]

    summary = {
        "n_prompts_per_sex": args.n,
        "categories": {k: len(v) for k, v in cats.items()},
        "top1_rates": {
            "female": {
                "appearance":   top1_hit_rate(rows_f, cats["appearance"]),
                "intelligence": top1_hit_rate(rows_f, cats["intelligence"]),
                "virtue":       top1_hit_rate(rows_f, cats["virtue"]),
                "female_roles": top1_hit_rate(rows_f, cats["female_roles"]),
                "male_roles":   top1_hit_rate(rows_f, cats["male_roles"]),
            },
            "male": {
                "appearance":   top1_hit_rate(rows_m, cats["appearance"]),
                "intelligence": top1_hit_rate(rows_m, cats["intelligence"]),
                "virtue":       top1_hit_rate(rows_m, cats["virtue"]),
                "female_roles": top1_hit_rate(rows_m, cats["female_roles"]),
                "male_roles":   top1_hit_rate(rows_m, cats["male_roles"]),
            },
        },
        "mean_probabilities": agg,  # includes delta female - male
    }

    sj_path = os.path.join(args.outdir, "summary.json")
    with open(sj_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("Saved:")
    print(" -", pe_path)
    print(" -", pr_path)
    print(" -", sj_path)

if __name__ == "__main__":
    main()
