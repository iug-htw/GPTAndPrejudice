#!/usr/bin/env python3
"""
circuits_cli.py
Run clean vs corrupt experiments with different metrics (logit_diff, kl),
and with *masked/mixed* patching so results aren't trivially flat.

New flags:
  --metric {kl, logit_diff}      # default 'kl'
  --final_pos_only               # patch only the final token (position-masked)
  --eps FLOAT                    # epsilon mixing (0.0=hard replace; 0.1 default)
"""
import argparse, json
from pathlib import Path
from typing import List
import tiktoken
import torch

from gpt_model import GPTModel, DEFAULT_CFG
from circuits import (
    rank_layers_by_gain,
    rank_layers_by_gain_kl,
)

from austen_circuits.circuit_tasks import (
    build_gender_swap_pairs,
    build_ioi_pairs,
    build_marriage_vs_wealth_pairs,
    build_emotion_vs_duty_pairs,
    to_token_ids,
)
from viz import layer_importance_bar

enc = tiktoken.get_encoding("gpt2")
def tokenize(texts): return [enc.encode(t) for t in texts]
def vocab_lookup(tok_str): return enc.encode(tok_str)[0]

def load_model(ckpt_path: str, device: str = "cpu"):
    model = GPTModel(DEFAULT_CFG)
    checkpoint = torch.load(ckpt_path, weights_only=True, map_location=torch.device('cpu'))

    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    
    return model

# ---------- tasks ----------
TASKS = {
    "gender_swap": build_gender_swap_pairs,
    "ioi": build_ioi_pairs,
    "marriage_vs_wealth": build_marriage_vs_wealth_pairs,
    "emotion_vs_duty": build_emotion_vs_duty_pairs,
}


def main():
    ap = argparse.ArgumentParser(description="Causal circuit scan with KL/logit metrics and masked/mixed patching.")
    ap.add_argument("--ckpt", required=True, help="Path to model weights (.pt/.pth)")
    ap.add_argument("--task", required=True, choices=TASKS.keys())
    ap.add_argument("--pairs", type=int, default=128)
    ap.add_argument("--metric", type=str, default="kl", choices=["kl", "logit_diff"])
    ap.add_argument("--final_pos_only", action="store_true", help="Patch only the final position.")
    ap.add_argument("--eps", type=float, default=0.1, help="Epsilon mixing strength (0.0=hard replace).")
    ap.add_argument("--out", required=True, help="Output directory")
    args = ap.parse_args()

    model = load_model(args.ckpt)

    # ----- data -----
    pairs = TASKS[args.task](n=args.pairs)
    mapped = to_token_ids(pairs, tokenize, vocab_lookup)

    # pad to same length
    toks_clean   = torch.nn.utils.rnn.pad_sequence([torch.tensor(m[0], dtype=torch.long) for m in mapped], batch_first=True)
    toks_corrupt = torch.nn.utils.rnn.pad_sequence([torch.tensor(m[1], dtype=torch.long) for m in mapped], batch_first=True)
    target_id, distract_id = mapped[0][2], mapped[0][3]

    # ----- patch config -----
    positions = "last" if args.final_pos_only else "last"  # default to last; easy and robust
    eps = None if (args.eps is None) else float(args.eps)
    # NOTE: set --eps 0.0 to do hard-replace at the masked positions; default 0.1 recommended.

    # ----- run -----
    if args.metric == "kl":
        gains = rank_layers_by_gain_kl(
            model,
            toks_clean,
            toks_corrupt,
            n_layers=DEFAULT_CFG["n_layers"],
            positions=positions,
            eps=eps,
        )
    else:
        gains = rank_layers_by_gain(
            model,
            toks_clean,
            toks_corrupt,
            target_id,
            distract_id,
            n_layers=DEFAULT_CFG["n_layers"],
            positions=positions,
            eps=eps,
        )

    # ----- save -----
    out_dir = Path(args.out); out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "layer_gains.json", "w") as f:
        json.dump({
            "gains": gains,
            "metric": args.metric,
            "pairs": args.pairs,
            "positions": "last",
            "eps": eps,
            "task": args.task
        }, f, indent=2)

    # bars in layer order
    vals = [g for (_, g) in sorted(gains, key=lambda x: x[0])]
    layer_importance_bar(vals, title=f"Layer gains ({args.metric})", savepath=str(out_dir / "layer_gains.png"))
    print("[OK] Saved:", out_dir / "layer_gains.png")


if __name__ == "__main__":
    main()
