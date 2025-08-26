
#!/usr/bin/env python3
"""
circuits_cli.py — run circuit discovery (layer-entry activation patching) and optional SAE feature injection.
Usage examples:
  python circuits_cli.py --cfg path/to/cfg.json --ckpt path/to/model.pt --task gender_swap --out out_dir
  python circuits_cli.py --cfg cfg.json --ckpt model.pt --task ioi --sae_ckpt sae.pt --sae_latent 1526 --sae_layer 6 --sae_scale 2.0
"""

import argparse
import os
from pathlib import Path
from typing import List, Tuple
import json

import torch
import torch.nn as nn

# Project imports (must exist in your repo)
from gpt_model import GPTModel
from circuits import rank_layers_by_gain, cache_clean, baseline_score, next_token_logit_diff
from circuit_tasks import (
    build_gender_swap_pairs,
    build_ioi_pairs,
    build_marriage_vs_wealth_pairs,
    build_emotion_vs_duty_pairs,
    to_token_ids,
)
from viz import layer_importance_bar

# Optional SAE
try:
    from sparse_auto_encoder import SparseAutoencoder
    SAE_AVAILABLE = True
except Exception:
    SAE_AVAILABLE = False

# ---------------- Tokenizer adapter ----------------
def try_load_tiktoken():
    try:
        import tiktoken
        return tiktoken.get_encoding("gpt2")
    except Exception:
        return None

def make_tokenize_with_tiktoken(enc):
    def tokenize(texts: List[str]) -> List[List[int]]:
        return [enc.encode(t) for t in texts]
    return tokenize

def make_vocab_lookup_with_tiktoken(enc):
    def vocab_lookup(token_str: str) -> int:
        # Return the first token id for the string (best-effort)
        ids = enc.encode(token_str)
        if len(ids) == 0:
            raise ValueError(f"String produced no tokens under tiktoken: {token_str!r}")
        return ids[0]
    return vocab_lookup

# Fallback (naive whitespace tokenizer; NOT aligned with GPT2!)
def naive_tokenize(texts: List[str]) -> List[List[int]]:
    vocab = {}
    next_id = 0
    out = []
    for t in texts:
        ids = []
        for w in t.split():
            if w not in vocab:
                vocab[w] = next_id
                next_id += 1
            ids.append(vocab[w])
        out.append(ids)
    return out

def naive_vocab_lookup(token_str: str) -> int:
    # This cannot match model vocab; kept only to avoid crashes if user wants to inspect flow without accuracy.
    raise RuntimeError("No tokenizer available. Please install `tiktoken` or provide your own tokenize/vocab_lookup.")

# ---------------- SAE intervention plan ----------------
class AddFeaturePlan:
    def __init__(self, L: int, delta_vec: torch.Tensor):
        self.L = L
        self.delta_vec = delta_vec  # [d_model]

    def maybe_replace_resid_pre(self, layer_idx: int, x: torch.Tensor) -> torch.Tensor:
        if layer_idx == self.L:
            # Broadcast delta to batch/seq
            return x + self.delta_vec.to(x.device).view(1, 1, -1)
        return x

    # No-ops for other hooks (kept for API compatibility)
    def maybe_replace_resid_post(self, layer_idx: int, x: torch.Tensor) -> torch.Tensor: return x
    def maybe_replace_head_z(self, layer_idx: int, z: torch.Tensor) -> torch.Tensor: return z
    def maybe_replace_mlp_out(self, layer_idx: int, h: torch.Tensor) -> torch.Tensor: return h

# ---------------- Tasks ----------------
TASKS = {
    "gender_swap": build_gender_swap_pairs,
    "ioi": build_ioi_pairs,
    "marriage_vs_wealth": build_marriage_vs_wealth_pairs,
    "emotion_vs_duty": build_emotion_vs_duty_pairs,
}

def main():
    parser = argparse.ArgumentParser(description="Circuit discovery CLI (layer patching + optional SAE feature injection)")
    parser.add_argument("--cfg", type=str, required=True, help="Path to model cfg.json (must include vocab_size, emb_dim, context_length, n_layers, drop_rate, etc.)")
    parser.add_argument("--ckpt", type=str, required=True, help="Path to model checkpoint (.pt/.pth)")
    parser.add_argument("--task", type=str, required=True, choices=list(TASKS.keys()), help="Which minimal-pair task to run")
    parser.add_argument("--pairs", type=int, default=64, help="How many pairs to sample")
    parser.add_argument("--device", type=str, default="cpu" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--out", type=str, required=True, help="Output directory")
    # SAE options
    parser.add_argument("--sae_ckpt", type=str, default=None, help="Path to SAE checkpoint")
    parser.add_argument("--sae_latent", type=int, default=None, help="Latent ID to inject (e.g., 'marriage' latent index)")
    parser.add_argument("--sae_layer", type=int, default=None, help="Layer index to inject at (resid_pre[L])")
    parser.add_argument("--sae_scale", type=float, default=2.0, help="Scale of decoded latent vector")
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- Load model ----
    with open(args.cfg, "r") as f:
        cfg = json.load(f)
    device = torch.device(args.device)
    model = GPTModel(cfg).to(device).eval()
    sd = torch.load(args.ckpt, map_location=device)
    if isinstance(sd, dict) and "state_dict" in sd:
        sd = sd["state_dict"]
    model.load_state_dict(sd, strict=False)

    # ---- Build pairs ----
    pairs_fn = TASKS[args.task]
    pairs = pairs_fn(n=args.pairs)

    # ---- Tokenizer ----
    enc = try_load_tiktoken()
    if enc is not None:
        tokenize = make_tokenize_with_tiktoken(enc)
        vocab_lookup = make_vocab_lookup_with_tiktoken(enc)
        tokenizer_name = "tiktoken:gpt2"
    else:
        tokenize = naive_tokenize
        vocab_lookup = naive_vocab_lookup
        tokenizer_name = "naive (not aligned!)"

    # ---- Token IDs ----
    try:
        mapped = to_token_ids(pairs, tokenize, vocab_lookup)
    except Exception as e:
        raise SystemExit(f"[Tokenizer error] {e}\nHint: install `tiktoken` (`pip install tiktoken`) to use GPT-2 tokenization.")

    toks_clean   = torch.nn.utils.rnn.pad_sequence([torch.tensor(m[0], dtype=torch.long) for m in mapped], batch_first=True).to(device)
    toks_corrupt = torch.nn.utils.rnn.pad_sequence([torch.tensor(m[1], dtype=torch.long) for m in mapped], batch_first=True).to(device)
    target_id    = mapped[0][2]
    distract_id  = mapped[0][3]

    # ---- Layer ranking via activation patching ----
    gains = rank_layers_by_gain(
        model,
        toks_clean,
        toks_corrupt,
        target_id,
        distract_id,
        n_layers=cfg["n_layers"],
    )

    # Save results
    gains_sorted_by_layer = sorted(gains, key=lambda x: x[0])
    with open(out_dir / "layer_gains.json", "w") as f:
        json.dump({"gains": gains, "tokenizer": tokenizer_name}, f, indent=2)

    # Plot
    vals = [g for (_, g) in gains_sorted_by_layer]
    layer_importance_bar(vals, title=f"Layer patch gains — {args.task}", savepath=str(out_dir / "layer_gains.png"))
    print("[OK] Saved layer patch gains to", out_dir / "layer_gains.png")

    # ---- Optional: SAE feature injection ----
    if args.sae_ckpt and (args.sae_latent is not None) and (args.sae_layer is not None):
        if not SAE_AVAILABLE:
            raise SystemExit("SAE module not available. Ensure sparse_auto_encoder.py is in PYTHONPATH.")
        SAE = SparseAutoencoder(input_dim=cfg["emb_dim"]).to(device).eval()
        sae_sd = torch.load(args.sae_ckpt, map_location=device)
        if isinstance(sae_sd, dict) and "state_dict" in sae_sd:
            sae_sd = sae_sd["state_dict"]
        SAE.load_state_dict(sae_sd, strict=False)

        # Decode latent to residual vector
        delta_vec = SAE.decode_single_latent(args.sae_latent, scale=args.sae_scale, device=device)  # [d_model]

        # Baseline on corrupt
        base = baseline_score(model, toks_corrupt, target_id, distract_id)

        # Inject at specified layer
        plan = AddFeaturePlan(args.sae_layer, delta_vec)
        with torch.no_grad():
            logits = model(toks_corrupt, enable_cache=False, intervention_plan=plan)
            if not torch.is_tensor(logits):
                logits = logits[0]
        effect = next_token_logit_diff(logits, target_id, distract_id) - base
        with open(out_dir / "sae_injection_result.json", "w") as f:
            json.dump({
                "task": args.task,
                "sae_latent": args.sae_latent,
                "sae_layer": args.sae_layer,
                "sae_scale": args.sae_scale,
                "effect_logit_diff_delta": effect,
            }, f, indent=2)
        print(f"[OK] SAE injection effect (Δ logit diff): {effect:.4f}  -> saved to sae_injection_result.json")
    else:
        print("[Info] SAE injection skipped (provide --sae_ckpt, --sae_latent, --sae_layer to enable).")

if __name__ == "__main__":
    main()
