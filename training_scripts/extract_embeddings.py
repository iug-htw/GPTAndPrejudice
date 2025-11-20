"""
Build sharded, layer-wise embedding datasets for Sparse Autoencoder (SAE) training.

Overview
--------
This utility reads the cleaned training + validation corpora, splits text into
medium-length sentences, runs them through the custom GPT model, and extracts
token-level hidden states for selected transformer layers (1-based indexing).
Embeddings are streamed to disk in **shards** to avoid exhausting host memory.

Why sharding?
-------------
Hidden-state arrays are large (tokens × d_model). To keep memory bounded, the
script accumulates up to `--shard_tokens` tokens per layer, writes a shard
`layer{L}_part{K}.npy`, clears the buffer, then continues.

Outputs (per layer L)
---------------------
- {out_dir}/layerL_part0.npy
- {out_dir}/layerL_part1.npy
  ...
Each .npy has shape [N_tokens_in_shard, d_model] and dtype `--save_dtype`.

Typical usage
-------------
python extract_embeddings.py \
  --train_file datasets/train_text_data.txt \
  --val_file   datasets/val_text_data.txt \
  --ckpt checkpoints/best_model.pth \
  --layers 1,2,3,4,5,6,7,8 \
  --out_dir sae_data \
  --shard_tokens 250000 \
  --save_dtype float16
"""

import os, re, sys, argparse
from typing import Dict, List
import numpy as np
import torch
import tiktoken

# Import your model definition
# Assumes gpt_model.py exposes GPTModel and a config dict "GPT_CONFIG_124M" (adjust if yours differ)
from gpt_model import GPTModel

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
GPT_CONFIG = {
    "vocab_size": 50257,
    "context_length": 256,
    "emb_dim": 896,
    "n_heads": 14,
    "n_layers": 8,
    "drop_rate": 0.2,
    "qkv_bias": True,
    "device": DEVICE,
}


# -----------------------------
# Configuration / CLI
# -----------------------------
def parse_args():
    p = argparse.ArgumentParser(description="Build SAE embedding dataset (layers 1..8).")
    p.add_argument("--train_file", type=str, default="train_text_data.txt",
                   help="Path to the training text file.")
    p.add_argument("--val_file", type=str, default="val_text_data.txt",
                   help="Path to the validation text file.")
    p.add_argument("--ckpt", type=str, default="checkpoints/best_model.pth",
                   help="Path to the trained GPT model checkpoint.")
    p.add_argument("--layers", type=str, default="1,2,3,4,5,6,7,8",
                   help="Comma-separated 1-based layer indices to extract (default: 1..8).")
    p.add_argument("--min_words", type=int, default=5,
                   help="Minimum words per sentence to keep.")
    p.add_argument("--max_words", type=int, default=60,
                   help="Maximum words per sentence to keep.")
    p.add_argument("--out_dir", type=str, default="sae_data",
                   help="Directory to write .npy files into.")
    p.add_argument("--shard_tokens", type=int, default=250_000,
                   help="Flush to disk after this many tokens per layer.")
    p.add_argument("--save_dtype", type=str, default="float16",
                   choices=["float16","float32","bfloat16"])
    return p.parse_args()

DTYPE_MAP = {"float16": np.float16, "float32": np.float32, "bfloat16": np.float16}  # np has no bf16

def flush_shard(layer, buf_list, out_dir, shard_idx, save_dtype):
    if not buf_list: return shard_idx
    arr = np.vstack(buf_list).astype(DTYPE_MAP[save_dtype], copy=False)
    out_path = os.path.join(out_dir, f"layer{layer}_part{shard_idx}.npy")
    np.save(out_path, arr)
    print(f"  [layer {layer}] wrote shard {shard_idx}: {arr.shape} -> {out_path}", flush=True)
    buf_list.clear()
    return shard_idx + 1

# -----------------------------
# Data loading / sentence split
# -----------------------------
SENT_SPLIT_REGEX = r"(?<=[.!?])\s+"

def load_full_text(train_file_path: str, val_file_path: str) -> str:
    with open(train_file_path, "r", encoding="utf-8") as f:
        train_data = f.read()
    with open(val_file_path, "r", encoding="utf-8") as f:
        val_data = f.read()
    full_text = train_data + val_data
    return "\n".join(full_text.split("<|endoftext|>"))

def split_and_filter_sentences(full_text: str, min_words: int = 5, max_words: int = 60) -> List[str]:
    sentences = re.split(SENT_SPLIT_REGEX, full_text)
    return [s.strip() for s in sentences if s and min_words < len(s.split()) < max_words]

# -----------------------------
# Tokenization helper
# -----------------------------
def build_tokenizer():
    return tiktoken.get_encoding("gpt2")

def text_to_token_ids(text: str, tokenizer) -> torch.Tensor:
    encoded = tokenizer.encode(text, allowed_special={'<|endoftext|>'})
    if len(encoded) == 0:
        return None
    return torch.tensor(encoded, dtype=torch.long).unsqueeze(0)  # [1, T]

@torch.no_grad()
def get_token_embeddings_for_layers(
    text: str,
    model: GPTModel,
    tokenizer,
    layers_1based: List[int]
) -> Dict[int, np.ndarray]:
    """
    Returns dict {layer_index_1based: np.ndarray of shape [T, d_model]} for each requested layer.
    """
    input_ids = text_to_token_ids(text, tokenizer)
    if input_ids is None:
        return {}  # skip empty
    model_device = next(model.parameters()).device
    input_ids = input_ids.to(model_device, non_blocking=True)

    if model_device.type == "cuda":
        ctx = torch.cuda.amp.autocast(enabled=False)
    else:
        # no autocast on CPU
        class _Dummy: 
            def __enter__(self): return None
            def __exit__(self, *a): return False
        ctx = _Dummy()

    with ctx:
        logits, hidden_states = model(input_ids, output_hidden_states=True)

        # hidden_states is expected to be a list/tuple of torch.Tensor with shape [1, T, d_model]
        out = {}
        max_idx = len(hidden_states)  # number of layers available (1-based in the user's convention)
        for l in layers_1based:
            if 1 <= l <= max_idx:
                # Convert to [T, d_model] on CPU numpy
                out[l] = hidden_states[l - 1].squeeze(0).detach().cpu().numpy()
            else:
                print(f"⚠️ Warning: requested layer {l} is out of range; model exposed {max_idx} hidden states.", file=sys.stderr, flush=True)
        return out


# -----------------------------
# Model loading
# -----------------------------
def load_model(ckpt_path: str) -> GPTModel:
    # Build model from your config and load weights
    model = GPTModel(GPT_CONFIG)
    checkpoint = torch.load(ckpt_path, weights_only=True, map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(DEVICE)
    model.eval()
    model = model.float()  # ensure fp32 weights

    return model

def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    layers_1based = [int(x.strip()) for x in args.layers.split(",") if x.strip()]

    print("• Loading text...", flush=True)
    full_text = load_full_text(args.train_file, args.val_file)

    print("• Splitting into sentences...", flush=True)
    sentences = split_and_filter_sentences(full_text, args.min_words, args.max_words)
    print(f"  Kept {len(sentences)} sentences.", flush=True)

    print("• Building tokenizer...", flush=True)
    tokenizer = build_tokenizer()

    print("• Loading model...", flush=True)
    model = load_model(args.ckpt)

    print("• Extracting token embeddings (streaming to shards)...", flush=True)
    total = len(sentences)
    log_every = max(1, total // 50)
    per_layer_buffers = {l: [] for l in layers_1based}
    per_layer_token_counts = {l: 0 for l in layers_1based}
    per_layer_shard_idx = {l: 0 for l in layers_1based}

    for i, sent in enumerate(sentences, 1):
        emb_dict = get_token_embeddings_for_layers(sent, model, tokenizer, layers_1based)
        for l in layers_1based:
            if l in emb_dict:
                per_layer_buffers[l].append(emb_dict[l])
                per_layer_token_counts[l] += emb_dict[l].shape[0]
                if per_layer_token_counts[l] >= args.shard_tokens:
                    per_layer_shard_idx[l] = flush_shard(
                        l, per_layer_buffers[l], args.out_dir, per_layer_shard_idx[l], args.save_dtype
                    )
                    per_layer_token_counts[l] = 0
        if i % log_every == 0 or i == total:
            print(f"  Processed {i}/{total} sentences", flush=True)

    # final flush
    for l in layers_1based:
        if per_layer_buffers[l]:
            per_layer_shard_idx[l] = flush_shard(
                l, per_layer_buffers[l], args.out_dir, per_layer_shard_idx[l], args.save_dtype
            )

    print("✓ Done (sharded).", flush=True)

if __name__ == "__main__":
    try:
        main()
    except Exception:
        import traceback; traceback.print_exc()
        sys.exit(1)
