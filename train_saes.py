#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Train one Sparse Autoencoder (SAE) per layer by combining all shards of that layer
into a single dataset (memory-mapped), then fitting one SAE per layer.

Expects shards like:
  sae_data/layer{L}_part0.npy, layer{L}_part1.npy, ...

Outputs per layer (under --out_dir):
  - sae_layer{L}.pth
  - sae_layer{L}.json
  - sae_layer{L}_losses.npz
"""

import os, json, argparse, glob
from typing import Dict, Tuple, List
import numpy as np
import torch
import torch.nn as nn, torch.optim as optim

# --- Prefer project SAE/trainer if present ---
from sparse_auto_encoder import SparseAutoencoder  # your project file

def train_sae(
    data: np.ndarray,
    sae: SparseAutoencoder,
    model_prefix: str,
    epochs: int = 500,
    batch_size: int = 64,
    lr: float = 5e-4,
    weight_decay: float = 1e-6,
    train_losses=None,
    val_losses=None,
    device: str = "cuda",
    patience: int = 10,
    val_frac: float = 0.1,
    seed: int = 42,
):
    """
    Streaming trainer over a (memmapped) numpy array. Avoids materializing the whole
    train/val tensors on GPU. Slices small batches, casts to fp32, moves to device.
    """
    if train_losses is None: train_losses = []
    if val_losses is None: val_losses = []

    torch.manual_seed(seed); np.random.seed(seed)
    device = torch.device(device if torch.cuda.is_available() and device.startswith("cuda") else "cpu")
    sae.to(device)

    N = data.shape[0]
    idx = np.random.permutation(N)
    n_val = max(1, int(N * val_frac))
    val_idx = idx[:n_val]
    tr_idx  = idx[n_val:]

    opt = optim.AdamW(sae.parameters(), lr=lr, weight_decay=weight_decay)
    mse = nn.MSELoss()

    best_val = float("inf"); best_state = None; no_improve = 0

    def epoch_pass(idxs: np.ndarray, train: bool) -> float:
        sae.train() if train else sae.eval()
        losses = []
        # iterate in batches over index array; DO NOT pre-materialize tensors
        for s in range(0, idxs.shape[0], batch_size):
            bidx = idxs[s : s + batch_size]
            # slice from memmap -> fp32 -> torch -> device
            xb_np = data[bidx].astype(np.float32, copy=False)
            xb = torch.from_numpy(xb_np)
            if device.type == "cuda":
                xb = xb.pin_memory().to(device, non_blocking=True)
            else:
                xb = xb.to(device)

            if train:
                opt.zero_grad(set_to_none=True)
                with torch.set_grad_enabled(True):
                    x_hat, _ = sae(xb)
                    loss = mse(x_hat, xb)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(sae.parameters(), 1.0)
                    opt.step()
            else:
                with torch.no_grad():
                    x_hat, _ = sae(xb)
                    loss = mse(x_hat, xb)

            losses.append(loss.item())
        return float(np.mean(losses)) if losses else float("nan")

    for ep in range(1, epochs + 1):
        tr = epoch_pass(tr_idx, True)
        va = epoch_pass(val_idx, False)
        train_losses.append(tr); val_losses.append(va)

        if va < best_val - 1e-6:
            best_val, best_state, no_improve = va, {k: v.detach().cpu().clone() for k, v in sae.state_dict().items()}, 0
        else:
            no_improve += 1

        if ep % max(1, epochs // 10) == 0 or ep <= 3:
            print(f"[{model_prefix}] Ep {ep:04d} | train {tr:.6f} | val {va:.6f}", flush=True)

        if no_improve >= patience:
            print(f"[{model_prefix}] Early stopping at epoch {ep}. Best val={best_val:.6f}", flush=True)
            break

    torch.save(best_state if best_state is not None else sae.state_dict(), f"{model_prefix}.pth")
    return train_losses, val_losses


# ---------- Utils ----------
def find_shards(layer: int, data_dir: str) -> List[str]:
    return sorted(glob.glob(os.path.join(data_dir, f"layer{layer}_part*.npy")))

def build_memmap_from_shards(shard_paths: List[str], out_path: str) -> Tuple[np.memmap, int, int]:
    """Concatenate shards into one memmap (float16 on disk), return memmap and shape."""
    if not shard_paths:
        raise FileNotFoundError("No shard paths provided")

    # Inspect first shard for dimension
    first = np.load(shard_paths[0], mmap_mode="r")  # returns memmap, not a context manager
    d = int(first.shape[1])

    # Count total rows and validate dims
    total_rows = 0
    for sp in shard_paths:
        arr = np.load(sp, mmap_mode="r")  # NO 'with' here
        if arr.shape[1] != d:
            raise ValueError(f"Shard {sp} has mismatched dim {arr.shape[1]} (expected {d}).")
        total_rows += int(arr.shape[0])

    # Create destination memmap (compact on disk; we'll cast to fp32 during training)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    X_all = np.memmap(out_path, mode="w+", dtype=np.float16, shape=(total_rows, d))

    # Fill sequentially
    cursor = 0
    for sp in shard_paths:
        arr = np.load(sp, mmap_mode="r")  # memmap view
        n = int(arr.shape[0])
        X_all[cursor:cursor + n, :] = arr  # implicit fp16 cast if shards are fp32
        cursor += n

    X_all.flush()
    # Reopen read-only
    X_all_ro = np.memmap(out_path, mode="r", dtype=np.float16, shape=(total_rows, d))
    return X_all_ro, total_rows, d

def capacity_for_layer(layer: int, input_dim: int) -> int:
    """
    Depth-aware capacity:
      layers 1-2  -> 3x
      layers 3-5  -> 4x
      layers 6-8  -> 5x
    """
    if layer <= 2:   mult = 3
    elif layer <= 5: mult = 4
    else:            mult = 5
    return mult * input_dim

# ---------- Main ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", type=str, default="sae_data")
    ap.add_argument("--out_dir", type=str, default="sae_models")
    ap.add_argument("--layers", type=str, default="1,2,3,4,5,6,7,8")
    ap.add_argument("--top_k", type=int, default=50, help="Fixed top-k; hidden_dim adapts by depth.")
    ap.add_argument("--epochs", type=int, default=500)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--weight_decay", type=float, default=1e-6)
    ap.add_argument("--patience", type=int, default=10)
    ap.add_argument("--val_frac", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--tmp_dir", type=str, default=None,
                    help="Where to place the combined memmap per layer. Defaults to out_dir.")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    if args.tmp_dir is None:
        args.tmp_dir = args.out_dir
    os.makedirs(args.tmp_dir, exist_ok=True)

    layers = [int(x.strip()) for x in args.layers.split(",") if x.strip()]
    print(f"Using project trainer: {USE_PROJECT_TRAINER}")
    print(f"Training SAEs for layers: {layers}")

    for layer in layers:
        print(f"\n— Layer {layer} —")
        shard_paths = find_shards(layer, args.data_dir)
        if not shard_paths:
            print(f"  ⚠️ No shards found for layer {layer} in {args.data_dir}; skipping.")
            continue

        # Build or load a single memmap for the layer (combine all shards)
        memmap_path = os.path.join(args.tmp_dir, f"layer{layer}_ALL.memmap")

        if os.path.exists(memmap_path):
            print(f"  Found existing memmap for layer {layer} at {memmap_path}; loading...")
            # Reconstruct shape info from shards (required to map correctly)
            first = np.load(shard_paths[0], mmap_mode="r")
            d = int(first.shape[1])
            total_rows = sum(np.load(sp, mmap_mode="r").shape[0] for sp in shard_paths)
            X_all = np.memmap(memmap_path, mode="r", dtype=np.float16, shape=(total_rows, d))
            input_dim = d
            n_rows = total_rows
        else:
            X_all, n_rows, input_dim = build_memmap_from_shards(shard_paths, memmap_path)
            print(f"  Combined {len(shard_paths)} shard(s) -> {X_all.shape} (memmap: {memmap_path})")

        # Depth-aware capacity
        hidden_dim = capacity_for_layer(layer, input_dim)
        top_k = min(args.top_k, hidden_dim) if args.top_k <= hidden_dim else max(1, hidden_dim // 20)
        if top_k != args.top_k:
            print(f"  top_k clipped to {top_k} (≤ hidden_dim)")

        # Build SAE and train on the full combined dataset
        sae = SparseAutoencoder(input_dim=input_dim, hidden_dim=hidden_dim, top_k=top_k)
        model_prefix = os.path.join(args.out_dir, f"sae_layer{layer}")

        # NOTE: train_sae casts to float32 tensors inside, so on-disk fp16 is fine.
        tl, vl = train_sae(
            X_all, sae,
            model_prefix=model_prefix,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            weight_decay=args.weight_decay,
            train_losses=[],
            val_losses=[],
            device=args.device,
            patience=args.patience,
            val_frac=args.val_frac,
            seed=args.seed,
        )

        # Save losses & metadata
        np.savez(os.path.join(args.out_dir, f"sae_layer{layer}_losses.npz"),
                 train_losses=np.array(tl, dtype=np.float32),
                 val_losses=np.array(vl, dtype=np.float32))
        meta: Dict = {
            "layer": layer,
            "n_tokens": int(n_rows),
            "input_dim": input_dim,
            "hidden_dim": hidden_dim,
            "top_k": top_k,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "patience": args.patience,
            "val_frac": args.val_frac,
            "seed": args.seed,
            "device": args.device,
            "project_trainer_used": USE_PROJECT_TRAINER,
            "final_train_loss": float(tl[-1]) if tl else None,
            "best_val_loss": float(np.min(vl)) if vl else None,
            "model_path": f"{model_prefix}.pth",
            "num_shards": len(shard_paths),
            "memmap_path": memmap_path,
        }
        with open(os.path.join(args.out_dir, f"sae_layer{layer}.json"), "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)
        print(f"  Saved model & metadata for layer {layer}")

    print("\nAll requested layers processed.")

if __name__ == "__main__":
    main()
