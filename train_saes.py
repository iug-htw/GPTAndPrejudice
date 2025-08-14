#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Train one Sparse Autoencoder (SAE) per layer for previously-extracted embeddings.

Expected data:
  sae_data/layer1_embeddings.npy
  ...
  sae_data/layer8_embeddings.npy

Outputs per layer (under sae_models/):
  - sae_layer{i}.pth       (state_dict)
  - sae_layer{i}.json      (metadata: dims, losses, hyperparams)
  - sae_layer{i}_losses.npz (arrays: train_losses, val_losses)

Usage:
  python train_all_saes.py \
      --data_dir sae_data \
      --layers 1,2,3,4,5,6,7,8 \
      --hidden_dim 3072 \
      --top_k 50 \
      --epochs 500 \
      --batch_size 64 \
      --lr 5e-4 \
      --weight_decay 1e-6 \
      --patience 10 \
      --val_frac 0.1 \
      --seed 42 \
      --device cuda
"""

import os, json, argparse, glob
from typing import Tuple, Dict
import numpy as np
import torch

USE_PROJECT_TRAINER = True
try:
    from sparse_auto_encoder import SparseAutoencoder, train_sae
except Exception:
    USE_PROJECT_TRAINER = False

# -----------------------------
# Minimal fallback trainer (used only if import fails)
# -----------------------------
if not USE_PROJECT_TRAINER:
    import torch.nn as nn, torch.optim as optim
    class SparseAutoencoder(nn.Module):
        """
        Simple linear SAE: x -> ReLU(W_enc x + b) -> top-k -> W_dec h + b
        """
        def __init__(self, input_dim: int, hidden_dim: int, top_k: int):
            super().__init__()
            self.input_dim = input_dim
            self.hidden_dim = hidden_dim
            self.top_k = top_k

            self.encoder = nn.Linear(input_dim, hidden_dim, bias=True)
            self.decoder = nn.Linear(hidden_dim, input_dim, bias=True)
            self.relu = nn.ReLU(inplace=True)

        def topk_mask(self, h: torch.Tensor, k: int) -> torch.Tensor:
            if k >= h.shape[-1]:
                return torch.ones_like(h)
            # keep top-k per row
            vals, idx = torch.topk(h, k, dim=-1)
            mask = torch.zeros_like(h)
            mask.scatter_(dim=-1, index=idx, src=torch.ones_like(vals))
            return mask

        def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
            h = self.relu(self.encoder(x))
            mask = self.topk_mask(h, self.top_k)
            h_sparse = h * mask
            x_hat = self.decoder(h_sparse)
            return x_hat, h_sparse

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
        Minimal trainer with early stopping on val MSE. Splits 'data' into train/val.
        Saves best state_dict to f"{model_prefix}.pth".
        """
        if train_losses is None: train_losses = []
        if val_losses is None: val_losses = []

        torch.manual_seed(seed)
        np.random.seed(seed)

        device = torch.device(device if torch.cuda.is_available() and device.startswith("cuda") else "cpu")
        sae.to(device)

        # split
        N = data.shape[0]
        idx = np.random.permutation(N)
        n_val = max(1, int(N * val_frac))
        val_idx, tr_idx = idx[:n_val], idx[n_val:]

        x_tr = torch.tensor(data[tr_idx], dtype=torch.float32, device=device)
        x_val = torch.tensor(data[val_idx], dtype=torch.float32, device=device)

        opt = optim.AdamW(sae.parameters(), lr=lr, weight_decay=weight_decay)
        mse = nn.MSELoss()

        best_val = float("inf")
        best_state = None
        no_improve = 0

        def run_epoch(x, train=True):
            losses = []
            if train:
                sae.train()
            else:
                sae.eval()

            # simple mini-batching
            for i in range(0, x.shape[0], batch_size):
                xb = x[i:i+batch_size]
                if train:
                    opt.zero_grad()
                with torch.set_grad_enabled(train):
                    x_hat, _ = sae(xb)
                    loss = mse(x_hat, xb)
                    if train:
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(sae.parameters(), 1.0)
                        opt.step()
                losses.append(loss.item())
            return float(np.mean(losses))

        for ep in range(1, epochs + 1):
            tr_loss = run_epoch(x_tr, train=True)
            va_loss = run_epoch(x_val, train=False)

            train_losses.append(tr_loss)
            val_losses.append(va_loss)

            if va_loss < best_val - 1e-6:
                best_val = va_loss
                best_state = {k: v.detach().cpu().clone() for k, v in sae.state_dict().items()}
                no_improve = 0
            else:
                no_improve += 1

            if ep % max(1, epochs // 10) == 0 or ep == 1:
                print(f"[{model_prefix}] Ep {ep:04d}: train {tr_loss:.6f} | val {va_loss:.6f}")

            if no_improve >= patience:
                print(f"[{model_prefix}] Early stopping at epoch {ep}. Best val={best_val:.6f}")
                break

        # save best
        if best_state is not None:
            torch.save(best_state, f"{model_prefix}.pth")
        else:
            torch.save(sae.state_dict(), f"{model_prefix}.pth")

        return train_losses, val_losses


# -----------------------------
# Utilities
# -----------------------------

def load_shards(layer: int, data_dir: str):
    return sorted(glob.glob(os.path.join(data_dir, f"layer{layer}_part*.npy")))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="sae_data")
    parser.add_argument("--out_dir", type=str, default="sae_models")
    parser.add_argument("--layers", type=str, default="1,2,3,4,5,6,7,8")
    parser.add_argument("--hidden_dim", type=int, default=3072)
    parser.add_argument("--top_k", type=int, default=50)
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-6)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--val_frac", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    layers = [int(x.strip()) for x in args.layers.split(",") if x.strip()]
    print(f"Using project trainer: {USE_PROJECT_TRAINER}")
    print(f"Training SAEs for layers: {layers}")

    for layer in layers:
        print(f"\n— Layer {layer} —")
        shard_paths = load_shards(layer, args.data_dir)
        if not shard_paths:
            print(f"  ⚠️ No shards found for layer {layer} in {args.data_dir}; skipping.")
            continue

        # Infer dims from first shard
        first = np.load(shard_paths[0], mmap_mode="r")
        input_dim = int(first.shape[1])
        print(f"  Found {len(shard_paths)} shard(s); dim={input_dim}")

        # Build SAE once; warm-start across shards
        hidden_dim = args.hidden_dim
        top_k = min(args.top_k, hidden_dim) if args.top_k <= hidden_dim else max(1, hidden_dim // 20)
        if top_k != args.top_k:
            print(f"  top_k clipped to {top_k} (must be ≤ hidden_dim)")
        sae = SparseAutoencoder(input_dim=input_dim, hidden_dim=hidden_dim, top_k=top_k)

        total_tokens = 0
        all_train_losses, all_val_losses = [], []

        for si, shard_path in enumerate(shard_paths):
            X = np.load(shard_path, mmap_mode="r")
            total_tokens += int(X.shape[0])
            print(f"  -> shard {si+1}/{len(shard_paths)}: {X.shape} ({shard_path})")

            # Option: fewer epochs per later shards (simple anneal)
            shard_epochs = args.epochs if si == 0 else max(1, args.epochs // 3)

            tl, vl = train_sae(
                X, sae,
                model_prefix=os.path.join(args.out_dir, f"sae_layer{layer}"),
                epochs=shard_epochs,
                batch_size=args.batch_size,
                lr=args.lr,
                weight_decay=args.weight_decay,
                train_losses=[],
                val_losses=[],
                device=args.device,
                patience=args.patience,
                val_frac=args.val_frac,
                seed=args.seed + si,  # vary seed per shard split
            )
            all_train_losses += tl
            all_val_losses += vl

        # Save final checkpoint (overwrites with final state)
        model_prefix = os.path.join(args.out_dir, f"sae_layer{layer}")
        torch.save(sae.state_dict(), f"{model_prefix}.pth")

        # Save losses & metadata
        np.savez(os.path.join(args.out_dir, f"sae_layer{layer}_losses.npz"),
                 train_losses=np.array(all_train_losses, dtype=np.float32),
                 val_losses=np.array(all_val_losses, dtype=np.float32))
        meta: Dict = {
            "layer": layer,
            "n_tokens": int(total_tokens),
            "input_dim": input_dim,
            "hidden_dim": hidden_dim,
            "top_k": top_k,
            "epochs_first_shard": args.epochs,
            "epochs_later_shards": max(1, args.epochs // 3),
            "batch_size": args.batch_size,
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "patience": args.patience,
            "val_frac": args.val_frac,
            "seed_base": args.seed,
            "device": args.device,
            "project_trainer_used": USE_PROJECT_TRAINER,
            "final_train_loss": float(all_train_losses[-1]) if all_train_losses else None,
            "best_val_loss": float(np.min(all_val_losses)) if all_val_losses else None,
            "model_path": f"{model_prefix}.pth",
            "num_shards": len(shard_paths),
        }
        with open(os.path.join(args.out_dir, f"sae_layer{layer}.json"), "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)
        print(f"  Saved model & metadata for layer {layer}")

    print("\nAll requested layers processed.")

if __name__ == "__main__":
    main()
