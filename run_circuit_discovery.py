
import argparse, os, json, csv, torch
from austen_circuits.data.ioi_data import build_ioi_dataset
from austen_circuits.utils.tensors import pad_and_stack
from austen_circuits.algorithms.metrics import mean_io_logit_diff
from austen_circuits.algorithms.circuit_discovery import (
    run_model, layerwise_resid_patching, heuristic_name_mover_heads
)

def load_model(ckpt_path: str, device: str = "cpu"):
    blob = torch.load(ckpt_path, weights_only=True, map_location=torch.device('cpu'))
    if isinstance(blob, dict) and "cfg" in blob and "state_dict" in blob:
        from gpt_model import GPTModel
        model = GPTModel(blob["cfg"]).to(device)
        model.load_state_dict(blob["state_dict"], strict=True)
    else:
        # fallback: assume the checkpoint is a plain state dict and there is a default CFG inside gpt_model.py
        from gpt_model import GPTModel, DEFAULT_CFG
        model = GPTModel(DEFAULT_CFG).to(device)
        model.load_state_dict(blob["model_state_dict"])
        model.to(device)
        model.eval()
    return model

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="Path to model checkpoint (.pth)")
    ap.add_argument("--n", type=int, default=256)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--outdir", default="out")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    model = load_model(args.ckpt, args.device)

    data = build_ioi_dataset(n=args.n, seed=0)

    # Pad & stack to tensors
    pad_id = getattr(getattr(model, "cfg", {}), "get", lambda k, d: 50256)("pad_token_id", 50256) if hasattr(model, "cfg") else 50256
    ids_clean   = pad_and_stack(data.token_ids_clean, pad_id=pad_id).to(args.device)
    ids_corrupt = pad_and_stack(data.token_ids_corrupt, pad_id=pad_id).to(args.device)

    # Clean/corrupt runs
    clean_logits, clean_cache, clean_attn = run_model(model, ids_clean,  collect_attn=True)
    corrupt_logits, corrupt_cache, _      = run_model(model, ids_corrupt, collect_attn=False)

    clean_score   = mean_io_logit_diff(clean_logits,   data.target_positions_clean,   data.io_token_ids, data.s_token_ids)
    corrupt_score = mean_io_logit_diff(corrupt_logits, data.target_positions_corrupt, data.io_token_ids, data.s_token_ids)
    gap = clean_score - corrupt_score

    print(f"Clean mean Δlogit(IO−S):   {clean_score:.4f}")
    print(f"Corrupt mean Δlogit(IO−S): {corrupt_score:.4f}")
    print(f"Gap (clean − corrupt):     {gap:.4f}")

    # Residual patching (graceful fallback if unsupported)
    patch_results = []
    try:
        patch_results = layerwise_resid_patching(
            model, clean_cache, ids_corrupt,
            data.target_positions_corrupt, data.io_token_ids, data.s_token_ids,
            mode="pre"
        )
    except Exception as e:
        print(f"[WARN] Residual patching not supported by this model/hook API: {e}")

    # Save patch results
    with open(os.path.join(args.outdir, "layer_patching_pre.csv"), "w", newline="") as f:
        w = csv.writer(f); w.writerow(["layer", "mode", "restored_mean"])
        for r in patch_results:
            w.writerow([r.layer, r.mode, f"{r.restored_mean:.6f}"])

    # Heuristic head scores
    head_scores = []
    if clean_attn:
        head_scores = heuristic_name_mover_heads(clean_attn, data.target_positions_clean, data.io_token_ids, ids_clean)
    with open(os.path.join(args.outdir, "heuristic_name_movers.csv"), "w", newline="") as f:
        w = csv.writer(f); w.writerow(["layer", "head", "mean_attention_mass_on_IO"])
        for L, h, s in head_scores[:100]:
            w.writerow([L, h, f"{s:.6f}"])

    with open(os.path.join(args.outdir, "summary.json"), "w") as f:
        json.dump({
            "clean_mean_logit_diff": clean_score,
            "corrupt_mean_logit_diff": corrupt_score,
            "gap": gap,
            "top_layer_restorations": sorted(
                [{"layer": r.layer, "restored_mean": r.restored_mean} for r in patch_results],
                key=lambda x: x["restored_mean"], reverse=True)[:10],
            "top_heuristic_heads": head_scores[:10]
        }, f, indent=2)

    print(f"\nSaved to {args.outdir}/layer_patching_pre.csv, {args.outdir}/heuristic_name_movers.csv, and summary.json")

if __name__ == "__main__":
    main()
